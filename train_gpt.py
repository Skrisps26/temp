"""
Parameter Golf: Competitive train_gpt.py
11L d=512 Transformer with XSA4, LeakyReLU(0.5)², BigramHash, VE128,
U-Net Skips, EMA+SWA, int8+zstd, SmearGate, and Legal Score-First TTT.
"""
from __future__ import annotations
import copy, glob, io, math, os, pickle, random, subprocess, sys, time, uuid
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ── compression ───────────────────────────────────────────────────────────────
try:
    import zstandard as _zstd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "zstandard"])
    import zstandard as _zstd
_ZL = int(os.environ.get("ZSTD_LEVEL", "22"))
compress_bytes   = lambda b: _zstd.ZstdCompressor(level=_ZL).compress(b)
decompress_bytes = lambda b: _zstd.ZstdDecompressor().decompress(b)
COMPRESSOR = f"zstd-{_ZL}"

try:
    from flash_attn import flash_attn_func
    HAS_FA = True
except ImportError:
    HAS_FA = False

# ── hyperparameters ───────────────────────────────────────────────────────────
class H:
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))

    iterations         = int(os.environ.get("ITERATIONS", 9000))
    warmdown_iters     = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps       = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524288))
    train_seq_len      = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    val_loss_every     = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size     = int(os.environ.get("VAL_BATCH_SIZE", 524288))
    train_log_every    = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    vocab_size    = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers    = int(os.environ.get("NUM_LAYERS", 9))
    model_dim     = int(os.environ.get("MODEL_DIM", 512))
    num_heads     = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads  = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult      = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = True
    rope_base     = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init  = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # XSA on last N layers
    xsa_last_n    = int(os.environ.get("XSA_LAST_N", 4))
    ln_scale      = int(os.environ.get("LN_SCALE", 1))

    # BigramHash
    bigram_vocab  = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim    = int(os.environ.get("BIGRAM_DIM", 128))

    # SmearGate
    smear_enabled = int(os.environ.get("SMEAR_ENABLED", 1))

    # VE128
    ve_enabled    = int(os.environ.get("VE_ENABLED", 1))
    ve_dim        = int(os.environ.get("VE_DIM", 128))
    ve_layers_str = os.environ.get("VE_LAYERS", "7,8")

    # Optimizer — Adam (NO weight decay), Muon momentum 0.95 matching reference
    tied_embed_lr      = float(os.environ.get("TIED_EMBED_LR", 0.035))
    matrix_lr          = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr          = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum      = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_warmup_start  = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_warmup_steps  = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 10))

    # EMA + SWA
    ema_enabled   = int(os.environ.get("EMA_ENABLED", 1))
    ema_decay     = float(os.environ.get("EMA_DECAY", 0.997))
    swa_enabled   = int(os.environ.get("SWA_ENABLED", 1))
    swa_every     = int(os.environ.get("SWA_EVERY", 50))

    # TTT
    ttt_enabled      = int(os.environ.get("TTT_ENABLED", 1))
    ttt_epochs       = int(os.environ.get("TTT_EPOCHS", 5))
    ttt_lr           = float(os.environ.get("TTT_LR", 0.001))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 262144))
    ttt_batch_seqs   = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip    = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    eval_stride      = int(os.environ.get("EVAL_STRIDE", 64))

    @staticmethod
    def ve_layer_set():
        return set(int(x) for x in H.ve_layers_str.split(",") if x.strip())

CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "q_gain", "skip_weight", "resid_mix",
                    "smear", "bigram", "ve_")

# ── Muon optimizer ────────────────────────────────────────────────────────────
@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps=10, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self):
        is_dist = dist.is_available() and dist.is_initialized()
        ws = dist.get_world_size() if is_dist else 1
        rk = dist.get_rank() if is_dist else 0
        for group in self.param_groups:
            lr, mom, ns, nest = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
            params = group["params"]
            if not params: continue
            total = sum(p.numel() for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            cur = 0
            for i, p in enumerate(params):
                if i % ws == rk and p.grad is not None:
                    g = p.grad
                    st = self.state[p]
                    if "buf" not in st: st["buf"] = torch.zeros_like(g)
                    buf = st["buf"]
                    buf.mul_(mom).add_(g)
                    g_eff = g.add(buf, alpha=mom) if nest else buf
                    g_o = zeropower_via_newtonschulz5(g_eff, steps=ns)
                    g_o *= max(1, g_o.size(0) / g_o.size(1)) ** 0.5
                    flat[cur:cur+p.numel()] = g_o.reshape(-1)
                cur += p.numel()
            if is_dist: dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            cur = 0
            for p in params:
                p.add_(flat[cur:cur+p.numel()].view_as(p).to(p.dtype), alpha=-lr)
                cur += p.numel()

# ── Quantization (int8 per-row, matching reference) ───────────────────────────
INT8_CLIP_Q = 0.9999984
INT8_KEEP_FLOAT_MAX_NUMEL = 65536

def quantize_int8_row(t: Tensor):
    t32 = t.float()
    if t32.ndim == 2:
        clip = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1).clamp(min=1e-8)
        clipped = torch.clamp(t32, -clip[:,None], clip[:,None])
        scale = (clip / 127.0).clamp(min=1.0/127.0)
        q = torch.clamp(torch.round(clipped / scale[:,None]), -127, 127).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()
    clip = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale_v = clip / 127.0 if clip > 0 else 1.0
    q = torch.clamp(torch.round(torch.clamp(t32, -clip, clip) / scale_v), -127, 127).to(torch.int8)
    return q.contiguous(), torch.tensor(scale_v, dtype=torch.float16)

def dequantize_int8_row(q, scale, shape):
    if scale.ndim > 0:
        return (q.float() * scale.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(torch.bfloat16).view(shape)
    return (q.float() * float(scale.item())).to(torch.bfloat16).view(shape)

def quantize_state_dict(sd):
    meta, buf = {}, io.BytesIO()
    for k, v in sd.items():
        is_ctrl = any(p in k for p in CONTROL_PATTERNS) or v.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL
        if is_ctrl:
            data = v.cpu().to(torch.float16).numpy().tobytes()
            meta[k] = {"kind": "fp16", "shape": list(v.shape), "off": buf.tell(), "n": len(data)}
            buf.write(data)
        else:
            q, s = quantize_int8_row(v.cpu())
            qb, sb = q.numpy().tobytes(), s.numpy().tobytes()
            meta[k] = {"kind": "int8", "shape": list(v.shape), "sshape": list(s.shape),
                       "qoff": buf.tell(), "qn": len(qb), "soff": buf.tell()+len(qb), "sn": len(sb)}
            buf.write(qb); buf.write(sb)
    return meta, buf.getvalue()

def dequantize_state_dict(meta, raw):
    sd = {}
    for k, m in meta.items():
        if m["kind"] == "fp16":
            arr = np.frombuffer(raw[m["off"]:m["off"]+m["n"]], dtype=np.float16).copy()
            sd[k] = torch.from_numpy(arr).to(torch.bfloat16).view(m["shape"])
        else:
            q = np.frombuffer(raw[m["qoff"]:m["qoff"]+m["qn"]], dtype=np.int8).copy()
            s = np.frombuffer(raw[m["soff"]:m["soff"]+m["sn"]], dtype=np.float16).copy()
            qt = torch.from_numpy(q).view(m["sshape"][0] if len(m["sshape"]) > 0 else 1, -1)
            st = torch.from_numpy(s).view(m["sshape"])
            if st.ndim > 0:
                sd[k] = (qt.float() * st.float().view(qt.shape[0], *([1]*(qt.ndim-1)))).to(torch.bfloat16).view(m["shape"])
            else:
                sd[k] = (qt.float() * float(st.item())).to(torch.bfloat16).view(m["shape"])
    return sd

# ── Tokenizer BPB utilities ──────────────────────────────────────────────────
def build_sp_luts(sp, vocab_size, device):
    sv = int(sp.vocab_size())
    ts = max(sv, vocab_size)
    bb = np.zeros(ts, dtype=np.int16); hs = np.zeros(ts, dtype=np.bool_); ib = np.ones(ts, dtype=np.bool_)
    for tid in range(sv):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        ib[tid] = False
        if sp.is_byte(tid): bb[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"): hs[tid] = True; piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=device),
            torch.tensor(hs, dtype=torch.bool, device=device),
            torch.tensor(ib, dtype=torch.bool, device=device))

# ── Data loading ──────────────────────────────────────────────────────────────
def load_shard(f):
    hdr = np.fromfile(f, dtype="<i4", count=256); n = int(hdr[2])
    return torch.from_numpy(np.fromfile(f, dtype="<u2", count=n, offset=256*4).astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = sorted(glob.glob(pattern))
        if not self.files: raise FileNotFoundError(f"No files: {pattern}")
        self.fi, self.pos = 0, 0; self.tokens = load_shard(self.files[0])
    def _adv(self):
        self.fi = (self.fi + 1) % len(self.files); self.tokens = load_shard(self.files[self.fi]); self.pos = 0
    def take(self, n):
        chunks, rem = [], n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._adv(); continue
            k = min(rem, avail); chunks.append(self.tokens[self.pos:self.pos+k]); self.pos += k; rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistTokenLoader:
    def __init__(self, pattern, rank, ws, device):
        self.rank, self.ws, self.device = rank, ws, device; self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, accum=1):
        lt = global_tokens // (self.ws * accum); span = lt + 1
        chunk = self.stream.take(span * self.ws)
        local = chunk[self.rank*span:self.rank*span+span].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

def load_val_tokens(pattern, seq_len):
    tokens = torch.cat([load_shard(f) for f in sorted(glob.glob(pattern))]).contiguous()
    usable = ((tokens.numel()-1) // seq_len) * seq_len
    return tokens[:usable+1]

# ── Model components ──────────────────────────────────────────────────────────
class CastedLinear(nn.Linear):
    def forward(self, x): return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0/(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim)), persistent=False)
        self._cache = None
    def forward(self, seq_len, device, dtype):
        if self._cache is None or self._cache[2] != seq_len or self._cache[3] != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cache = (f.cos()[None,None].detach(), f.sin()[None,None].detach(), seq_len, device)
        return self._cache[0].to(dtype), self._cache[1].to(dtype)

def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class BigramHash(nn.Module):
    def __init__(self, vocab_size, bigram_vocab, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab = bigram_vocab
        self.embed = nn.Embedding(bigram_vocab, bigram_dim)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        nn.init.normal_(self.embed.weight, std=0.005)
        nn.init.zeros_(self.proj.weight)
    def forward(self, ids):
        prev = torch.cat([torch.zeros_like(ids[:,:1]), ids[:,:-1]], dim=1)
        bigram_ids = (prev * 1024 + ids) % self.bigram_vocab
        return self.proj(self.embed(bigram_ids))

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:,:1]), x[:,:-1]], dim=1)
        return x * (1 - g) + x_prev * g

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, nh, nkv, rope_base, qk_gain, use_xsa=False):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, dim // nh
        self.use_xsa = use_xsa
        kv_dim = nkv * self.hd
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.q_gain = nn.Parameter(torch.full((nh,), qk_gain, dtype=torch.float32))
        self.rotary = Rotary(self.hd, base=rope_base)

    def forward(self, x, ve_embed=None):
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = F.rms_norm(q, (self.hd,)), F.rms_norm(k, (self.hd,))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None,:,None,None]
        if ve_embed is not None:
            v = v + ve_embed.reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        if self.use_xsa:
            if self.nkv != self.nh:
                k = k.repeat_interleave(self.nh // self.nkv, dim=1)
                v = v.repeat_interleave(self.nh // self.nkv, dim=1)
            mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype), diagonal=1)
            mask = mask + torch.diag(torch.full((T,), float("-inf"), device=x.device, dtype=q.dtype))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               enable_gqa=(self.nkv != self.nh))
        return self.proj(y.transpose(1,2).contiguous().reshape(B, T, D))

class MLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        self.fc = CastedLinear(dim, dim * mult, bias=False)
        self.proj = CastedLinear(dim * mult, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class Block(nn.Module):
    def __init__(self, dim, nh, nkv, mult, rope_base, qk_gain, layer_idx, total_layers, use_xsa=False):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(dim, nh, nkv, rope_base, qk_gain, use_xsa=use_xsa)
        self.mlp = MLP(dim, mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())

    def forward(self, x, x0, ve_embed=None):
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None,None,:] * x + mix[1][None,None,:] * x0
        normed = F.rms_norm(x, (x.size(-1),))
        attn_out = self.attn(normed, ve_embed=ve_embed)
        ln_s = 1.0 / math.sqrt(self.layer_idx + 1) if H.ln_scale else 1.0
        x = x + self.attn_scale.to(x.dtype)[None,None,:] * attn_out * ln_s
        normed_m = F.rms_norm(x, (x.size(-1),))
        x = x + self.mlp_scale.to(x.dtype)[None,None,:] * self.mlp(normed_m) * ln_s
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        args = H
        dim, L = args.model_dim, args.num_layers
        self.tok_emb = nn.Embedding(args.vocab_size, dim)
        nn.init.normal_(self.tok_emb.weight, std=0.005)
        self.bigram = BigramHash(args.vocab_size, args.bigram_vocab, args.bigram_dim, dim) if args.bigram_vocab > 0 else None
        self.smear = SmearGate(dim) if args.smear_enabled else None
        self.num_enc = L // 2
        self.num_dec = L - self.num_enc
        self.num_skips = min(self.num_enc, self.num_dec)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skips, dim, dtype=torch.float32))
        xsa_start = L - args.xsa_last_n
        self.blocks = nn.ModuleList([
            Block(dim, args.num_heads, args.num_kv_heads, args.mlp_mult,
                  args.rope_base, args.qk_gain_init, i, L,
                  use_xsa=(i >= xsa_start))
            for i in range(L)
        ])
        self.logit_softcap = args.logit_softcap
        self.ve_layers = args.ve_layer_set() if args.ve_enabled else set()
        if self.ve_layers:
            self.ve_embed = nn.Embedding(args.vocab_size, args.ve_dim)
            self.ve_projs = nn.ParameterDict({
                str(li): nn.Parameter(torch.empty(args.num_kv_heads * (dim // args.num_heads), args.ve_dim))
                for li in self.ve_layers
            })
            for p in self.ve_projs.values():
                nn.init.orthogonal_(p, gain=args.ve_dim**-0.5)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        x = self.tok_emb(ids)
        if self.bigram is not None: x = x + self.bigram(ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None: x = self.smear(x)
        x0 = x
        skips = []
        for i in range(self.num_enc):
            ve_e = F.linear(self.ve_embed(ids), self.ve_projs[str(i)]) if i in self.ve_layers else None
            x = self.blocks[i](x, x0, ve_embed=ve_e)
            skips.append(x)
        for i in range(self.num_dec):
            li = self.num_enc + i
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None,None,:] * skips.pop()
            ve_e = F.linear(self.ve_embed(ids), self.ve_projs[str(li)]) if li in self.ve_layers else None
            x = self.blocks[li](x, x0, ve_embed=ve_e)
        x = F.rms_norm(x, (x.size(-1),)).reshape(-1, x.size(-1))
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if targets is not None:
            return F.cross_entropy(logits.float(), targets.reshape(-1), reduction="mean")
        return logits

# ── Evaluation ────────────────────────────────────────────────────────────────
def eval_val(model, rank, ws, device, accum, val_tokens, bb_lut, hs_lut, ib_lut):
    seq_len = H.train_seq_len
    local_seqs = max(H.val_batch_size // (ws * accum * seq_len), 1)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    s0, s1 = (total_seqs * rank) // ws, (total_seqs * (rank + 1)) // ws
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(s0, s1, local_seqs):
            be = min(bs + local_seqs, s1)
            rs, re = bs * seq_len, be * seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
            with torch.autocast("cuda", torch.bfloat16):
                bl = model(x, y).detach()
            n = float(y.numel())
            ls += bl.to(torch.float64) * n; tc += n
            prev, tgt = x.reshape(-1), y.reshape(-1)
            tb = bb_lut[tgt].to(torch.int16)
            tb += (hs_lut[tgt] & ~ib_lut[prev]).to(torch.int16)
            bc += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [ls, tc, bc]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = ls / tc; bpt = vl.item() / math.log(2.0); tpb = tc.item() / bc.item()
    model.train()
    return float(vl.item()), float(bpt * tpb)

# ── Sliding window eval ──────────────────────────────────────────────────────
def eval_sliding(model, rank, ws, device, val_tokens, bb_lut, hs_lut, ib_lut, stride=64):
    seq_len = H.train_seq_len
    total = val_tokens.numel() - 1
    starts = list(range(0, total - seq_len + 1, stride))
    my_starts = starts[rank::ws]
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_starts), 16):
            batch = my_starts[bi:bi+16]
            xs = torch.stack([val_tokens[p:p+seq_len] for p in batch]).to(device, dtype=torch.int64)
            ys = torch.stack([val_tokens[p+1:p+seq_len+1] for p in batch]).to(device, dtype=torch.int64)
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(xs)
            logits = logits.view(len(batch), seq_len, -1)
            ptl = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), ys.reshape(-1),
                                  reduction="none").reshape(len(batch), seq_len)
            for i, pos in enumerate(batch):
                cs = max(0, seq_len - stride) if pos > 0 else 0
                ls += ptl[i, cs:].double().sum(); tc += ptl[i, cs:].numel()
                tb = bb_lut[ys[i, cs:]].to(torch.int16)
                tb += (hs_lut[ys[i, cs:]] & ~ib_lut[xs[i, cs:]]).to(torch.int16)
                bc += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [ls, tc, bc]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = ls / tc; bpt = vl.item() / math.log(2.0); tpb = tc.item() / bc.item()
    model.train()
    return float(vl.item()), float(bpt * tpb)

# ── Legal Score-First TTT ─────────────────────────────────────────────────────
def execute_ttt(base_model, rank, ws, device, val_tokens, bb_lut, hs_lut, ib_lut):
    args = H
    chunk_size, stride, seq_len = args.ttt_chunk_tokens, args.eval_stride, args.train_seq_len
    model = GPT().to(device).bfloat16()
    raw_sd = base_model.state_dict()
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.requires_grad_(True)
    mlp_proj_params, mlp_fc_params, other_params = [], [], []
    for n, p in model.named_parameters():
        if "mlp.proj" in n: mlp_proj_params.append(p)
        elif "mlp.fc" in n: mlp_fc_params.append(p)
        else: other_params.append(p)
    ttt_ddp = DDP(model, device_ids=[device.index])
    opt = torch.optim.AdamW([
        {"params": other_params, "lr": args.ttt_lr},
        {"params": mlp_proj_params, "lr": args.ttt_lr * 3.0},
        {"params": mlp_fc_params, "lr": args.ttt_lr * 0.5},
    ], weight_decay=0.0)
    ls_total = torch.zeros((), device=device, dtype=torch.float64)
    tc_total = torch.zeros((), device=device, dtype=torch.float64)
    bc_total = torch.zeros((), device=device, dtype=torch.float64)
    total_len = val_tokens.numel() - 1
    chunk_starts = list(range(0, total_len, chunk_size))
    total_chunks = len(chunk_starts)
    for ci, start in enumerate(chunk_starts):
        end = min(start + chunk_size, total_len)
        chunk = val_tokens[start:end+1].to(device, dtype=torch.int64)
        # SCORE phase
        model.eval()
        starts_list = list(range(0, chunk.numel() - seq_len, stride))
        my_starts = starts_list[rank::ws]
        with torch.no_grad():
            for bi in range(0, len(my_starts), 16):
                batch = my_starts[bi:bi+16]
                xs = torch.stack([chunk[p:p+seq_len] for p in batch])
                ys = torch.stack([chunk[p+1:p+seq_len+1] for p in batch])
                with torch.autocast("cuda", torch.bfloat16):
                    logits = model(xs)
                logits = logits.view(len(batch), seq_len, -1)
                ptl = F.cross_entropy(logits.float().reshape(-1, H.vocab_size), ys.reshape(-1),
                                      reduction="none").reshape(len(batch), seq_len)
                for i, pos in enumerate(batch):
                    cs = 0 if pos == 0 and ci == 0 else seq_len - stride
                    ls_total += ptl[i, cs:].double().sum(); tc_total += ptl[i, cs:].numel()
                    tb = bb_lut[ys[i, cs:]].to(torch.int16)
                    tb += (hs_lut[ys[i, cs:]] & ~ib_lut[xs[i, cs:]]).to(torch.int16)
                    bc_total += tb.to(torch.float64).sum()
        if ci == total_chunks - 1: break
        # TRAIN phase
        model.train()
        seq_starts = list(range(0, chunk.numel() - seq_len, seq_len))
        seq_starts = seq_starts[:(len(seq_starts) // ws) * ws]
        my_seqs = seq_starts[rank::ws]
        for epoch in range(args.ttt_epochs):
            progress = (ci * args.ttt_epochs + epoch) / (total_chunks * args.ttt_epochs)
            cos_lr = 0.5 * (1.0 + math.cos(math.pi * progress))
            for g in opt.param_groups:
                if "initial_lr" not in g: g["initial_lr"] = g["lr"]
                g["lr"] = g["initial_lr"] * cos_lr
            for bi in range(0, len(my_seqs), args.ttt_batch_seqs):
                batch = my_seqs[bi:bi+args.ttt_batch_seqs]
                if not batch: continue
                xs = torch.stack([chunk[p:p+seq_len] for p in batch])
                ys = torch.stack([chunk[p+1:p+seq_len+1] for p in batch])
                with torch.autocast("cuda", torch.bfloat16):
                    loss = ttt_ddp(xs, ys)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ttt_ddp.parameters(), args.ttt_grad_clip)
                opt.step(); opt.zero_grad(set_to_none=True)
    if dist.is_available() and dist.is_initialized():
        for t in [ls_total, tc_total, bc_total]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = ls_total / tc_total; bpt = vl.item() / math.log(2.0); tpb = tc_total.item() / bc_total.item()
    return float(bpt * tpb)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = H
    code = Path(__file__).read_text(encoding="utf-8")
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    accum = 8 // ws
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(True)

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True); logfile = f"logs/{args.run_id}.txt"
    def log(msg, console=True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a") as f: print(msg, file=f)

    log(code, console=False)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_val_tokens(args.val_files, args.train_seq_len)
    bb_lut, hs_lut, ib_lut = build_sp_luts(sp, args.vocab_size, device)
    log(f"val tokens: {val_tokens.numel()-1}")

    base_model = GPT().to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    with torch.no_grad():
        for n, p in base_model.named_parameters():
            if (p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

    # EMA model — MUST deepcopy BEFORE torch.compile (compile modifies the module in-place)
    ema_model = copy.deepcopy(base_model)
    ema_model.requires_grad_(False)

    compiled = torch.compile(base_model, dynamic=False)
    model = DDP(compiled, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled

    # SWA accumulator (tight: starts accumulating from warmdown phase)
    swa_started = False
    if args.swa_enabled:
        swa_sd = None
        swa_count = 0

    # Optimizer groups — Adam WITHOUT weight decay (matching reference)
    block_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_PATTERNS)]
    scalar_params = [p for n, p in block_params if p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)
    assigned_ids = {id(p) for p in scalar_params} | {id(p) for p in matrix_params} | {id(base_model.tok_emb.weight)}
    for name, p in base_model.named_parameters():
        if any(x in name for x in ["bigram", "smear", "ve_"]):
            if id(p) not in assigned_ids:
                scalar_params.append(p); assigned_ids.add(id(p))

    opt_tok = torch.optim.Adam([{"params": [base_model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
                                 betas=(0.9, 0.95), fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                    betas=(0.9, 0.95), fused=True)
    opts = [opt_tok, opt_muon, opt_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log(f"params: {n_params} | layers: {args.num_layers} | dim: {args.model_dim}")
    log(f"ws: {ws} accum: {accum} | batch_tokens: {args.train_batch_tokens}")

    train_loader = DistTokenLoader(args.train_files, rank, ws, device)
    max_wall_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    grad_scale = 1.0 / accum

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wall_ms is None:
            wd_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if step >= wd_start else 1.0
        step_ms = elapsed_ms / max(step, 1)
        wd_ms = args.warmdown_iters * step_ms
        rem_ms = max(max_wall_ms - elapsed_ms, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    # Compile warmup
    if args.warmup_steps > 0:
        init_sd = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in opts]
        model.train()
        for ws_i in range(args.warmup_steps):
            for o in opts: o.zero_grad(set_to_none=True)
            for ms in range(accum):
                if distributed: model.require_backward_grad_sync = ms == accum - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, accum)
                with torch.autocast("cuda", torch.bfloat16):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for o in opts: o.step()
            for o in opts: o.zero_grad(set_to_none=True)
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(opts, init_opt): o.load_state_dict(s)
        for o in opts: o.zero_grad(set_to_none=True)
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistTokenLoader(args.train_files, rank, ws, device)
        ema_model.load_state_dict(init_sd, strict=True)
        log(f"warmup done ({args.warmup_steps} steps)")

    # Training loop
    train_ms, stop_after = 0.0, None
    torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0

    while True:
        last = step == args.iterations or (stop_after is not None and step >= stop_after)

        if last or (args.val_loss_every > 0 and step % args.val_loss_every == 0 and step > 0):
            torch.cuda.synchronize(); train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(model, rank, ws, device, accum, val_tokens, bb_lut, hs_lut, ib_lut)
            log(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} time:{train_ms:.0f}ms avg:{train_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()

        if last:
            if stop_after is not None and step < args.iterations: log(f"early stop at {step}")
            break

        elapsed = train_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed)

        # Muon momentum warmup
        frac = min(step / args.muon_warmup_steps, 1.0) if args.muon_warmup_steps > 0 else 1.0
        cur_mom = (1-frac) * args.muon_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups: g["momentum"] = cur_mom

        for o in opts:
            for g in o.param_groups: g["lr"] = g["base_lr"] * scale

        for o in opts: o.zero_grad(set_to_none=True)
        tl = torch.zeros((), device=device)
        for ms in range(accum):
            if distributed: model.require_backward_grad_sync = ms == accum - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, accum)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y)
            tl += loss.detach()
            (loss * grad_scale).backward()
        tl /= accum

        for o in opts: o.step()
        for o in opts: o.zero_grad(set_to_none=True)

        # EMA
        if args.ema_enabled:
            with torch.no_grad():
                for pe, pb in zip(ema_model.parameters(), base_model.parameters()):
                    pe.mul_(args.ema_decay).add_(pb, alpha=1.0-args.ema_decay)

        # Tight SWA — only accumulate during warmdown phase (scale < 1.0)
        if args.swa_enabled and scale < 1.0:
            if not swa_started:
                swa_sd = {k: v.clone().double() for k, v in base_model.state_dict().items()}
                swa_count = 0
                swa_started = True
            if step % args.swa_every == 0:
                with torch.no_grad():
                    swa_count += 1
                    for k, v in base_model.state_dict().items():
                        swa_sd[k] += v.double()

        step += 1
        approx = train_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log(f"step:{step}/{args.iterations} loss:{tl.item():.4f} time:{approx:.0f}ms avg:{approx/step:.2f}ms")

        reached = max_wall_ms is not None and approx >= max_wall_ms
        if distributed and max_wall_ms is not None:
            rt = torch.tensor(int(reached), device=device)
            dist.all_reduce(rt, op=dist.ReduceOp.MAX)
            reached = bool(rt.item())
        if stop_after is None and reached:
            stop_after = step

    log(f"peak mem: {torch.cuda.max_memory_allocated()//1024//1024} MiB")

    # Pick best model: evaluate EMA, and optionally SWA
    eval_model = ema_model if args.ema_enabled else base_model

    if args.swa_enabled and swa_started and swa_count > 0:
        swa_final_sd = {}
        with torch.no_grad():
            for k in swa_sd:
                swa_final_sd[k] = (swa_sd[k] / (swa_count + 1)).to(base_model.state_dict()[k].dtype)
        # Quick eval of SWA model
        swa_model = copy.deepcopy(base_model)
        swa_model.load_state_dict(swa_final_sd, strict=True)
        swa_compiled = torch.compile(swa_model, dynamic=False)
        swa_wrapped = DDP(swa_compiled, device_ids=[local_rank], broadcast_buffers=False) if distributed else swa_compiled
        swa_vl, swa_vb = eval_val(swa_wrapped, rank, ws, device, accum, val_tokens, bb_lut, hs_lut, ib_lut)
        log(f"SWA val_loss:{swa_vl:.4f} val_bpb:{swa_vb:.4f} (count={swa_count})")
        # Use EMA for comparison
        ema_sd = {k.replace("_orig_mod.", ""): v for k, v in ema_model.state_dict().items()}
        base_model.load_state_dict(ema_sd, strict=True)
        ema_vl, ema_vb = eval_val(model, rank, ws, device, accum, val_tokens, bb_lut, hs_lut, ib_lut)
        log(f"EMA val_loss:{ema_vl:.4f} val_bpb:{ema_vb:.4f}")
        if swa_vb < ema_vb:
            log("Using SWA model (better)")
            eval_model = swa_model
        else:
            log("Using EMA model (better)")

    # Serialize with int8 + zstd compression
    log("Serializing model...")
    sd = {k.replace("_orig_mod.", ""): v for k, v in eval_model.state_dict().items()}
    meta, raw = quantize_state_dict(sd)
    compressed = compress_bytes(raw)
    payload = pickle.dumps({"m": meta, "c": compressed})

    if master:
        Path("final_model.ptz").write_bytes(payload)
        code_bytes = len(code.encode("utf-8"))
        log(f"Model: {len(payload)} bytes | Code: {code_bytes} bytes | Total: {len(payload)+code_bytes} bytes")
        log(f"Target < 16,000,000 bytes: {'PASS' if len(payload)+code_bytes < 16_000_000 else 'FAIL'}")

    # Round-trip validation
    log("Round-trip validation...")
    loaded = pickle.loads(payload)
    decompressed = decompress_bytes(loaded["c"])
    rt_sd = dequantize_state_dict(loaded["m"], decompressed)
    base_model.load_state_dict(rt_sd, strict=True)
    torch.cuda.synchronize(); t_eval = time.perf_counter()
    q_vl, q_vb = eval_val(model, rank, ws, device, accum, val_tokens, bb_lut, hs_lut, ib_lut)
    torch.cuda.synchronize()
    log(f"int8+{COMPRESSOR} roundtrip val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f} eval:{1000*(time.perf_counter()-t_eval):.0f}ms")

    # Sliding window eval
    log("Sliding window eval...")
    torch.cuda.synchronize(); t_sw = time.perf_counter()
    sw_vl, sw_vb = eval_sliding(model, rank, ws, device, val_tokens, bb_lut, hs_lut, ib_lut, stride=args.eval_stride)
    torch.cuda.synchronize()
    log(f"sliding_window val_loss:{sw_vl:.4f} val_bpb:{sw_vb:.4f} eval:{1000*(time.perf_counter()-t_sw):.0f}ms")

    # TTT
    if args.ttt_enabled:
        log("Starting Legal Score-First TTT...")
        eval_model.load_state_dict(rt_sd, strict=True)
        torch.cuda.synchronize(); t_ttt = time.perf_counter()
        ttt_bpb = execute_ttt(eval_model, rank, ws, device, val_tokens, bb_lut, hs_lut, ib_lut)
        torch.cuda.synchronize()
        log(f"TTT val_bpb:{ttt_bpb:.4f} time:{1000*(time.perf_counter()-t_ttt):.0f}ms")
        log(f"FINAL val_bpb:{ttt_bpb:.4f}")
    else:
        log(f"FINAL val_bpb:{sw_vb:.4f}")

    if distributed:
        dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
