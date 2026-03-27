"""
train_gpt_shift.py  —  Shift-Mixed Progressive-RoPE Transformer (INT4 Packed Edition)
===================================================================================
Optimized for 8x H100s, ~8B tokens, < 1.0 val_bpb, and absolute < 16MB artifact size.

Key Architecture (31.8M Parameters):
  - 10 Layers, d=512, GQA (8H/2KV), SmearGate MLP (mlp_mult=3)
  - 4-bit Bitwise Packing: Two weights per byte, slashing artifact size to ~15.5MB.
  - Deep QAT: 15% Late Fake-Quantization to INT4 (-8 to 7) for stability.
  - Single "Golden Epoch" across 8B tokens.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import pickle
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ── zstd compression ──────────────────────────────────────────────────────────
try:
    import zstandard as _zstd
    _ZL = int(os.environ.get("ZSTD_LEVEL", "22"))
    compress_bytes   = lambda b: _zstd.ZstdCompressor(level=_ZL).compress(b)
    decompress_bytes = lambda b: _zstd.ZstdDecompressor().decompress(b)
    COMPRESSOR = f"zstd-{_ZL}"
except ImportError:
    import zlib
    compress_bytes   = lambda b: zlib.compress(b, level=9)
    decompress_bytes = lambda b: zlib.decompress(b)
    COMPRESSOR = "zlib-9"

# ── FlashAttention 3 ──────────────────────────────────────────────────────────
try:
    from flash_attn import flash_attn_func
    import torch as _torch
    _gpu_name = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else ""
    HAS_FA3 = "H100" in _gpu_name or "H200" in _gpu_name
except ImportError:
    HAS_FA3 = False

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS (Tuned for 8x H100, 8B Tokens, 1 Hour)
# ─────────────────────────────────────────────────────────────────────────────
class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))

    val_loss_every    = int(os.environ.get("VAL_LOSS_EVERY",    500))
    train_log_every   = int(os.environ.get("TRAIN_LOG_EVERY",   100))
    max_val_tokens    = int(os.environ.get("MAX_VAL_TOKENS",    0)) 

    # 8B Token Run Math: 1M batch size * 7600 iters = 7.6B tokens (Sweet spot)
    iterations             = int(os.environ.get("ITERATIONS",          7600))
    warmdown_iters         = int(os.environ.get("WARMDOWN_ITERS",      1500))
    warmup_steps           = int(os.environ.get("WARMUP_STEPS",        300))
    train_batch_tokens     = int(os.environ.get("TRAIN_BATCH_TOKENS",  1_048_576))
    train_seq_len          = int(os.environ.get("TRAIN_SEQ_LEN",       1024))
    max_wallclock_seconds  = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0.0)) # Disable time limit for strict step count

    # 31.8M Parameter Architecture
    vocab_size    = int(os.environ.get("VOCAB_SIZE",    1024))
    num_layers    = int(os.environ.get("NUM_LAYERS",    10))
    num_heads     = int(os.environ.get("NUM_HEADS",     8))
    num_kv_heads  = int(os.environ.get("NUM_KV_HEADS",  2))
    model_dim     = int(os.environ.get("MODEL_DIM",     512))
    mlp_mult      = int(os.environ.get("MLP_MULT",      3))
    tie_embeddings= bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base     = float(os.environ.get("ROPE_BASE",   10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    xsa_layers    = int(os.environ.get("XSA_LAYERS",    4))

    # Progressive RoPE dims (head_dim = 64)
    rope_dims_early = int(os.environ.get("ROPE_DIMS_EARLY", 8)) 
    rope_dims_mid   = int(os.environ.get("ROPE_DIMS_MID",   16)) 
    rope_dims_late  = int(os.environ.get("ROPE_DIMS_LATE",  32)) 

    shift_alpha_init = float(os.environ.get("SHIFT_ALPHA_INIT", 0.5))
    mlp_temperature  = float(os.environ.get("MLP_TEMPERATURE", 1.5))

    # Optimizer
    tied_embed_lr         = float(os.environ.get("TIED_EMBED_LR",         0.03))
    tied_embed_init_std   = float(os.environ.get("TIED_EMBED_INIT_STD",   0.005))
    matrix_lr             = float(os.environ.get("MATRIX_LR",             0.02))
    scalar_lr             = float(os.environ.get("SCALAR_LR",             0.02))
    mousse_momentum       = float(os.environ.get("MOUSSE_MOMENTUM",       0.99))
    mousse_beta2          = float(os.environ.get("MOUSSE_BETA2",          0.95))
    mousse_backend_steps  = int(os.environ.get("MOUSSE_BACKEND_STEPS",    10)) # Increased to 10 for H100 speed
    mousse_wd             = float(os.environ.get("MOUSSE_WD",             0.04))
    mousse_warmup_start   = float(os.environ.get("MOUSSE_WARMUP_START",   0.85))
    mousse_warmup_steps   = int(os.environ.get("MOUSSE_WARMUP_STEPS",     500))
    beta1                 = float(os.environ.get("BETA1",                 0.9))
    beta2                 = float(os.environ.get("BETA2",                 0.95))
    adam_eps              = float(os.environ.get("ADAM_EPS",              1e-8))
    adam_wd               = float(os.environ.get("ADAM_WD",               0.04))
    grad_clip_norm        = float(os.environ.get("GRAD_CLIP_NORM",        0.3))

    ema_decay             = float(os.environ.get("EMA_DECAY",             0.997))
    late_qat_frac = float(os.environ.get("LATE_QAT_FRAC", 0.02)) # 15% Deep QAT curriculum
    sliding_window_stride = int(os.environ.get("SLIDING_WINDOW_STRIDE",   64))


# ─────────────────────────────────────────────────────────────────────────────
# MOUSSE OPTIMIZER (arXiv:2603.09697)
# ─────────────────────────────────────────────────────────────────────────────
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return (X.T if G.size(0) > G.size(1) else X).to(G.dtype)

class Mousse(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, beta2=0.999, eps=1e-8, nesterov=True, wd=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, 
                                      beta2=beta2, eps=eps, nesterov=nesterov, wd=wd))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        rank = dist.get_rank() if is_dist else 0

        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, ns = group["lr"], group["momentum"], group["backend_steps"]
            beta2, eps, wd, nesterov = group["beta2"], group["eps"], group["wd"], group["nesterov"]

            total = sum(p.numel() for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0

            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad.float()
                    st = self.state[p]
                    if "v" not in st:
                        st["v"] = torch.zeros_like(g)
                        st["buf"] = torch.zeros_like(g)
                        st["t"] = 0
                    st["t"] += 1
                    st["v"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    v_hat = st["v"] / (1.0 - beta2 ** st["t"])
                    g_pre = g / (v_hat.sqrt() + eps)
                    buf = st["buf"]
                    buf.mul_(momentum).add_(g_pre)
                    g_eff = g_pre.add(buf, alpha=momentum) if nesterov else buf.clone()
                    g_ortho = zeropower_via_newtonschulz5(g_eff.bfloat16(), steps=ns)
                    g_ortho = g_ortho * max(1, g_ortho.size(0) / g_ortho.size(1)) ** 0.5
                    flat[curr : curr + p.numel()] = g_ortho.reshape(-1)
                curr += p.numel()

            if is_dist: dist.all_reduce(flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = flat[curr : curr + p.numel()].view_as(p).to(p.dtype)
                if wd > 0: p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# ─────────────────────────────────────────────────────────────────────────────
# INT4 BITWISE QUANTIZATION (PACKED)
# ─────────────────────────────────────────────────────────────────────────────
CONTROL_PATTERNS = tuple(p for p in os.environ.get(
    "CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,depth_gate,shift_alpha",
).split(",") if p)
FP16_PATTERNS  = ("tok_emb.weight",)
MAX_FLOAT_NUMEL = 65_536
INT4_LO, INT4_HI = -8, 7

def _fake_quant_int4(w: Tensor) -> Tensor:
    """STE int4 fake quantization during Deep QAT."""
    w32   = w.float()
    scale = w32.abs().max().clamp(min=1e-8) / float(INT4_HI)
    q     = torch.clamp(torch.round(w32 / scale), INT4_LO, INT4_HI)
    return w + (q * scale - w).detach() 

def _quantize_tensor_int4_packed(t: Tensor):
    """Quantize to INT4 and pack two values into one uint8 byte."""
    t32 = t.float()
    hi = float(INT4_HI)
    if t32.ndim == 2:
        clip = torch.quantile(t32.abs(), 0.9999, dim=1).clamp(min=1e-8)
        scale = (clip / hi).clamp(min=1.0 / hi)
        q = torch.clamp(torch.round(t32 / scale[:, None]), INT4_LO, INT4_HI).to(torch.int8)
        scale_out = scale.to(torch.float16).contiguous()
    else:
        clip = float(torch.quantile(t32.abs().flatten(), 0.9999).item())
        scale = torch.tensor(max(clip / hi, 1.0 / hi), dtype=torch.float32)
        q = torch.clamp(torch.round(t32 / scale), INT4_LO, INT4_HI).to(torch.int8)
        scale_out = scale

    q_flat = q.flatten()
    if q_flat.numel() % 2 != 0:
        q_flat = F.pad(q_flat, (0, 1), value=0)
        
    q_shifted = (q_flat + 8).to(torch.uint8)
    packed = (q_shifted[0::2] << 4) | q_shifted[1::2]
    return packed.contiguous(), scale_out

def unpack_int4_tensor(packed: Tensor, original_shape: list, per_row: bool, scale: Tensor) -> Tensor:
    """Unpack bits and reverse scaling."""
    high_bits = (packed >> 4) & 0x0F
    low_bits  = packed & 0x0F
    
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    unpacked[0::2] = high_bits
    unpacked[1::2] = low_bits
    
    unpacked_int8 = unpacked.to(torch.int8) - 8
    original_numel = int(np.prod(original_shape))
    if unpacked_int8.numel() > original_numel:
        unpacked_int8 = unpacked_int8[:original_numel]
        
    q = unpacked_int8.reshape(original_shape).float()
    dq = (q * scale.view(-1, *([1] * (q.ndim - 1))) if per_row else q * float(scale.item()))
    return dq

def quantize_state_dict(sd: dict) -> tuple[bytes, dict]:
    buf, meta = io.BytesIO(), {}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        if any(p in name for p in FP16_PATTERNS):
            data = t.to(torch.float16).numpy().tobytes()
            meta[name] = {"kind": "fp16", "shape": list(t.shape), "offset": buf.tell(), "n": len(data)}
            buf.write(data)
        elif (not t.is_floating_point() or t.numel() <= MAX_FLOAT_NUMEL or any(p in name for p in CONTROL_PATTERNS)):
            data = t.to(torch.float32).numpy().tobytes()
            meta[name] = {"kind": "fp32", "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."), "offset": buf.tell(), "n": len(data)}
            buf.write(data)
        else:
            q_packed, scale = _quantize_tensor_int4_packed(t)
            qb, sb = q_packed.numpy().tobytes(), scale.numpy().tobytes()
            meta[name] = {"kind": "int4_packed", "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."),
                          "per_row": (t.ndim == 2), "scale_shape": list(scale.shape),
                          "q_off": buf.tell(), "q_n": len(qb), "s_off": buf.tell() + len(qb), "s_n": len(sb)}
            buf.write(qb); buf.write(sb)
    return compress_bytes(buf.getvalue()), meta

def dequantize_state_dict(compressed: bytes, meta: dict) -> dict:
    raw, out = decompress_bytes(compressed), {}
    for name, m in meta.items():
        kind = m["kind"]
        if kind == "fp16":
            arr = np.frombuffer(raw, dtype=np.float16, count=int(np.prod(m["shape"])), offset=m["offset"])
            out[name] = torch.from_numpy(arr.copy()).reshape(m["shape"]).to(torch.bfloat16)
        elif kind == "fp32":
            arr = np.frombuffer(raw, dtype=np.float32, count=int(np.prod(m["shape"])), offset=m["offset"])
            t = torch.from_numpy(arr.copy()).reshape(m["shape"])
            dt = m.get("dtype", "float32")
            out[name] = t.to(getattr(torch, dt)) if dt != "float32" else t
        elif kind == "int4_packed":
            packed = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, count=m["q_n"], offset=m["q_off"]).copy())
            scale = torch.from_numpy(np.frombuffer(raw, dtype=np.float16, count=int(np.prod(m["scale_shape"])), offset=m["s_off"]).copy()).reshape(m["scale_shape"]).float()
            dq = unpack_int4_tensor(packed, m["shape"], m["per_row"], scale)
            out[name] = dq.to(getattr(torch, m["dtype"])).contiguous()
    return out

def save_artifact(model: nn.Module, path: str) -> int:
    compressed, meta = quantize_state_dict(model.state_dict())
    payload = pickle.dumps({"c": compressed, "m": meta})
    Path(path).write_bytes(payload)
    return len(payload)

def load_artifact(path: str, model: nn.Module) -> None:
    obj = pickle.loads(Path(path).read_bytes())
    model.load_state_dict(dequantize_state_dict(obj["c"], obj["m"]), strict=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & EVAL (Unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _load_shard(file: Path) -> Tensor:
    hb = 256 * 4
    hdr = np.fromfile(file, dtype="<i4", count=256)
    n = int(hdr[2])
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=n, offset=hb).astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.fi, self.pos = 0, 0
        self.tokens = _load_shard(self.files[0])

    def _adv(self):
        self.fi = (self.fi + 1) % len(self.files)
        self.tokens = _load_shard(self.files[self.fi])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks, rem = [], n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._adv(); continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k; rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, gas):
        lt = global_tokens // (self.world_size * gas)
        span = lt + 1
        chunk = self.stream.take(span * self.world_size)
        local = chunk[self.rank * span : self.rank * span + span].to(torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

def build_sentencepiece_luts(sp, vocab_size: int, device):
    sz = max(int(sp.vocab_size()), vocab_size)
    bb, hs, ib = np.zeros(sz, dtype=np.int16), np.zeros(sz, dtype=np.bool_), np.ones(sz, dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        ib[tid] = False
        if sp.is_byte(tid):
            bb[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            hs[tid] = True; piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=device), torch.tensor(hs, dtype=torch.bool, device=device), torch.tensor(ib, dtype=torch.bool, device=device))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    tokens = torch.cat([_load_shard(Path(f)) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]

def _byte_count(x, y, bb, hs, ib):
    tgt, prev = y.reshape(-1), x.reshape(-1)
    tb = bb[tgt].to(torch.int16)
    tb += (hs[tgt] & ~ib[prev]).to(torch.int16)
    return tb.to(torch.float64).sum()

def _reduce(ls, tc, bc, is_dist):
    if is_dist:
        for t in (ls, tc, bc): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (ls / tc).item()
    return vl, (vl / math.log(2)) * (tc.item() / bc.item())

def eval_val(args, model, rank, world_size, device, gas, val_tokens, bb, hs, ib):
    lbt = args.train_batch_tokens // (world_size * gas)
    lbs = max(lbt // args.train_seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    if args.max_val_tokens > 0:
        total_seqs = max(min(total_seqs, args.max_val_tokens // args.train_seq_len), 1)
    ss, se = (total_seqs * rank) // world_size, (total_seqs * (rank+1)) // world_size
    ls, tc, bc = torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(ss, se, lbs):
            bse = min(bss + lbs, se)
            local = val_tokens[bss*args.train_seq_len : bse*args.train_seq_len+1].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                bl = model(x, y).detach()
            n = float(y.numel())
            ls += bl.to(torch.float64) * n; tc += n; bc += _byte_count(x, y, bb, hs, ib)
    model.train()
    return _reduce(ls, tc, bc, dist.is_available() and dist.is_initialized())

@torch.no_grad()
def eval_val_sliding_window(args, base_model, rank, world_size, device, val_tokens, bb, hs, ib):
    stride, seq_len = args.sliding_window_stride, args.train_seq_len
    base_model.eval()
    ls, tc, bc = torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64)
    total = val_tokens.numel() - 1
    if args.max_val_tokens > 0: total = min(total, args.max_val_tokens)
    starts = list(range(0, total - seq_len + 1, stride))
    my_starts = starts[rank::world_size]
    WB = 16
    with torch.inference_mode():
        for bi in range(0, len(my_starts), WB):
            batch = my_starts[bi : bi + WB]
            xs = torch.stack([val_tokens[p : p+seq_len]   for p in batch]).to(device, dtype=torch.int64)
            ys = torch.stack([val_tokens[p+1 : p+seq_len+1] for p in batch]).to(device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base_model.forward_logits(xs)
            ptl = F.cross_entropy(logits.float().reshape(-1, args.vocab_size), ys.reshape(-1), reduction="none").reshape(len(batch), seq_len)
            for i, pos in enumerate(batch):
                cs = 0 if pos == 0 else seq_len - stride
                ls += ptl[i, cs:].double().sum()
                tc += ptl[i, cs:].numel()
                bc += _byte_count(xs[i:i+1, cs:], ys[i:i+1, cs:], bb, hs, ib)
    base_model.train()
    return _reduce(ls, tc, bc, dist.is_available() and dist.is_initialized())

# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

def _ortho(w: Tensor, scale: float = 1.0): nn.init.orthogonal_(w, gain=scale)

def restore_fp32(module: nn.Module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

class ShiftMixedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int, alpha_init: float = 0.5):
        super().__init__()
        self.weight = nn.Embedding(vocab_size, model_dim)
        self.alpha  = nn.Parameter(torch.tensor(alpha_init))
    def forward(self, ids: Tensor) -> Tensor:
        x = self.weight(ids)
        x_prev = torch.roll(x, 1, dims=1)
        x_prev[:, 0, :] = 0
        return x + self.alpha.to(x.dtype) * x_prev

class ProgressiveRotary(nn.Module):
    def __init__(self, max_dims: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, max_dims, 2, dtype=torch.float32) / max_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._sl, self._cos, self._sin = 0, None, None
    def forward(self, sl: int, device, dtype):
        if self._cos is None or self._sl != sl or self._cos.device != device:
            t = torch.arange(sl, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cos, self._sin = f.cos()[None, None], f.sin()[None, None]
            self._sl = sl
        return self._cos.to(dtype), self._sin.to(dtype)

def apply_rope_partial(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    h = rope_dims // 2
    x_r, x_p = x[..., :rope_dims], x[..., rope_dims:]
    c, s = cos[..., :h], sin[..., :h]
    x_a, x_b = x_r[..., :h], x_r[..., h:]
    rotated = torch.cat([x_a * c - x_b * s, x_a * s + x_b * c], dim=-1)
    return torch.cat([rotated, x_p], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_dims, use_xsa, rope_base, qk_gain_init):
        super().__init__()
        self.nh, self.nkv = num_heads, num_kv_heads
        self.hd = dim // num_heads
        self.rope_dims, self.use_xsa = rope_dims, use_xsa
        kv_dim = num_kv_heads * self.hd
        self.c_q, self.c_k, self.c_v = CastedLinear(dim, dim, bias=False), CastedLinear(dim, kv_dim, bias=False), CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.ones(num_heads, dtype=torch.float32) * qk_gain_init)
        self.rotary = ProgressiveRotary(self.hd, base=rope_base)
        for w in (self.c_q, self.c_k, self.c_v): _ortho(w.weight, dim ** -0.5)
        self._xsa_mask_cache = {}

    def _get_xsa_mask(self, T: int, device, dtype) -> Tensor:
        key = (T, device, str(dtype))
        if key not in self._xsa_mask_cache:
            mask = torch.triu(torch.full((T, T), float("-inf"), device=device, dtype=dtype), diagonal=1)
            mask.diagonal().fill_(float("-inf"))
            self._xsa_mask_cache = {key: mask}
        return self._xsa_mask_cache[key]

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q, k, v = self.c_q(x).reshape(B, T, self.nh, self.hd).transpose(1, 2), self.c_k(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2), self.c_v(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q, k = apply_rope_partial(q, cos, sin, self.rope_dims), apply_rope_partial(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        if self.nkv != self.nh:
            rep = self.nh // self.nkv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        
        if self.use_xsa and HAS_FA3:
            attn_mask = self._get_xsa_mask(T, x.device, q.dtype)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        elif HAS_FA3:
            q_, k_, v_ = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
            y = flash_attn_func(q_, k_, v_, causal=True).transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, self.nh * self.hd))

class SmearGateMLP(nn.Module):
    def __init__(self, dim, mlp_mult, temperature: float = 1.5):
        super().__init__()
        hidden = dim * mlp_mult
        self.gate, self.up, self.proj = CastedLinear(dim, hidden, bias=False), CastedLinear(dim, hidden, bias=False), CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.temperature = temperature
        _ortho(self.gate.weight, dim ** -0.5); _ortho(self.up.weight, dim ** -0.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.gate(x) * self.temperature).square() * self.up(x))

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_dims, use_xsa, rope_base, qk_gain_init, layer_idx, mlp_temperature):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_dims, use_xsa, rope_base, qk_gain_init)
        self.mlp = SmearGateMLP(dim, mlp_mult, temperature=mlp_temperature)
        self.attn_scale, self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32)), nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())
        self.depth_gate = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix, dg = self.resid_mix.to(x.dtype), self.depth_gate.to(x.dtype)
        x = mix[0][None, None] * x + mix[1][None, None] * x0
        x = x + dg * self.attn_scale.to(x.dtype)[None, None] * self.attn(self.attn_norm(x)) * self.ln_scale
        x = x + dg * self.mlp_scale.to(x.dtype)[None, None] * self.mlp(self.mlp_norm(x)) * self.ln_scale
        return x

class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args, self.tie_embeddings, self.logit_softcap = args, args.tie_embeddings, args.logit_softcap
        self.tok_emb = ShiftMixedEmbedding(args.vocab_size, args.model_dim, args.shift_alpha_init)
        self.num_enc, self.num_dec = args.num_layers // 2, args.num_layers - (args.num_layers // 2)
        self.num_skips = min(self.num_enc, self.num_dec)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skips, args.model_dim, dtype=torch.float32))

        thirds = args.num_layers // 3
        self.rope_dims_list = ([args.rope_dims_early] * thirds + [args.rope_dims_mid] * thirds + [args.rope_dims_late] * (args.num_layers - 2 * thirds))
        xsa_start = args.num_layers - args.xsa_layers

        self.blocks = nn.ModuleList([
            Block(dim=args.model_dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
                  rope_dims=self.rope_dims_list[i], use_xsa=(i >= xsa_start), rope_base=args.rope_base, qk_gain_init=1.5,
                  layer_idx=i, mlp_temperature=args.mlp_temperature) for i in range(args.num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if args.tie_embeddings else CastedLinear(args.model_dim, args.vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        nn.init.normal_(self.tok_emb.weight.weight, std=args.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False): nn.init.zeros_(m.weight)

    def _backbone(self, ids: Tensor) -> Tensor:
        x = F.rms_norm(self.tok_emb(ids), (self.tok_emb(ids).size(-1),))
        x0, skips = x, []
        for i in range(self.num_enc):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.num_dec):
            if skips: x = x + self.skip_weights[i].to(x.dtype)[None, None] * skips.pop()
            x = self.blocks[self.num_enc + i](x, x0)
        return self.final_norm(x)

    def forward_logits(self, ids: Tensor) -> Tensor:
        x = self._backbone(ids)
        lp = F.linear(x, self.tok_emb.weight.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)

    def forward(self, ids: Tensor, targets: Tensor) -> Tensor:
        logits = self.forward_logits(ids)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean")

    def apply_late_qat(self):
        """Apply STE int4 fake-quantization to large weights (Deep QAT phase)."""
        with torch.no_grad():
            for name, p in self.named_parameters():
                if (p.ndim == 2 and p.numel() > MAX_FLOAT_NUMEL and not any(pat in name for pat in CONTROL_PATTERNS) and not any(pat in name for pat in FP16_PATTERNS)):
                    p.data = _fake_quant_int4(p.data)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    use_compile = bool(int(os.environ.get("USE_COMPILE", "1")))
    if use_compile: zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank, world_size, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))
    gas = int(os.environ.get("GRAD_ACCUM", str(8 // world_size)))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not HAS_FA3:
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        flash = sys.platform != "win32"
        enable_cudnn_sdp(False); enable_flash_sdp(flash); enable_mem_efficient_sdp(False); enable_math_sdp(not flash)

    logfile = f"logs/{args.run_id}.txt" if master else None
    if master: os.makedirs("logs", exist_ok=True)
    def log(msg, console=True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bb, hs, ib = build_sentencepiece_luts(sp, args.vocab_size, device)

    base_model = GPT(args).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_fp32(base_model)
    compiled = torch.compile(base_model, dynamic=False, fullgraph=True) if use_compile else base_model
    model = DDP(compiled, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled
    ema_model = copy.deepcopy(base_model)
    ema_model.requires_grad_(False)

    def update_ema(decay: float):
        with torch.no_grad():
            for p_ema, p_base in zip(ema_model.parameters(), base_model.parameters()):
                p_ema.data.mul_(decay).add_(p_base.data, alpha=1.0 - decay)

    block_np = list(base_model.blocks.named_parameters())
    matrix_p = [p for n, p in block_np if p.ndim == 2 and not any(pat in n for pat in CONTROL_PATTERNS)]
    scalar_p = [p for n, p in block_np if p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_p.append(base_model.skip_weights)

    opt_tok = torch.optim.AdamW([{"params": list(base_model.tok_emb.parameters()), "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    opt_mousse = Mousse(matrix_p, lr=args.matrix_lr, momentum=args.mousse_momentum, backend_steps=args.mousse_backend_steps, beta2=args.mousse_beta2, wd=args.mousse_wd)
    for g in opt_mousse.param_groups: g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.AdamW([{"params": scalar_p, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizers = [opt_tok, opt_mousse, opt_scalar]

    def zero_grad():
        for o in optimizers: o.zero_grad(set_to_none=True)

    def lr_mul(step):
        if args.warmdown_iters <= 0: return 1.0
        ws = max(args.iterations - args.warmdown_iters, 0)
        return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0

    def should_late_qat(step):
        return step > args.iterations * (1.0 - args.late_qat_frac)

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    
    # Warmup Loop
    if args.warmup_steps > 0:
        init_sd = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws_i in range(args.warmup_steps):
            zero_grad()
            for micro in range(gas):
                if distributed: model.require_backward_grad_sync = micro == gas - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, gas)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): loss = model(x, y)
                (loss / gas).backward()
            for o in optimizers: o.step()
            zero_grad()
        base_model.load_state_dict(init_sd, strict=True)
        for o, st in zip(optimizers, init_opt, strict=True): o.load_state_dict(st)
        zero_grad()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        ema_model.load_state_dict(base_model.state_dict())

    # Main Training Loop
    t0 = time.perf_counter()
    train_ms = 0.0
    for step in range(args.iterations + 1):
        last = (step == args.iterations)
        
        if not last and args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0:
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, gas, val_tokens, bb, hs, ib)
            log(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{train_ms:.0f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last: break

        scale = lr_mul(step)
        zero_grad()
        tl = torch.zeros((), device=device)
        for micro in range(gas):
            if distributed: model.require_backward_grad_sync = micro == gas - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, gas)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): loss = model(x, y)
            tl += loss.detach()
            (loss / gas).backward()
        tl /= gas

        frac = min(step / args.mousse_warmup_steps, 1.0) if args.mousse_warmup_steps > 0 else 1.0
        cur_mom = (1-frac) * args.mousse_warmup_start + frac * args.mousse_momentum
        for g in opt_mousse.param_groups: g["momentum"] = cur_mom
        for o in optimizers:
            for g in o.param_groups: g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        
        for o in optimizers: o.step()
        zero_grad()
        
        # Stop corrupting EMA once QAT starts
        if not should_late_qat(step):
            update_ema(args.ema_decay)

        if should_late_qat(step):
            base_model.apply_late_qat()

        if should_late_qat(step):
            base_model.apply_late_qat()

        if args.train_log_every > 0 and step % args.train_log_every == 0:
            approx = train_ms + 1000.0 * (time.perf_counter() - t0)
            log(f"step:{step}/{args.iterations} train_loss:{tl.item():.4f} step_avg:{approx/max(step,1):.2f}ms")

    # ── Final Eval and Serialization ──────────────────────────────────────────
    log("\n[EMA] loading EMA weights for final eval...")
    ema_sd, base_sd = ema_model.state_dict(), base_model.state_dict()
    fixed_sd = {k: ema_sd[k].to(base_sd[k].dtype).contiguous() for k in base_sd}
    base_model.load_state_dict(fixed_sd, strict=True)

    log("[final eval] sliding-window (stride=64)...")
    torch.cuda.synchronize()
    swl, swb = eval_val_sliding_window(args, base_model, rank, world_size, device, val_tokens, bb, hs, ib)
    log(f"pre_quant val_loss:{swl:.4f} val_bpb:{swb:.4f}")

    if master:
        nb = save_artifact(base_model, "final_model.ptz")
        cb = len(code.encode("utf-8"))
        total = nb + cb
        log(f"\nartifact:{nb}B  code:{cb}B  total:{total}B  ({total/1e6:.3f}MB)")
        log("OK: within 16MB" if total <= 16_000_000 else "WARNING: EXCEEDS 16MB")

    if distributed: dist.barrier()
    load_artifact("final_model.ptz", base_model)
    torch.cuda.synchronize()
    rtl, rtb = eval_val_sliding_window(args, base_model, rank, world_size, device, val_tokens, bb, hs, ib)
    log(f"roundtrip val_loss:{rtl:.4f} val_bpb:{rtb:.4f}")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
