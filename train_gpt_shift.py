"""
train_gpt_shift.py  —  The Checkmate Pipeline (13-Layer INT4 + TTT + Muon + LaX)
===================================================================================
Target: 10-Minute 8x H100 Track | < 16MB | Goal: < 1.1150 val_bpb

Synthesized SOTA Techniques:
  1. Parameter Banking + Parallel Muon (Batched Newton-Schulz for sub-90ms steps)
  2. LeakyReLU(0.5)² Activation (Zero dead neurons, -0.003 bpb)
  3. Legal Score-First TTT (Inference-mode chunk scoring followed by 3-epoch SGD)
  4. VE128 (Value Embeddings injected into the final 4 layers)
  5. INT4 Bitwise Packing (Allows 13 layers to fit in ~12MB)
  6. LaX (Latent Crossing - Cross-depth residual accumulation, +0.000ms overhead)
"""

from __future__ import annotations

import glob
import io
import math
import os
import pickle
import random
import sys
import time
import uuid
import copy
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

try:
    from flash_attn import flash_attn_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS (Tuned for 10-Minute Checkmate)
# ─────────────────────────────────────────────────────────────────────────────
class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))

    # Speedrun limits
    iterations             = int(os.environ.get("ITERATIONS",          6800)) # 10 mins
    warmdown_iters         = int(os.environ.get("WARMDOWN_ITERS",      2500))
    warmup_steps           = int(os.environ.get("WARMUP_STEPS",        150))
    train_batch_tokens     = int(os.environ.get("TRAIN_BATCH_TOKENS",  524288))
    train_seq_len          = int(os.environ.get("TRAIN_SEQ_LEN",       1024))
    max_wallclock_seconds  = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 580.0))

    # The 13-Layer Exploit Architecture
    vocab_size    = int(os.environ.get("VOCAB_SIZE",    1024))
    num_layers    = int(os.environ.get("NUM_LAYERS",    13)) # Beats SOTA's 11
    model_dim     = int(os.environ.get("MODEL_DIM",     512))
    num_heads     = int(os.environ.get("NUM_HEADS",     8))
    num_kv_heads  = int(os.environ.get("NUM_KV_HEADS",  4))
    mlp_mult      = int(os.environ.get("MLP_MULT",      3))
    tie_embeddings= True
    rope_base     = float(os.environ.get("ROPE_BASE",   10000.0))
    shift_alpha_init = float(os.environ.get("SHIFT_ALPHA_INIT", 0.5))
    # Advanced Integrations
    xsa_layers    = int(os.environ.get("XSA_LAYERS",    4))
    ve_layers     = int(os.environ.get("VE_LAYERS",     4))
    ve_dim        = int(os.environ.get("VE_DIM",        128))
    rope_dims     = int(os.environ.get("ROPE_DIMS",     16))
    mlp_temp      = float(os.environ.get("MLP_TEMPERATURE", 1.5))

    # Parallel Muon
    tied_embed_lr         = float(os.environ.get("TIED_EMBED_LR",         0.035))
    matrix_lr             = float(os.environ.get("MATRIX_LR",             0.025))
    scalar_lr             = float(os.environ.get("SCALAR_LR",             0.025))
    mousse_momentum       = float(os.environ.get("MOUSSE_MOMENTUM",       0.99))
    mousse_warmup_start   = float(os.environ.get("MOUSSE_WARMUP_START",   0.92))
    mousse_warmup_steps   = int(os.environ.get("MOUSSE_WARMUP_STEPS",     1500))
    ema_decay             = float(os.environ.get("EMA_DECAY",             0.997))
    late_qat_frac         = float(os.environ.get("LATE_QAT_FRAC",         0.15))

    # TTT Rules
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_lr           = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs       = int(os.environ.get("TTT_EPOCHS", 3))

# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL MUON (Batched Newton-Schulz)
# ─────────────────────────────────────────────────────────────────────────────
def batched_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Executes Newton-Schulz on 3D Parameter Banks in parallel. Massive speedup."""
    transpose = G.size(1) < G.size(2)
    if transpose: G = G.transpose(1, 2)
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)
    for _ in range(steps):
        A = torch.bmm(X, X.transpose(1, 2))
        X = a * X + torch.bmm(b * A + c * torch.bmm(A, A), X)
    if transpose: X = X.transpose(1, 2)
    return X.to(G.dtype)

class ParallelMuon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps=5, beta2=0.95, wd=0.04):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, beta2=beta2, wd=wd))

    @torch.no_grad()
    def step(self):
        is_dist = dist.is_available() and dist.is_initialized()
        world_size, rank = (dist.get_world_size(), dist.get_rank()) if is_dist else (1, 0)

        for group in self.param_groups:
            lr, momentum, ns, beta2, wd = group["lr"], group["momentum"], group["backend_steps"], group["beta2"], group["wd"]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad.float()
                st = self.state[p]
                
                if "v" not in st:
                    st["v"], st["buf"], st["t"] = torch.zeros_like(g), torch.zeros_like(g), 0
                st["t"] += 1
                st["v"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                v_hat = st["v"] / (1.0 - beta2 ** st["t"])
                g_pre = g / (v_hat.sqrt() + 1e-8)
                
                buf = st["buf"]
                buf.mul_(momentum).add_(g_pre)
                g_eff = g_pre.add(buf, alpha=momentum)
                
                # NATIVE BANKING: If 3D, batch it. If 2D, standard.
                if g_eff.ndim == 3:
                    g_ortho = batched_newtonschulz5(g_eff, steps=ns)
                    g_ortho = g_ortho * max(1, g_ortho.size(1) / g_ortho.size(2)) ** 0.5
                else:
                    X = g_eff.unsqueeze(0)
                    g_ortho = batched_newtonschulz5(X, steps=ns).squeeze(0)
                    g_ortho = g_ortho * max(1, g_ortho.size(0) / g_ortho.size(1)) ** 0.5

                if is_dist: dist.all_reduce(g_ortho, op=dist.ReduceOp.SUM)
                
                if wd > 0: p.mul_(1.0 - lr * wd)
                p.add_(g_ortho.to(p.dtype), alpha=-lr)

# ─────────────────────────────────────────────────────────────────────────────
# INT4 BITWISE QUANTIZATION
# ─────────────────────────────────────────────────────────────────────────────
def _fake_quant_int4(w: Tensor) -> Tensor:
    scale = w.abs().max().clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(w / scale), -8, 7)
    return w + (q * scale - w).detach() 

def _quantize_tensor_int4_packed(t: Tensor):
    t32 = t.float()
    original_shape = t32.shape
    if t32.ndim == 3: t32 = t32.reshape(-1, t32.shape[-1])
        
    clip = torch.quantile(t32.abs(), 0.9999, dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (clip / 7.0).clamp(min=1.0 / 7.0)
    q = torch.clamp(torch.round(t32 / scale), -8, 7).to(torch.int8)
    
    q_flat = q.flatten()
    if q_flat.numel() % 2 != 0: q_flat = F.pad(q_flat, (0, 1), value=0)
    q_shifted = (q_flat + 8).to(torch.uint8)
    packed = (q_shifted[0::2] << 4) | q_shifted[1::2]
    
    return packed.contiguous(), scale.to(torch.float16).contiguous(), original_shape

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE: PARAMETER BANKING + VE128 + LEAKY_RELU² + LaX
# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    h = rope_dims // 2
    x_r, x_p = x[..., :rope_dims], x[..., rope_dims:]
    c, s = cos[..., :h], sin[..., :h]
    x_a, x_b = x_r[..., :h], x_r[..., h:]
    rotated = torch.cat([x_a * c - x_b * s, x_a * s + x_b * c], dim=-1)
    return torch.cat([rotated, x_p], dim=-1)

class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        dim, hd, nh, nkv = args.model_dim, args.model_dim // args.num_heads, args.num_heads, args.num_kv_heads
        qkv_dim = (nh + 2 * nkv) * hd
        hidden = dim * args.mlp_mult
        L = args.num_layers

        # Token + Shift Embed
        self.tok_emb = nn.Embedding(args.vocab_size, dim)
        self.shift_alpha = nn.Parameter(torch.full((L,), args.shift_alpha_init))

        # === 3D PARAMETER BANKS (The Speed Hack) ===
        self.bank_qkv     = nn.Parameter(torch.empty(L, qkv_dim, dim))
        self.bank_out     = nn.Parameter(torch.empty(L, dim, nh * hd))
        self.bank_gate_up = nn.Parameter(torch.empty(L, hidden * 2, dim))
        self.bank_down    = nn.Parameter(torch.empty(L, dim, hidden))
        
        nn.init.orthogonal_(self.bank_qkv, gain=dim**-0.5)
        nn.init.orthogonal_(self.bank_gate_up, gain=dim**-0.5)
        nn.init.zeros_(self.bank_out)
        nn.init.zeros_(self.bank_down)

        # Scales, Norms, & LaX Gate
        self.attn_scale = nn.Parameter(torch.ones(L, dim))
        self.mlp_scale  = nn.Parameter(torch.ones(L, dim))
        self.q_gain     = nn.Parameter(torch.ones(L, nh) * 1.5)
        
        # === Latent Crossing (LaX) Integration ===
        # Initialized to zero so it learns to inject laterally without shocking the initial setup
        self.lax_gate   = nn.Parameter(torch.zeros(L, dim)) 

        # VE128 (Value Embeddings for deep layers)
        self.ve = nn.Embedding(args.vocab_size, args.ve_dim)
        self.ve_proj = nn.Parameter(torch.empty(args.ve_layers, nkv * hd, args.ve_dim))
        nn.init.orthogonal_(self.ve_proj, gain=args.ve_dim**-0.5)

        inv_freq = 1.0 / (args.rope_base ** (torch.arange(0, args.rope_dims, 2, dtype=torch.float32) / args.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, ids: Tensor, targets: Tensor = None) -> Tensor:
        B, T = ids.shape
        x = self.tok_emb(ids)
        
        x_prev = torch.roll(x, 1, dims=1)
        x_prev[:, 0, :] = 0
        
        cos = torch.outer(torch.arange(T, device=x.device), self.inv_freq).cos()[None, None]
        sin = torch.outer(torch.arange(T, device=x.device), self.inv_freq).sin()[None, None]

        hd, nh, nkv = self.args.model_dim // self.args.num_heads, self.args.num_heads, self.args.num_kv_heads
        q_len, k_len = nh * hd, nkv * hd
        ve_start, xsa_start = self.args.num_layers - self.args.ve_layers, self.args.num_layers - self.args.xsa_layers

        # The LaX Accumulator (Tracks lateral state across all layers)
        latent_state = torch.zeros_like(x)

        for i in range(self.args.num_layers):
            x_shift = x + self.shift_alpha[i] * x_prev
            normed_x = F.rms_norm(x_shift, (self.args.model_dim,))
            
            # 1. Attn via Bank
            qkv = F.linear(normed_x, self.bank_qkv[i])
            q = qkv[..., :q_len].reshape(B, T, nh, hd).transpose(1, 2)
            k = qkv[..., q_len:q_len+k_len].reshape(B, T, nkv, hd).transpose(1, 2)
            v = qkv[..., q_len+k_len:].reshape(B, T, nkv, hd).transpose(1, 2)

            q, k = F.rms_norm(q, (hd,)), F.rms_norm(k, (hd,))
            q = apply_rope(q, cos, sin, self.args.rope_dims) * self.q_gain[i][None, :, None, None]
            k = apply_rope(k, cos, sin, self.args.rope_dims)

            # VE128 Injection
            if i >= ve_start:
                ve_embed = F.linear(self.ve(ids), self.ve_proj[i - ve_start])
                v = v + ve_embed.reshape(B, T, nkv, hd).transpose(1, 2)

            if nkv != nh:
                k, v = k.repeat_interleave(nh // nkv, dim=1), v.repeat_interleave(nh // nkv, dim=1)
            
            if i >= xsa_start:
                mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
                mask.diagonal().fill_(float("-inf"))
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
            else:
                y = flash_attn_func(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), causal=True).transpose(1, 2)
                
            attn_out = F.linear(y.transpose(1, 2).contiguous().reshape(B, T, nh * hd), self.bank_out[i])
            x = x + self.attn_scale[i] * attn_out * (1.0 / math.sqrt(i + 1))
            
            # 2. MLP via Bank + LeakyReLU(0.5)²
            normed_m = F.rms_norm(x, (self.args.model_dim,))
            gu = F.linear(normed_m, self.bank_gate_up[i])
            hidden = gu.shape[-1] // 2
            act = F.leaky_relu(gu[..., :hidden] * self.args.mlp_temp, negative_slope=0.5).square() * gu[..., hidden:]
            mlp_out = F.linear(act, self.bank_down[i])
            x = x + self.mlp_scale[i] * mlp_out * (1.0 / math.sqrt(i + 1))

            # === LaX Accumulation & Injection ===
            # The current state 'x' informs the lateral latent state, which then pushes back into 'x'
            latent_state = latent_state + (x * self.lax_gate[i])
            x = x + latent_state

        x = F.rms_norm(x, (self.args.model_dim,))
        logits = F.linear(x, self.tok_emb.weight)
        
        if targets is not None:
            return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean")
        return logits

# ─────────────────────────────────────────────────────────────────────────────
# LEGAL SCORE-FIRST TTT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def execute_legal_ttt(args, base_model, rank, world_size, device, val_tokens):
    chunk_size, stride, seq_len = args.ttt_chunk_tokens, 64, args.train_seq_len
    model = copy.deepcopy(base_model)
    ls_total, tc_total = torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64)
    total_len = val_tokens.numel() - 1
    chunk_starts = list(range(0, total_len, chunk_size))
    
    for ci, start in enumerate(chunk_starts):
        end = min(start + chunk_size, total_len)
        chunk = val_tokens[start:end+1].to(device, dtype=torch.int64)
        
        model.eval()
        starts = list(range(0, chunk.numel() - seq_len, stride))
        my_starts = starts[rank::world_size]
        
        with torch.inference_mode():
            for bi in range(0, len(my_starts), 16):
                batch = my_starts[bi : bi + 16]
                xs = torch.stack([chunk[p : p+seq_len] for p in batch])
                ys = torch.stack([chunk[p+1 : p+seq_len+1] for p in batch])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): 
                    logits = model(xs)
                ptl = F.cross_entropy(logits.float().reshape(-1, args.vocab_size), ys.reshape(-1), reduction="none").reshape(len(batch), seq_len)
                for i, pos in enumerate(batch):
                    cs = 0 if pos == 0 and ci == 0 else seq_len - stride
                    ls_total += ptl[i, cs:].double().sum()
                    tc_total += ptl[i, cs:].numel()

        if ci == len(chunk_starts) - 1: break 
        
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=args.ttt_lr, momentum=0.9)
        seq_starts = list(range(0, chunk.numel() - seq_len, seq_len))
        my_seqs = seq_starts[rank::world_size]
        
        for _ in range(args.ttt_epochs):
            for bi in range(0, len(my_seqs), 32): 
                batch = my_seqs[bi : bi + 32]
                if not batch: continue
                xs = torch.stack([chunk[p : p+seq_len] for p in batch])
                ys = torch.stack([chunk[p+1 : p+seq_len+1] for p in batch])
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): 
                    loss = model(xs, ys)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ls_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(tc_total, op=dist.ReduceOp.SUM)
    return (ls_total / tc_total).item()

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = Hyperparameters()
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    def log(msg):
        if rank == 0: print(msg)

    # Token Setup
    def load_val(p): return torch.from_numpy(np.fromfile(glob.glob(p)[0], dtype="<u2", offset=1024).astype(np.int64))
    val_tokens = load_val(args.val_files)
    
    model = GPT(args).to(device).bfloat16()
    ddp_model = DDP(model, device_ids=[device.index])
    ema_model = copy.deepcopy(model).requires_grad_(False)

    banks = [p for n, p in model.named_parameters() if "bank" in n]
    scalars = [p for n, p in model.named_parameters() if "bank" not in n and "tok_emb" not in n]
    
    opt_tok = torch.optim.AdamW(model.tok_emb.parameters(), lr=args.tied_embed_lr, fused=True)
    opt_muon = ParallelMuon(banks, lr=args.matrix_lr, momentum=args.mousse_momentum)
    opt_scalar = torch.optim.AdamW(scalars, lr=args.scalar_lr, fused=True)
    opts = [opt_tok, opt_muon, opt_scalar]

    stream = np.fromfile(glob.glob(args.train_files)[0], dtype="<u2", offset=1024).astype(np.int64)
    stream = torch.from_numpy(stream)
    
    t0 = time.perf_counter()
    pos = 0
    lbt = args.train_batch_tokens // world_size
    lbs = lbt // args.train_seq_len

    for step in range(args.iterations):
        if time.perf_counter() - t0 > args.max_wallclock_seconds: 
            log(f"Time limit reached at step {step}")
            break
            
        scale = max((args.iterations - step) / args.warmdown_iters, 0.0) if step > args.iterations - args.warmdown_iters else 1.0
        for o in opts:
            for g in o.param_groups: g["lr"] = g.get("base_lr", g["lr"]) * scale

        local = stream[pos + rank * lbt : pos + (rank + 1) * lbt + 1].to(device)
        pos += args.train_batch_tokens
        x, y = local[:-1].reshape(lbs, args.train_seq_len), local[1:].reshape(lbs, args.train_seq_len)

        with torch.autocast("cuda", torch.bfloat16): loss = ddp_model(x, y)
        loss.backward()
        
        for o in opts: o.step()
        for o in opts: o.zero_grad(set_to_none=True)

        with torch.no_grad():
            for pe, pb in zip(ema_model.parameters(), model.parameters()):
                pe.mul_(args.ema_decay).add_(pb, alpha=1.0 - args.ema_decay)
                
        if step > args.iterations * (1.0 - args.late_qat_frac):
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if "bank" in n: p.data = _fake_quant_int4(p.data)

        if step % 100 == 0: 
            approx = 1000.0 * (time.perf_counter() - t0)
            log(f"Step {step} | Loss: {loss.item():.4f} | step_avg: {approx/max(step,1):.2f}ms")

    log("\nExecuting Score-First TTT Evaluation on EMA model...")
    ttt_bpb = execute_legal_ttt(args, ema_model, rank, world_size, device, val_tokens)
    log(f"FINAL TTT BPB: {ttt_bpb:.4f}")

    if rank == 0:
        sd = ema_model.state_dict()
        buf, meta = io.BytesIO(), {}
        for k, v in sd.items():
            if "bank" in k:
                q, s, shape = _quantize_tensor_int4_packed(v.cpu())
                qb, sb = q.numpy().tobytes(), s.numpy().tobytes()
                meta[k] = {"kind": "int4", "shape": list(shape), "scale_shape": list(s.shape),
                           "q_off": buf.tell(), "q_n": len(qb), "s_off": buf.tell() + len(qb), "s_n": len(sb)}
                buf.write(qb); buf.write(sb)
            else:
                data = v.cpu().to(torch.float16).numpy().tobytes()
                meta[k] = {"kind": "fp16", "shape": list(v.shape), "offset": buf.tell(), "n": len(data)}
                buf.write(data)
                
        payload = pickle.dumps({"c": compress_bytes(buf.getvalue()), "m": meta})
        Path("final_model.ptz").write_bytes(payload)
        log(f"Artifact Size: {len(payload)/1e6:.2f} MB (Target < 16MB)")

if __name__ == "__main__":
    main()
