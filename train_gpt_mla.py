"""
train_gpt_shift.py  —  Shift-Mixed Progressive-RoPE Transformer
================================================================
Novel contributions vs #315 (current best non-TTT at 1.1250):

  1. Shift-Mixed Embedding (replaces BigramHash)
     x = E[tok] + α * causal_roll(x, 1)
     Zero params vs BigramHash's 10240×128 table. Implicit bigram context.
     Causal: position 0 gets zeros, no future leakage.

  2. Progressive RoPE (replaces uniform 16/64)
     Layers  0-3 : 8/64 dims   (local syntax — little position needed)
     Layers  4-7 : 16/64 dims  (mid-range)
     Layers  8-10: 32/64 dims  (global semantics)
     Principled allocation of positional capacity across depth.

  3. Mousse optimizer (replaces standard Muon)
     Diagonal curvature preconditioning before Newton-Schulz orthogonalization.
     ~12% more effective training at ~3% overhead. (arXiv:2603.09697)

  4. Layer-depth gate (additive, alongside XSA)
     x = x + g_l * block(x)  where g_l is learnable per-layer scalar init=1.
     Gives network ability to suppress weak layers at no extra cost.

Proven techniques kept from leaderboard (#315 / #287 / #198):
  - 11 layers, d=512, GQA (8H/2KV), SmearGate MLP (mlp_mult=3)
  - XSA (Exclusive Self-Attention) on last 4 layers
  - EMA weight averaging (decay=0.997) — beats SWA by 0.003 bpb
  - LN Scale: RMSNorm output × 1/√(layer+1)
  - Late QAT: STE int6 enabled in last 4% of training steps
  - OrthoInit on all linear weights
  - int6 weights + fp16 tied embedding + zstd-22
  - U-Net skip connections
  - Sliding window evaluation (stride=64)
  - FA3 if flash_attn installed, SDPA fallback

Batch sizing:
  A40  48GB:  TRAIN_BATCH_TOKENS=524288  (default)
  H100 80GB:  TRAIN_BATCH_TOKENS=786432  (set env var)

Run:
  # A40 single GPU test
  USE_COMPILE=0 MAX_VAL_TOKENS=524288 python train_gpt_shift.py

  # 8×H100 submission
  TRAIN_BATCH_TOKENS=786432 torchrun --nproc_per_node=8 train_gpt_shift.py
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
    compress_bytes = lambda b: _zstd.ZstdCompressor(level=_ZL).compress(b)
    decompress_bytes = lambda b: _zstd.ZstdDecompressor().decompress(b)
    COMPRESSOR = f"zstd-{_ZL}"
except ImportError:
    import zlib

    compress_bytes = lambda b: zlib.compress(b, level=9)
    decompress_bytes = lambda b: zlib.decompress(b)
    COMPRESSOR = "zlib-9 (pip install zstandard for zstd-22)"

# ── FlashAttention 3 ──────────────────────────────────────────────────────────
try:
    from flash_attn import flash_attn_func

    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    max_val_tokens = int(os.environ.get("MAX_VAL_TOKENS", 0))  # 0=full

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    # ↓ Default = A40 48GB safe. Set to 786432 for H100 80GB.
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 2))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # XSA: apply Exclusive Self-Attention to the last N layers
    xsa_layers = int(os.environ.get("XSA_LAYERS", 4))

    # Progressive RoPE dims per layer group (must divide head_dim evenly)
    # head_dim = 512/8 = 64
    rope_dims_early = int(
        os.environ.get("ROPE_DIMS_EARLY", 8)
    )  # layers 0..num_layers//3
    rope_dims_mid = int(os.environ.get("ROPE_DIMS_MID", 16))  # layers //3..2*//3
    rope_dims_late = int(os.environ.get("ROPE_DIMS_LATE", 32))  # layers 2*//3..end

    # Shift-mixed embedding learnable scale init
    shift_alpha_init = float(os.environ.get("SHIFT_ALPHA_INIT", 0.5))

    # Optimizer
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    mousse_momentum = float(os.environ.get("MOUSSE_MOMENTUM", 0.99))
    mousse_beta2 = float(os.environ.get("MOUSSE_BETA2", 0.999))
    mousse_backend_steps = int(os.environ.get("MOUSSE_BACKEND_STEPS", 5))
    mousse_wd = float(os.environ.get("MOUSSE_WD", 0.04))
    mousse_warmup_start = float(os.environ.get("MOUSSE_WARMUP_START", 0.85))
    mousse_warmup_steps = int(os.environ.get("MOUSSE_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # EMA weight averaging
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    # Late QAT: enable STE int6 fake-quant in last late_qat_frac of steps
    late_qat_frac = float(os.environ.get("LATE_QAT_FRAC", 0.04))

    # Sliding window eval
    sliding_window_stride = int(os.environ.get("SLIDING_WINDOW_STRIDE", 64))


# ─────────────────────────────────────────────────────────────────────────────
# MOUSSE OPTIMIZER  (arXiv:2603.09697)
# Curvature-aware Muon: diagonal Shampoo preconditioning + Newton-Schulz
# ─────────────────────────────────────────────────────────────────────────────


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return (X.T if G.size(0) > G.size(1) else X).to(G.dtype)


class Mousse(torch.optim.Optimizer):
    """
    Mousse: Muon + diagonal curvature preconditioning.
    Maintains exponential moving average of squared gradients (v_t) per parameter.
    Preconditions gradient by 1/sqrt(v_t + eps) before Newton-Schulz.
    This adapts the effective learning rate per-element based on local curvature.
    """

    def __init__(
        self,
        params,
        lr,
        momentum,
        backend_steps,
        beta2=0.999,
        eps=1e-8,
        nesterov=True,
        wd=0.0,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                beta2=beta2,
                eps=eps,
                nesterov=nesterov,
                wd=wd,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        rank = dist.get_rank() if is_dist else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum, ns = group["lr"], group["momentum"], group["backend_steps"]
            beta2, eps, wd = group["beta2"], group["eps"], group["wd"]
            nesterov = group["nesterov"]

            total = sum(p.numel() for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0

            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad.float()
                    st = self.state[p]

                    # Diagonal curvature estimate (EMA of squared gradients)
                    if "v" not in st:
                        st["v"] = torch.zeros_like(g)
                        st["buf"] = torch.zeros_like(g)
                        st["t"] = 0
                    st["t"] += 1
                    st["v"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    # Bias-corrected curvature
                    v_hat = st["v"] / (1.0 - beta2 ** st["t"])

                    # Precondition: scale gradient by 1/sqrt(v_hat + eps)
                    g_pre = g / (v_hat.sqrt() + eps)

                    # Momentum buffer on preconditioned gradient
                    buf = st["buf"]
                    buf.mul_(momentum).add_(g_pre)
                    g_eff = g_pre.add(buf, alpha=momentum) if nesterov else buf.clone()

                    # Newton-Schulz orthogonalization (operates on bfloat16)
                    g_ortho = zeropower_via_newtonschulz5(g_eff.bfloat16(), steps=ns)
                    g_ortho = g_ortho * max(1, g_ortho.size(0) / g_ortho.size(1)) ** 0.5

                    flat[curr : curr + p.numel()] = g_ortho.reshape(-1)
                curr += p.numel()

            if is_dist:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = flat[curr : curr + p.numel()].view_as(p).to(p.dtype)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# QUANTIZATION: int6 + fp16 embed + zstd-22
# ─────────────────────────────────────────────────────────────────────────────

CONTROL_PATTERNS = tuple(
    p
    for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,depth_gate,shift_alpha",
    ).split(",")
    if p
)

FP16_PATTERNS = ("tok_emb.weight",)
MAX_FLOAT_NUMEL = 65_536
INT6_LO, INT6_HI = -32, 31


def _fake_quant_int6(w: Tensor) -> Tensor:
    """STE int6 fake quantization — used during Late QAT phase."""
    w32 = w.float()
    scale = w32.abs().max().clamp(min=1e-8) / float(INT6_HI)
    q = torch.clamp(torch.round(w32 / scale), INT6_LO, INT6_HI)
    return w + (q * scale - w).detach()  # STE: forward=quantized, backward=identity


def _quantize_tensor_int6(t: Tensor):
    """Per-row int6 for 2D, per-tensor for 1D."""
    t32 = t.float()
    hi = float(INT6_HI)
    if t32.ndim == 2:
        clip = torch.quantile(t32.abs(), 0.9999984, dim=1).clamp(min=1e-8)
        scale = (clip / hi).clamp(min=1.0 / hi)
        q = (
            torch.clamp(
                torch.round(
                    torch.clamp(t32, -clip[:, None], clip[:, None]) / scale[:, None]
                ),
                INT6_LO,
                INT6_HI,
            )
            .to(torch.int8)
            .contiguous()
        )
        return q, scale.to(torch.float16).contiguous()
    clip = float(torch.quantile(t32.abs().flatten(), 0.9999984).item())
    scale = torch.tensor(max(clip / hi, 1.0 / hi), dtype=torch.float32)
    q = (
        torch.clamp(
            torch.round(torch.clamp(t32, -clip, clip) / scale), INT6_LO, INT6_HI
        )
        .to(torch.int8)
        .contiguous()
    )
    return q, scale


def quantize_state_dict(sd: dict) -> tuple[bytes, dict]:
    buf = io.BytesIO()
    meta = {}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        if any(p in name for p in FP16_PATTERNS):
            data = t.to(torch.float16).numpy().tobytes()
            meta[name] = {
                "kind": "fp16",
                "shape": list(t.shape),
                "offset": buf.tell(),
                "n": len(data),
            }
            buf.write(data)
        elif (
            not t.is_floating_point()
            or t.numel() <= MAX_FLOAT_NUMEL
            or any(p in name for p in CONTROL_PATTERNS)
        ):
            data = t.to(torch.float32).numpy().tobytes()
            meta[name] = {
                "kind": "fp32",
                "shape": list(t.shape),
                "dtype": str(t.dtype).removeprefix("torch."),
                "offset": buf.tell(),
                "n": len(data),
            }
            buf.write(data)
        else:
            q, scale = _quantize_tensor_int6(t)
            qb, sb = q.numpy().tobytes(), scale.numpy().tobytes()
            meta[name] = {
                "kind": "int6",
                "shape": list(t.shape),
                "dtype": str(t.dtype).removeprefix("torch."),
                "per_row": (t.ndim == 2),
                "scale_shape": list(scale.shape),
                "q_off": buf.tell(),
                "q_n": len(qb),
                "s_off": buf.tell() + len(qb),
                "s_n": len(sb),
            }
            buf.write(qb)
            buf.write(sb)
    return compress_bytes(buf.getvalue()), meta


def dequantize_state_dict(compressed: bytes, meta: dict) -> dict:
    raw = decompress_bytes(compressed)
    out = {}
    for name, m in meta.items():
        kind = m["kind"]
        if kind == "fp16":
            arr = np.frombuffer(
                raw,
                dtype=np.float16,
                count=int(np.prod(m["shape"])),
                offset=m["offset"],
            )
            out[name] = (
                torch.from_numpy(arr.copy()).reshape(m["shape"]).to(torch.bfloat16)
            )
        elif kind == "fp32":
            arr = np.frombuffer(
                raw,
                dtype=np.float32,
                count=int(np.prod(m["shape"])),
                offset=m["offset"],
            )
            t = torch.from_numpy(arr.copy()).reshape(m["shape"])
            dt = m.get("dtype", "float32")
            out[name] = t.to(getattr(torch, dt)) if dt != "float32" else t
        else:
            q = (
                torch.from_numpy(
                    np.frombuffer(
                        raw,
                        dtype=np.int8,
                        count=int(np.prod(m["shape"])),
                        offset=m["q_off"],
                    ).copy()
                )
                .reshape(m["shape"])
                .float()
            )
            scale = (
                torch.from_numpy(
                    np.frombuffer(
                        raw,
                        dtype=np.float16,
                        count=int(np.prod(m["scale_shape"])),
                        offset=m["s_off"],
                    ).copy()
                )
                .reshape(m["scale_shape"])
                .float()
            )
            dq = (
                q * scale.view(-1, *([1] * (q.ndim - 1)))
                if m["per_row"]
                else q * float(scale.item())
            )
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
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────


def _load_shard(file: Path) -> Tensor:
    hb = 256 * 4
    hdr = np.fromfile(file, dtype="<i4", count=256)
    if hdr.size != 256 or int(hdr[0]) != 20240520 or int(hdr[1]) != 1:
        raise ValueError(f"Bad shard: {file}")
    n = int(hdr[2])
    if file.stat().st_size != hb + n * 2:
        raise ValueError(f"Size mismatch: {file}")
    return torch.from_numpy(
        np.fromfile(file, dtype="<u2", count=n, offset=hb).astype(np.uint16, copy=False)
    )


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
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
                self._adv()
                continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            rem -= k
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
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER EVAL
# ─────────────────────────────────────────────────────────────────────────────


def build_sentencepiece_luts(sp, vocab_size: int, device):
    sz = max(int(sp.vocab_size()), vocab_size)
    bb = np.zeros(sz, dtype=np.int16)
    hs = np.zeros(sz, dtype=np.bool_)
    ib = np.ones(sz, dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        ib[tid] = False
        if sp.is_byte(tid):
            bb[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            hs[tid] = True
            piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(bb, dtype=torch.int16, device=device),
        torch.tensor(hs, dtype=torch.bool, device=device),
        torch.tensor(ib, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
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
        for t in (ls, tc, bc):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (ls / tc).item()
    return vl, (vl / math.log(2)) * (tc.item() / bc.item())


def eval_val(args, model, rank, world_size, device, gas, val_tokens, bb, hs, ib):
    lbt = args.train_batch_tokens // (world_size * gas)
    lbs = max(lbt // args.train_seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    if args.max_val_tokens > 0:
        total_seqs = max(min(total_seqs, args.max_val_tokens // args.train_seq_len), 1)
    ss = (total_seqs * rank) // world_size
    se = (total_seqs * (rank + 1)) // world_size
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(ss, se, lbs):
            bse = min(bss + lbs, se)
            local = val_tokens[
                bss * args.train_seq_len : bse * args.train_seq_len + 1
            ].to(device=device, dtype=torch.int64)
            x, y = (
                local[:-1].reshape(-1, args.train_seq_len),
                local[1:].reshape(-1, args.train_seq_len),
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                bl = model(x, y).detach()
            n = float(y.numel())
            ls += bl.to(torch.float64) * n
            tc += n
            bc += _byte_count(x, y, bb, hs, ib)
    model.train()
    return _reduce(ls, tc, bc, dist.is_available() and dist.is_initialized())


@torch.no_grad()
def eval_val_sliding_window(
    args, base_model, rank, world_size, device, val_tokens, bb, hs, ib
):
    stride, seq_len = args.sliding_window_stride, args.train_seq_len
    base_model.eval()
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    total = val_tokens.numel() - 1
    if args.max_val_tokens > 0:
        total = min(total, args.max_val_tokens)
    starts = list(range(0, total - seq_len + 1, stride))
    my_starts = starts[rank::world_size]
    WB = 16
    with torch.inference_mode():
        for bi in range(0, len(my_starts), WB):
            batch = my_starts[bi : bi + WB]
            xs = torch.stack([val_tokens[p : p + seq_len] for p in batch]).to(
                device, dtype=torch.int64
            )
            ys = torch.stack([val_tokens[p + 1 : p + seq_len + 1] for p in batch]).to(
                device, dtype=torch.int64
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base_model.forward_logits(xs)
            ptl = F.cross_entropy(
                logits.float().reshape(-1, args.vocab_size),
                ys.reshape(-1),
                reduction="none",
            ).reshape(len(batch), seq_len)
            for i, pos in enumerate(batch):
                cs = 0 if pos == 0 else seq_len - stride
                ls += ptl[i, cs:].double().sum()
                tc += ptl[i, cs:].numel()
                bc += _byte_count(xs[i : i + 1, cs:], ys[i : i + 1, cs:], bb, hs, ib)
    base_model.train()
    return _reduce(ls, tc, bc, dist.is_available() and dist.is_initialized())


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(
            x,
            self.weight.to(x.dtype),
            self.bias.to(x.dtype) if self.bias is not None else None,
        )


def _ortho(w: Tensor, scale: float = 1.0):
    nn.init.orthogonal_(w, gain=scale)


def restore_fp32(module: nn.Module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (
                p.ndim < 2 or any(pat in name for pat in CONTROL_PATTERNS)
            ) and p.dtype != torch.float32:
                p.data = p.data.float()


# ── Shift-Mixed Embedding ─────────────────────────────────────────────────────
class ShiftMixedEmbedding(nn.Module):
    """
    Implicit bigram context via causal roll.
    x = E[tok] + α * prev_tok_emb
    α is learnable. Zero extra tables, zero extra memory beyond one scalar.
    Causal: position 0 gets zeros (no future leakage).
    """

    def __init__(self, vocab_size: int, model_dim: int, alpha_init: float = 0.5):
        super().__init__()
        self.weight = nn.Embedding(vocab_size, model_dim)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, ids: Tensor) -> Tensor:
        x = self.weight(ids)  # [B, T, D]
        x_prev = torch.roll(x, 1, dims=1)
        x_prev[:, 0, :] = 0  # causal: mask wraparound
        return x + self.alpha.to(x.dtype) * x_prev


# ── Progressive Rotary Embedding ──────────────────────────────────────────────
class ProgressiveRotary(nn.Module):
    """
    Builds full RoPE tables for max_dims, applies only first rope_dims.
    Early layers: fewer dims → less position bias (local processing).
    Late layers: more dims → richer position signal (global reasoning).
    """

    def __init__(self, max_dims: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, max_dims, 2, dtype=torch.float32) / max_dims)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._sl, self._cos, self._sin = 0, None, None

    def forward(self, sl: int, device, dtype):
        if self._cos is None or self._sl != sl or self._cos.device != device:
            t = torch.arange(sl, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cos = f.cos()[None, None]  # [1,1,T,max_dims//2]
            self._sin = f.sin()[None, None]
            self._sl = sl
        return self._cos.to(dtype), self._sin.to(dtype)


def apply_rope_partial(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    """Apply RoPE to first rope_dims of head_dim, leave remaining unrotated."""
    h = rope_dims // 2
    x_r = x[..., :rope_dims]
    x_p = x[..., rope_dims:]
    c, s = cos[..., :h], sin[..., :h]
    x_a, x_b = x_r[..., :h], x_r[..., h:]
    rotated = torch.cat([x_a * c - x_b * s, x_a * s + x_b * c], dim=-1)
    return torch.cat([rotated, x_p], dim=-1)


# ── Attention with XSA + Progressive RoPE + FA3 ──────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads, num_kv_heads, rope_dims, use_xsa, rope_base, qk_gain_init
    ):
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.nh, self.nkv = num_heads, num_kv_heads
        self.hd = dim // num_heads
        self.rope_dims = rope_dims
        self.use_xsa = use_xsa
        kv_dim = num_kv_heads * self.hd

        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.ones(num_heads, dtype=torch.float32) * qk_gain_init
        )

        self.rotary = ProgressiveRotary(self.hd, base=rope_base)
        s = dim**-0.5
        for w in (self.c_q, self.c_k, self.c_v):
            _ortho(w.weight, s)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q = self.c_q(x).reshape(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rope_partial(q, cos, sin, self.rope_dims)
        k = apply_rope_partial(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]

        # GQA expansion
        if self.nkv != self.nh:
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if self.use_xsa:
            # XSA: prevent self-attention (each position cannot attend to itself)
            # Removes self-value bias — key contribution of @unnir's XSA
            diag_mask = torch.full(
                (T, T), float("-inf"), device=x.device, dtype=q.dtype
            )
            diag_mask.fill_diagonal_(0.0)
            # Causal mask combined with XSA diagonal block
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype),
                diagonal=1,
            )
            attn_mask = causal_mask + diag_mask
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False
            )
        elif HAS_FA3:
            # FlashAttention 3: [B, T, H, hd] layout
            q_ = q.transpose(1, 2).contiguous()
            k_ = k.transpose(1, 2).contiguous()
            v_ = v.transpose(1, 2).contiguous()
            y = flash_attn_func(q_, k_, v_, causal=True).transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().reshape(B, T, self.nh * self.hd)
        return self.proj(y)


# ── SmearGate MLP ─────────────────────────────────────────────────────────────
class SmearGateMLP(nn.Module):
    """relu(gate(x))^2 * up(x) — proven better than SwiGLU at this scale."""

    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = dim * mlp_mult
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.up = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        s = dim**-0.5
        _ortho(self.gate.weight, s)
        _ortho(self.up.weight, s)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.gate(x)).square() * self.up(x))


# ── Transformer Block ─────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        rope_dims,
        use_xsa,
        rope_base,
        qk_gain_init,
        layer_idx,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_dims, use_xsa, rope_base, qk_gain_init
        )
        self.mlp = SmearGateMLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack([torch.ones(dim), torch.zeros(dim)]).float()
        )
        # Depth gate: learnable per-layer scalar, init=1
        self.depth_gate = nn.Parameter(torch.ones(1, dtype=torch.float32))
        # LN Scale factor: 1/sqrt(layer_idx + 1) — reduces noise from deep layers
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None] * x + mix[1][None, None] * x0
        # Depth-gated residuals (novel: adaptive per-layer contribution)
        dg = self.depth_gate.to(x.dtype)
        attn_out = self.attn(self.attn_norm(x))
        # LN scale: multiplies norm output by 1/sqrt(layer+1)
        x = x + dg * self.attn_scale.to(x.dtype)[None, None] * attn_out * self.ln_scale
        x = (
            x
            + dg
            * self.mlp_scale.to(x.dtype)[None, None]
            * self.mlp(self.mlp_norm(x))
            * self.ln_scale
        )
        return x


# ── Full GPT ──────────────────────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.tie_embeddings = args.tie_embeddings
        self.logit_softcap = args.logit_softcap
        head_dim = args.model_dim // args.num_heads

        # Shift-Mixed Embedding (novel replacement for BigramHash)
        self.tok_emb = ShiftMixedEmbedding(
            args.vocab_size, args.model_dim, args.shift_alpha_init
        )

        self.num_enc = args.num_layers // 2
        self.num_dec = args.num_layers - self.num_enc
        self.num_skips = min(self.num_enc, self.num_dec)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skips, args.model_dim, dtype=torch.float32)
        )

        # Progressive RoPE dims per layer
        thirds = args.num_layers // 3
        self.rope_dims_list = (
            [args.rope_dims_early] * thirds
            + [args.rope_dims_mid] * thirds
            + [args.rope_dims_late] * (args.num_layers - 2 * thirds)
        )

        # XSA applied to last xsa_layers layers
        xsa_start = args.num_layers - args.xsa_layers

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=args.model_dim,
                    num_heads=args.num_heads,
                    num_kv_heads=args.num_kv_heads,
                    mlp_mult=args.mlp_mult,
                    rope_dims=self.rope_dims_list[i],
                    use_xsa=(i >= xsa_start),
                    rope_base=args.rope_base,
                    qk_gain_init=1.5,
                    layer_idx=i,
                )
                for i in range(args.num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None
            if args.tie_embeddings
            else CastedLinear(args.model_dim, args.vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        nn.init.normal_(self.tok_emb.weight.weight, std=args.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def _backbone(self, ids: Tensor) -> Tensor:
        x = self.tok_emb(ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0, skips = x, []
        for i in range(self.num_enc):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_dec):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None] * skips.pop()
            x = self.blocks[self.num_enc + i](x, x0)
        return self.final_norm(x)

    def forward_logits(self, ids: Tensor) -> Tensor:
        x = self._backbone(ids)
        lp = (
            F.linear(x, self.tok_emb.weight.weight)
            if self.tie_embeddings
            else self.lm_head(x)
        )
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)

    def forward(self, ids: Tensor, targets: Tensor) -> Tensor:
        logits = self.forward_logits(ids)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )

    def apply_late_qat(self):
        """Apply STE int6 fake-quantization to all large weight matrices."""
        with torch.no_grad():
            for name, p in self.named_parameters():
                if (
                    p.ndim == 2
                    and p.numel() > MAX_FLOAT_NUMEL
                    and not any(pat in name for pat in CONTROL_PATTERNS)
                    and not any(pat in name for pat in FP16_PATTERNS)
                ):
                    p.data = _fake_quant_int6(p.data)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────


def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    use_compile = bool(int(os.environ.get("USE_COMPILE", "1")))
    if use_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    gas = 8 // world_size
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not HAS_FA3:
        from torch.backends.cuda import (
            enable_cudnn_sdp,
            enable_flash_sdp,
            enable_math_sdp,
            enable_mem_efficient_sdp,
        )

        flash = sys.platform != "win32"
        enable_cudnn_sdp(False)
        enable_flash_sdp(flash)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(not flash)

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log(msg, console=True):
        if not master:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log(code, console=False)
    log(
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout,
        console=False,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE mismatch: {args.vocab_size} vs {int(sp.vocab_size())}"
        )

    ddir = Path(args.data_path).resolve()
    nshards = len(list(ddir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bb, hs, ib = build_sentencepiece_luts(sp, args.vocab_size, device)

    log(f"val_bpb:enabled  tokenizer:{args.tokenizer_path}  compressor:{COMPRESSOR}")
    log(f"train_shards:{nshards}  val_tokens:{val_tokens.numel() - 1}  FA3:{HAS_FA3}")

    # Build model
    base_model = GPT(args).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_fp32(base_model)

    compiled = (
        torch.compile(base_model, dynamic=False, fullgraph=True)
        if use_compile
        else base_model
    )
    if not use_compile:
        log("[info] torch.compile disabled")
    model = (
        DDP(compiled, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled
    )

    # ── EMA shadow model ──────────────────────────────────────────────────────
    ema_model = copy.deepcopy(base_model)
    ema_model.requires_grad_(False)

    def update_ema(decay: float):
        with torch.no_grad():
            for p_ema, p_base in zip(ema_model.parameters(), base_model.parameters()):
                p_ema.data.mul_(decay).add_(p_base.data, alpha=1.0 - decay)

    # ── Optimizers ────────────────────────────────────────────────────────────
    block_np = list(base_model.blocks.named_parameters())
    matrix_p = [
        p
        for n, p in block_np
        if p.ndim == 2 and not any(pat in n for pat in CONTROL_PATTERNS)
    ]
    scalar_p = [
        p
        for n, p in block_np
        if p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_p.append(base_model.skip_weights)

    tok_lr = args.tied_embed_lr
    emb_params = list(base_model.tok_emb.parameters())

    opt_tok = torch.optim.AdamW(
        [{"params": emb_params, "lr": tok_lr, "base_lr": tok_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )

    opt_mousse = Mousse(
        matrix_p,
        lr=args.matrix_lr,
        momentum=args.mousse_momentum,
        backend_steps=args.mousse_backend_steps,
        beta2=args.mousse_beta2,
        wd=args.mousse_wd,
    )
    for g in opt_mousse.param_groups:
        g["base_lr"] = args.matrix_lr

    opt_scalar = torch.optim.AdamW(
        [{"params": scalar_p, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizers = [opt_tok, opt_mousse, opt_scalar]

    nparams = sum(p.numel() for p in base_model.parameters())
    rope_str = (
        f"early={args.rope_dims_early} mid={args.rope_dims_mid} "
        f"late={args.rope_dims_late}"
    )
    log(
        f"params:{nparams}  layers:{args.num_layers}  dim:{args.model_dim}  "
        f"heads:{args.num_heads}/{args.num_kv_heads}  mlp_mult:{args.mlp_mult}"
    )
    log(f"xsa_layers:{args.xsa_layers}  progressive_rope:{rope_str}")
    log(f"ema_decay:{args.ema_decay}  late_qat_frac:{args.late_qat_frac}")
    log(
        f"batch:{args.train_batch_tokens}  seq:{args.train_seq_len}  "
        f"world:{world_size}  gas:{gas}  compile:{use_compile}"
    )

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad():
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    mwms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if mwms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if ws <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        wd_ms = args.warmdown_iters * step_ms
        rem = max(mwms - elapsed_ms, 0.0)
        return rem / max(wd_ms, 1e-9) if rem <= wd_ms else 1.0

    # Late QAT: enable in last late_qat_frac of total steps
    def should_late_qat(step, elapsed_ms):
        if mwms is not None:
            return elapsed_ms > mwms * (1.0 - args.late_qat_frac)
        return step > args.iterations * (1.0 - args.late_qat_frac)

    # ── Warmup ────────────────────────────────────────────────────────────────
    if args.warmup_steps > 0:
        init_sd = {
            n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()
        }
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws_i in range(args.warmup_steps):
            zero_grad()
            for micro in range(gas):
                if distributed:
                    model.require_backward_grad_sync = micro == gas - 1
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, gas
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    loss = model(x, y)
                (loss / gas).backward()
            for o in optimizers:
                o.step()
            zero_grad()
            if (
                args.warmup_steps <= 20
                or (ws_i + 1) % 10 == 0
                or ws_i + 1 == args.warmup_steps
            ):
                log(f"warmup:{ws_i + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_sd, strict=True)
        for o, st in zip(optimizers, init_opt, strict=True):
            o.load_state_dict(st)
        zero_grad()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device
        )
        # Re-sync EMA after warmup reset
        ema_model.load_state_dict(base_model.state_dict())

    # ── Main training loop ────────────────────────────────────────────────────
    train_ms = 0.0
    stop_after: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last = step == args.iterations or (
            stop_after is not None and step >= stop_after
        )
        elapsed_ms = train_ms + 1000.0 * (time.perf_counter() - t0)

        if not last and args.val_loss_every > 0 and step % args.val_loss_every == 0:
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(
                args, model, rank, world_size, device, gas, val_tokens, bb, hs, ib
            )
            log(
                f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                f"train_time:{train_ms:.0f}ms step_avg:{train_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last:
            if stop_after is not None and step < args.iterations:
                log(f"stopping_early step:{step}")
            break

        scale = lr_mul(step, elapsed_ms)
        zero_grad()
        tl = torch.zeros((), device=device)
        for micro in range(gas):
            if distributed:
                model.require_backward_grad_sync = micro == gas - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, gas
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            tl += loss.detach()
            (loss / gas).backward()
        tl /= gas

        # Mousse momentum warmup
        frac = (
            min(step / args.mousse_warmup_steps, 1.0)
            if args.mousse_warmup_steps > 0
            else 1.0
        )
        cur_mom = (1 - frac) * args.mousse_warmup_start + frac * args.mousse_momentum
        for g in opt_mousse.param_groups:
            g["momentum"] = cur_mom

        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers:
            o.step()
        zero_grad()

        # EMA update every step
        update_ema(args.ema_decay)

        # Late QAT: apply STE fake-quant to base model weights
        if should_late_qat(step, elapsed_ms):
            base_model.apply_late_qat()

        step += 1
        approx = train_ms + 1000.0 * (time.perf_counter() - t0)

        if args.train_log_every > 0 and (
            step <= 10 or step % args.train_log_every == 0 or stop_after is not None
        ):
            log(
                f"step:{step}/{args.iterations} train_loss:{tl.item():.4f} "
                f"train_time:{approx:.0f}ms step_avg:{approx / step:.2f}ms"
            )

        reached = mwms is not None and approx >= mwms
        if distributed and mwms is not None:
            rc = torch.tensor(int(reached), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached = bool(rc.item())
        if stop_after is None and reached:
            stop_after = step

    log(f"peak_mem:{torch.cuda.max_memory_allocated() // 1024 // 1024}MiB")

    # ── Use EMA model for final eval and serialization ────────────────────────
    log("\n[EMA] loading EMA weights for final eval...")
    base_model.load_state_dict(ema_model.state_dict())

    # Final sliding-window eval (pre-quant)
    log("[final eval] sliding-window (stride=64, context≥960 tokens)...")
    torch.cuda.synchronize()
    tsw = time.perf_counter()
    swl, swb = eval_val_sliding_window(
        args, base_model, rank, world_size, device, val_tokens, bb, hs, ib
    )
    torch.cuda.synchronize()
    log(
        f"pre_quant val_loss:{swl:.4f} val_bpb:{swb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - tsw):.0f}ms"
    )

    # Serialize
    if master:
        nb = save_artifact(base_model, "final_model.ptz")
        cb = len(code.encode("utf-8"))
        total = nb + cb
        log(f"\nartifact:{nb}B  code:{cb}B  total:{total}B  ({total / 1e6:.3f}MB)")
        log("OK: within 16MB" if total <= 16_000_000 else "WARNING: EXCEEDS 16MB")

    # Roundtrip validation
    if distributed:
        dist.barrier()
    load_artifact("final_model.ptz", base_model)
    torch.cuda.synchronize()
    trt = time.perf_counter()
    rtl, rtb = eval_val_sliding_window(
        args, base_model, rank, world_size, device, val_tokens, bb, hs, ib
    )
    torch.cuda.synchronize()
    log(
        f"roundtrip val_loss:{rtl:.4f} val_bpb:{rtb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - trt):.0f}ms"
    )
    log(f"roundtrip_exact val_loss:{rtl:.8f} val_bpb:{rtb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
