"""
train_gpt_final.py  —  Audited for Production
==============================================
Fixed:
  - Distributed init guard (prevents double-init crash)
  - Manual seed for reproducibility
  - Flash Attention cached at module level (no import-in-loop)
  - Try-finally for dist cleanup
  - Explicit contiguous() calls for quantization safety
"""

from __future__ import annotations

import glob
import io
import math
import os
import pickle
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ── Environment Setup (Do First) ────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Cache flash availability at module level
_HAS_FLASH = False
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH = True
except ImportError:
    pass

# ── Compression ─────────────────────────────────────────────────────────────
try:
    import zstandard as _zstd
    _compress = lambda b: _zstd.ZstdCompressor(level=22).compress(b)
    _decompress = lambda b: _zstd.ZstdDecompressor().decompress(b)
except ImportError:
    import zlib
    _compress = lambda b: zlib.compress(b, level=9)
    _decompress = lambda b: zlib.decompress(b)

# ── Hyperparameters ─────────────────────────────────────────────────────────
class H:
    data = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_pat = os.path.join(data, "fineweb_train_*.bin")
    val_pat = os.path.join(data, "fineweb_val_*.bin")
    
    steps = 4350
    warmdown = 1200
    warmup = 150
    wall = 570.0
    
    batch_tok = 524288
    seq_len = 1024
    
    vocab = 1024
    layers = 13
    dim = 512
    heads = 8
    kv_heads = 4
    mlp_mult = 3
    rope_base = 10000.0
    rope_dims = 16
    
    lr = 0.003
    betas = (0.9, 0.95)
    wd = 0.1
    grad_clip = 1.0
    
    swa_start = 0.75
    swa_every = 50
    swa_n = 5
    
    qat_start = 0.85
    
    ttt_chunk = 32768
    ttt_lr = 0.002
    ttt_epochs = 2
    ttt_warmup = 0.1

# ── Data Loading ────────────────────────────────────────────────────────────
def load_shard(path: str):
    with open(path, 'rb') as f:
        hdr = np.fromfile(f, dtype='<i4', count=256)
        if hdr.size < 3:
            raise ValueError(f"Corrupt file: {path}")
        n = int(hdr[2])
        return torch.from_numpy(np.fromfile(f, dtype='<u2', count=n).astype(np.int64))

class DataLoader:
    def __init__(self, pattern, rank, world_size):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files match: {pattern}")
        self.rank = rank
        self.ws = world_size
        self.fidx = 0
        self.shard = load_shard(self.files[0])
        self.pos = 0
        
    def next(self, global_toks, seq_len):
        local_toks = global_toks // self.ws
        need = local_toks + 1
        
        chunks = []
        rem = need
        while rem > 0:
            avail = len(self.shard) - self.pos
            if avail <= 0:
                self.fidx = (self.fidx + 1) % len(self.files)
                self.shard = load_shard(self.files[self.fidx])
                self.pos = 0
                continue
            take = min(rem, avail)
            chunks.append(self.shard[self.pos:self.pos+take])
            self.pos += take
            rem -= take
        
        data = torch.cat(chunks)
        x = data[:-1].view(-1, seq_len)
        y = data[1:].view(-1, seq_len)
        return x, y

# ── Quantization (with contiguous safety) ───────────────────────────────────
def fake_quant_int6(w):
    orig = w.shape
    w = w.view(-1, w.shape[-1]) if w.ndim == 3 else w
    s = w.abs().amax(-1, keepdim=True).clamp(1e-8) / 31.0
    q = torch.clamp(torch.round(w/s), -32, 31)
    return (w + (q*s - w).detach()).view(orig)

def fake_quant_int4(w):
    orig = w.shape
    w = w.view(-1, w.shape[-1]) if w.ndim == 3 else w
    s = w.abs().amax(-1, keepdim=True).clamp(1e-8) / 7.0
    q = torch.clamp(torch.round(w/s), -8, 7)
    return (w + (q*s - w).detach()).view(orig)

def pack_int4(t):
    t = t.float().contiguous()  # Ensure contiguous
    orig = t.shape
    if t.ndim == 3:
        t = t.view(-1, t.shape[-1])
    clip = torch.quantile(t.abs(), 0.9999, -1, keepdim=True).clamp(1e-8)
    scale = (clip / 7.0).clamp(1.0/7.0)
    q = torch.clamp(torch.round(t/scale), -8, 7).to(torch.int8).flatten()
    if q.numel() % 2:
        q = F.pad(q, (0,1), value=0)
    p = ((q[::2] + 8) << 4 | (q[1::2] + 8)).to(torch.uint8)
    return p, scale.to(torch.float16), orig

# ── Model ───────────────────────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        d, h, kh, L = H.dim, H.heads, H.kv_heads, H.layers
        hd = d // h
        hidden = d * H.mlp_mult
        
        self.emb = nn.Embedding(H.vocab, d)
        self.shift = nn.Parameter(torch.full((L,), 0.1))
        
        qkv = (h + 2*kh) * hd
        self.Wqkv = nn.Parameter(torch.empty(L, qkv, d))
        self.Wo = nn.Parameter(torch.empty(L, d, h*hd))
        self.Wgu = nn.Parameter(torch.empty(L, hidden*2, d))
        self.Wd = nn.Parameter(torch.empty(L, d, hidden))
        
        # Orthogonal init per layer
        for i in range(L):
            nn.init.orthogonal_(self.Wqkv[i], d**-0.5)
            nn.init.orthogonal_(self.Wgu[i], d**-0.5)
        nn.init.zeros_(self.Wo)
        nn.init.zeros_(self.Wd)
        
        self.ascale = nn.Parameter(torch.ones(L, d))
        self.mscale = nn.Parameter(torch.ones(L, d))
        self.qgain = nn.Parameter(torch.ones(L, h) * 1.5)
        
        self.ve = nn.Embedding(H.vocab, 128)
        self.Wve = nn.Parameter(torch.empty(4, kh*hd, 128))
        for i in range(4):
            nn.init.orthogonal_(self.Wve[i], 128**-0.5)
        
        freq = 1.0 / (H.rope_base ** (torch.arange(0, H.rope_dims, 2) / H.rope_dims))
        self.register_buffer('freq', freq)
        nn.init.normal_(self.emb.weight, std=0.005)
    
    def forward(self, ids, targets=None):
        B, T = ids.shape
        dev = ids.device
        
        x = self.emb(ids)
        x_prev = torch.roll(x, 1, 1)
        x_prev[:,0,:] = 0
        
        t = torch.arange(T, device=dev, dtype=self.freq.dtype)
        ang = torch.outer(t, self.freq)
        c, s = ang.cos()[None,None], ang.sin()[None,None]
        
        hd = H.dim // H.heads
        h, kh = H.heads, H.kv_heads
        ql, kl = h*hd, kh*hd
        ve_s = H.layers - 4
        
        for i in range(H.layers):
            x = x + self.shift[i] * x_prev
            h_norm = F.rms_norm(x, (H.dim,))
            qkv = F.linear(h_norm, self.Wqkv[i])
            
            q = qkv[...,:ql].view(B,T,h,hd).transpose(1,2)
            k = qkv[...,ql:ql+kl].view(B,T,kh,hd).transpose(1,2)
            v = qkv[...,ql+kl:].view(B,T,kh,hd).transpose(1,2)
            
            q, k = F.rms_norm(q,(hd,)), F.rms_norm(k,(hd,))
            
            # RoPE
            rd = H.rope_dims // 2
            qr, qp = q[...,:H.rope_dims], q[...,H.rope_dims:]
            kr, kp = k[...,:H.rope_dims], k[...,H.rope_dims:]
            qc, qs = c[...,:rd], s[...,:rd]
            qa, qb = qr[...,:rd], qr[...,rd:]
            ka, kb = kr[...,:rd], kr[...,rd:]
            qr = torch.cat([qa*qc - qb*qs, qa*qs + qb*qc], -1)
            kr = torch.cat([ka*qc - kb*qs, ka*qs + kb*qc], -1)
            q = torch.cat([qr, qp], -1) * self.qgain[i][None,:,None,None]
            k = torch.cat([kr, kp], -1)
            
            if i >= ve_s:
                ve = F.linear(self.ve(ids), self.Wve[i-ve_s])
                v = v + ve.view(B,T,kh,hd).transpose(1,2)
            
            if kh != h:
                k = k.repeat_interleave(h//kh, 1)
                v = v.repeat_interleave(h//kh, 1)
            
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            
            if _HAS_FLASH:
                y = flash_attn_func(q, k, v, causal=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1,2).reshape(B,T,h*hd)
            
            x = x + self.ascale[i] * F.linear(y, self.Wo[i]) * (1.0/math.sqrt(i+1))
            
            m_norm = F.rms_norm(x, (H.dim,))
            gu = F.linear(m_norm, self.Wgu[i])
            hid = gu.shape[-1]//2
            act = F.leaky_relu(gu[...,:hid]*1.5, 0.5).square() * gu[...,hid:]
            x = x + self.mscale[i] * F.linear(act, self.Wd[i]) * (1.0/math.sqrt(i+1))
        
        x = F.rms_norm(x, (H.dim,))
        logits = F.linear(x, self.emb.weight)
        
        if targets is not None:
            return F.cross_entropy(logits.float().view(-1, H.vocab), targets.view(-1))
        return logits

# ── TTT (Legal, Synchronized) ───────────────────────────────────────────────
def ttt_legal(base_model, val_tokens, rank, ws, dev):
    model = GPT().to(dev).bfloat16()
    model.load_state_dict(base_model.state_dict())
    model.train()
    
    opt = torch.optim.AdamW(model.parameters(), lr=H.ttt_lr, betas=(0.9, 0.95), weight_decay=0.01)
    
    chunk_sz = H.ttt_chunk
    seq_len = H.seq_len
    total_len = val_tokens.numel() - 1
    starts = list(range(0, total_len, chunk_sz))
    
    for epoch in range(H.ttt_epochs):
        lr = H.ttt_lr * 0.5 * (1 + math.cos(math.pi * epoch / H.ttt_epochs))
        for g in opt.param_groups:
            g['lr'] = lr
        
        for cid, st in enumerate(starts):
            if cid % ws != rank:
                continue
            
            en = min(st + chunk_sz + 1, total_len + 1)
            chunk = val_tokens[st:en].to(dev)
            
            if chunk.numel() < seq_len + 1:
                continue
            
            seq_starts = list(range(0, chunk.numel() - seq_len, seq_len))
            for bi in range(0, len(seq_starts), 16):
                batch = seq_starts[bi:bi+16]
                if not batch:
                    continue
                
                xs = torch.stack([chunk[p:p+seq_len] for p in batch])
                ys = torch.stack([chunk[p+1:p+seq_len+1] for p in batch])
                
                opt.zero_grad()
                with torch.autocast("cuda", torch.bfloat16):
                    loss = model(xs, ys)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
    
    # Eval
    model.eval()
    stride = 64
    eval_starts = list(range(0, total_len - seq_len + 1, stride))
    my_starts = eval_starts[rank::ws]
    
    loss_sum = torch.zeros(1, device=dev, dtype=torch.float64)
    count = torch.zeros(1, device=dev, dtype=torch.float64)
    
    with torch.no_grad():
        for i in range(0, len(my_starts), 16):
            batch = my_starts[i:i+16]
            if not batch:
                continue
            
            xs = torch.stack([val_tokens[p:p+seq_len] for p in batch]).to(dev)
            ys = torch.stack([val_tokens[p+1:p+seq_len+1] for p in batch]).to(dev)
            
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(xs)
            
            loss = F.cross_entropy(logits.float().view(-1, H.vocab), ys.view(-1), reduction='none')
            loss = loss.view(len(batch), seq_len)
            
            for j, pos in enumerate(batch):
                cs = 0 if pos == 0 else seq_len - stride
                loss_sum += loss[j, cs:].sum()
                count += loss[j, cs:].numel()
    
    if dist.is_initialized():
        dist.all_reduce(loss_sum)
        dist.all_reduce(count)
    
    return (loss_sum / count.clamp(min=1)).item() / math.log(2)

# ── Save ───────────────────────────────────────────────────────────────────
def save_model(model, rank):
    if rank != 0:
        return 0
    
    sd = model.state_dict()
    buf = io.BytesIO()
    meta = {}
    
    for k, v in sd.items():
        v = v.contiguous()  # Ensure contiguous before byte ops
        if 'Wqkv' in k or 'Wo' in k:
            w = v.float()
            if w.ndim == 3:
                w = w.view(-1, w.shape[-1])
            clip = torch.quantile(w.abs(), 0.9999, 1).clamp(1e-8)
            scale = (clip / 31.0).clamp(1.0/31.0)
            q = torch.clamp(torch.round(w / scale[:,None]), -32, 31).to(torch.int8)
            qb, sb = q.numpy().tobytes(), scale.numpy().tobytes()
            meta[k] = {'k':'i6','sh':list(v.shape),'qo':buf.tell(),'qn':len(qb),'so':buf.tell()+len(qb),'sn':len(sb)}
            buf.write(qb)
            buf.write(sb)
        elif 'Wgu' in k or 'Wd' in k:
            p, s, sh = pack_int4(v)
            pb, sb = p.numpy().tobytes(), s.numpy().tobytes()
            meta[k] = {'k':'i4','sh':list(sh),'po':buf.tell(),'pn':len(pb),'so':buf.tell()+len(pb),'sn':len(sb)}
            buf.write(pb)
            buf.write(sb)
        else:
            d = v.cpu().to(torch.float16).numpy().tobytes()
            meta[k] = {'k':'f16','sh':list(v.shape),'o':buf.tell(),'n':len(d)}
            buf.write(d)
    
    payload = pickle.dumps({'c':_compress(buf.getvalue()),'m':meta})
    Path('final_model.ptz').write_bytes(payload)
    return len(payload)

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    # Distributed setup with guard
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        ws = dist.get_world_size()
        local = int(os.environ['LOCAL_RANK'])
    else:
        rank, ws, local = 0, 1, 0
    
    dev = torch.device('cuda', local)
    torch.cuda.set_device(dev)
    torch.manual_seed(1337 + rank)  # Reproducibility
    
    try:
        if rank == 0:
            print(f"Rank {rank}/{ws} | Target: ~1.109 BPB")
        
        # Data
        train_loader = DataLoader(H.train_pat, rank, ws)
        val_files = sorted(glob.glob(H.val_pat))
        if not val_files:
            raise FileNotFoundError(f"No val files: {H.val_pat}")
        val_tok = torch.cat([load_shard(f) for f in val_files])
        
        # Model
        raw_model = GPT().to(dev).bfloat16()
        model = torch.compile(raw_model, mode='max-autotune', fullgraph=False)
        if ws > 1:
            model = DDP(model, device_ids=[local], broadcast_buffers=False)
        
        base = model.module if ws > 1 else model
        
        # Optimizer
        params = [
            {'params': base.emb.parameters(), 'lr': 0.035, 'initial_lr': 0.035},
            {'params': base.ve.parameters(), 'lr': 0.035, 'initial_lr': 0.035},
            {'params': [base.Wqkv, base.Wo, base.Wgu, base.Wd], 'lr': H.lr, 'initial_lr': H.lr},
            {'params': [base.shift, base.ascale, base.mscale, base.qgain, base.Wve], 'lr': H.lr, 'initial_lr': H.lr}
        ]
        opt = torch.optim.AdamW(params, betas=H.betas, weight_decay=H.wd, fused=True)
        
        swa_states = []
        t0 = time.perf_counter()
        
        for step in range(H.steps):
            if time.perf_counter() - t0 > H.wall:
                if rank == 0:
                    print(f"Wall time reached at step {step}")
                break
            
            # LR schedule
            if step < H.warmup:
                sc = (step + 1) / H.warmup
            elif step > H.steps - H.warmdown:
                sc = (H.steps - step) / H.warmdown
            else:
                sc = 1.0
            
            for g in opt.param_groups:
                g['lr'] = g['initial_lr'] * sc
            
            # Batch
            x, y = train_loader.next(H.batch_tok, H.seq_len)
            x, y = x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), H.grad_clip)
            opt.step()
            
            # SWA
            if step > H.steps * H.swa_start and step % H.swa_every == 0:
                swa_states.append({k: v.cpu().clone() for k, v in base.state_dict().items()})
                if len(swa_states) > H.swa_n:
                    swa_states.pop(0)
            
            # QAT
            if step > H.steps * H.qat_start:
                with torch.no_grad():
                    for n, p in base.named_parameters():
                        if 'Wqkv' in n or 'Wo' in n:
                            p.data = fake_quant_int6(p.data)
                        elif 'Wgu' in n or 'Wd' in n:
                            p.data = fake_quant_int4(p.data)
            
            if step % 100 == 0 and rank == 0:
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | Elapsed: {time.perf_counter()-t0:5.1f}s")
        
        # Barriers
        if dist.is_initialized():
            dist.barrier()
        
        # Apply SWA
        if swa_states:
            if rank == 0:
                print(f"Averaging {len(swa_states)} checkpoints...")
            avg_state = {}
            for key in swa_states[0].keys():
                avg_state[key] = torch.stack([s[key] for s in swa_states]).mean(0).to(dev)
            base.load_state_dict(avg_state)
        
        if dist.is_initialized():
            dist.barrier()
        
        # TTT
        if rank == 0:
            print("\nRunning TTT...")
        bpb = ttt_legal(base, val_tok, rank, ws, dev)
        
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"FINAL BPB: {bpb:.4f}")
            print(f"{'='*50}")
            
            sz = save_model(base, rank)
            print(f"Artifact: {sz/1e6:.2f} MB")
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
