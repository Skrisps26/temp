"""
Parameter Golf: Matching SOTA recipe exactly.
11L d=512, 3x MLP, LeakyReLU(0.5)², XSA4, BigramHash1536, VE128,
Partial RoPE 16/64, U-Net Skips, EMA+SWA, GPTQ-lite int6+LZMA,
SmearGate, Legal Score-First TTT (SGD, 3ep, 32K chunks).
"""
from __future__ import annotations
import copy,glob,io,lzma,math,os,pickle,random,subprocess,sys,time,uuid
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor,nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ── compression: LZMA (matching SOTA) ────────────────────────────────────────
compress_bytes = lambda b: lzma.compress(b, preset=9|lzma.PRESET_EXTREME)
decompress_bytes = lambda b: lzma.decompress(b)
COMPRESSOR = "lzma-9e"

# ── hyperparameters (exact SOTA values) ───────────────────────────────────────
class H:
    data_path = os.environ.get("DATA_PATH","./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path,"fineweb_train_*.bin")
    val_files = os.path.join(data_path,"fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH","./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID",str(uuid.uuid4()))
    seed = int(os.environ.get("SEED",1337))
    iterations = int(os.environ.get("ITERATIONS",9000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS",3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS",20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS",524288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN",1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS",600.0))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY",0))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE",524288))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY",200))
    vocab_size = int(os.environ.get("VOCAB_SIZE",1024))
    num_layers = int(os.environ.get("NUM_LAYERS",11))
    model_dim = int(os.environ.get("MODEL_DIM",512))
    num_heads = int(os.environ.get("NUM_HEADS",8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS",4))
    mlp_mult = int(os.environ.get("MLP_MULT",3))
    tie_embeddings = True
    rope_base = float(os.environ.get("ROPE_BASE",10000.0))
    rope_dims = int(os.environ.get("ROPE_DIMS",16))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP",30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT",1.5))
    xsa_last_n = int(os.environ.get("XSA_LAST_N",4))
    ln_scale = int(os.environ.get("LN_SCALE",1))
    bigram_vocab = int(os.environ.get("BIGRAM_VOCAB_SIZE",1536))
    bigram_dim = int(os.environ.get("BIGRAM_DIM",128))
    smear_enabled = int(os.environ.get("SMEAR_ENABLED",1))
    ve_enabled = int(os.environ.get("VE_ENABLED",1))
    ve_dim = int(os.environ.get("VE_DIM",128))
    ve_layers_str = os.environ.get("VE_LAYERS","9,10")
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR",0.035))
    matrix_lr = float(os.environ.get("MATRIX_LR",0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR",0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM",0.99))
    muon_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START",0.92))
    muon_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS",1500))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS",10))
    muon_wd = float(os.environ.get("MUON_WD",0.04))
    adam_wd = float(os.environ.get("ADAM_WD",0.04))
    ema_enabled = int(os.environ.get("EMA_ENABLED",1))
    ema_decay = float(os.environ.get("EMA_DECAY",0.997))
    swa_enabled = int(os.environ.get("SWA_ENABLED",1))
    swa_every = int(os.environ.get("SWA_EVERY",50))
    late_qat = int(os.environ.get("LATE_QAT",1))
    late_qat_frac = float(os.environ.get("LATE_QAT_THRESHOLD",0.15))
    ttt_enabled = int(os.environ.get("TTT_ENABLED",1))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS",3))
    ttt_lr = float(os.environ.get("TTT_LR",0.002))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS",32768))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS",32))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM",0.9))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP",1.0))
    eval_stride = int(os.environ.get("EVAL_STRIDE",64))
    @staticmethod
    def ve_layer_set():
        return set(int(x) for x in H.ve_layers_str.split(",") if x.strip())

CONTROL_PATTERNS = ("attn_scale","mlp_scale","q_gain","skip_weight","resid_mix","smear","bigram","ve_")

# ── Muon optimizer ────────────────────────────────────────────────────────────
@torch.compile
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-7):
    a,b,c = 3.4445,-4.7750,2.0315
    X = G.bfloat16(); X /= X.norm()+eps
    tr = G.size(0)>G.size(1)
    if tr: X=X.T
    for _ in range(steps):
        A=X@X.T; B=b*A+c*A@A; X=a*X+B@X
    return X.T if tr else X

class Muon(torch.optim.Optimizer):
    def __init__(s,params,lr,momentum,backend_steps=10,nesterov=True,wd=0.0):
        super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,wd=wd))
    @torch.no_grad()
    def step(s):
        dd=dist.is_available() and dist.is_initialized()
        ws=dist.get_world_size() if dd else 1; rk=dist.get_rank() if dd else 0
        for g in s.param_groups:
            lr,mom,ns,nest,wd=g["lr"],g["momentum"],g["backend_steps"],g["nesterov"],g["wd"]
            ps=g["params"]
            if not ps: continue
            tot=sum(p.numel() for p in ps)
            fl=torch.zeros(tot,device=ps[0].device,dtype=torch.bfloat16); c=0
            for i,p in enumerate(ps):
                if wd>0: p.mul_(1.0-lr*wd)
                if i%ws==rk and p.grad is not None:
                    gi=p.grad; st=s.state[p]
                    if "buf" not in st: st["buf"]=torch.zeros_like(gi)
                    buf=st["buf"]; buf.mul_(mom).add_(gi)
                    ge=gi.add(buf,alpha=mom) if nest else buf
                    go=zeropower_via_newtonschulz5(ge,steps=ns)
                    go*=max(1,go.size(0)/go.size(1))**0.5
                    fl[c:c+p.numel()]=go.reshape(-1)
                c+=p.numel()
            if dd: dist.all_reduce(fl,op=dist.ReduceOp.SUM)
            c=0
            for p in ps:
                p.add_(fl[c:c+p.numel()].view_as(p).to(p.dtype),alpha=-lr); c+=p.numel()

# ── int6 quantization (GPTQ-lite) ────────────────────────────────────────────
INT6_CLIP_Q = 0.9999
def _quant_int6_row(t):
    t32=t.float()
    if t32.ndim==2:
        clip=torch.quantile(t32.abs(),INT6_CLIP_Q,dim=1).clamp(min=1e-8)
        clipped=torch.clamp(t32,-clip[:,None],clip[:,None])
        scale=(clip/31.0).clamp(min=1.0/31.0)
        q=torch.clamp(torch.round(clipped/scale[:,None]),-32,31).to(torch.int8)
        return q.contiguous(),scale.to(torch.float16).contiguous()
    clip=float(torch.quantile(t32.abs().flatten(),INT6_CLIP_Q).item()) if t32.numel() else 0.0
    sv=clip/31.0 if clip>0 else 1.0
    q=torch.clamp(torch.round(torch.clamp(t32,-clip,clip)/sv),-32,31).to(torch.int8)
    return q.contiguous(),torch.tensor(sv,dtype=torch.float16)

def _fake_quant_int6(t):
    q,s=_quant_int6_row(t)
    if s.ndim>0: return (q.float()*s.float().view(q.shape[0],*([1]*(q.ndim-1)))).to(t.dtype)
    return (q.float()*float(s.item())).to(t.dtype)

def quantize_state_dict(sd):
    meta,buf={},io.BytesIO()
    for k,v in sd.items():
        is_ctrl=any(p in k for p in CONTROL_PATTERNS) or v.numel()<=65536
        if is_ctrl:
            d=v.cpu().to(torch.float16).numpy().tobytes()
            meta[k]={"k":"f","s":list(v.shape),"o":buf.tell(),"n":len(d)}; buf.write(d)
        else:
            q,s=_quant_int6_row(v.cpu())
            qb,sb=q.numpy().tobytes(),s.numpy().tobytes()
            meta[k]={"k":"q","s":list(v.shape),"ss":list(s.shape),"qo":buf.tell(),"qn":len(qb),"so":buf.tell()+len(qb),"sn":len(sb)}
            buf.write(qb); buf.write(sb)
    return meta,buf.getvalue()

def dequantize_state_dict(meta,raw):
    sd={}
    for k,m in meta.items():
        if m["k"]=="f":
            arr=np.frombuffer(raw[m["o"]:m["o"]+m["n"]],dtype=np.float16).copy()
            sd[k]=torch.from_numpy(arr).to(torch.bfloat16).view(m["s"])
        else:
            q=torch.from_numpy(np.frombuffer(raw[m["qo"]:m["qo"]+m["qn"]],dtype=np.int8).copy())
            s=torch.from_numpy(np.frombuffer(raw[m["so"]:m["so"]+m["sn"]],dtype=np.float16).copy())
            q=q.view(m["ss"][0] if m["ss"] else 1,-1)
            s=s.view(m["ss"])
            if s.ndim>0: sd[k]=(q.float()*s.float().view(q.shape[0],*([1]*(q.ndim-1)))).to(torch.bfloat16).view(m["s"])
            else: sd[k]=(q.float()*float(s.item())).to(torch.bfloat16).view(m["s"])
    return sd

# ── Data loading & BPB ───────────────────────────────────────────────────────
def build_sp_luts(sp,vs,dev):
    sv=int(sp.vocab_size()); ts=max(sv,vs)
    bb=np.zeros(ts,dtype=np.int16); hs=np.zeros(ts,dtype=np.bool_); ib=np.ones(ts,dtype=np.bool_)
    for t in range(sv):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
        ib[t]=False
        if sp.is_byte(t): bb[t]=1; continue
        pc=sp.id_to_piece(t)
        if pc.startswith("▁"): hs[t]=True; pc=pc[1:]
        bb[t]=len(pc.encode("utf-8"))
    return (torch.tensor(bb,dtype=torch.int16,device=dev),torch.tensor(hs,dtype=torch.bool,device=dev),torch.tensor(ib,dtype=torch.bool,device=dev))

def load_shard(f):
    hdr=np.fromfile(f,dtype="<i4",count=256); n=int(hdr[2])
    return torch.from_numpy(np.fromfile(f,dtype="<u2",count=n,offset=256*4).astype(np.uint16,copy=False))

class TokenStream:
    def __init__(s,pat):
        s.files=sorted(glob.glob(pat))
        if not s.files: raise FileNotFoundError(pat)
        s.fi=0; s.pos=0; s.tokens=load_shard(s.files[0])
    def _adv(s): s.fi=(s.fi+1)%len(s.files); s.tokens=load_shard(s.files[s.fi]); s.pos=0
    def take(s,n):
        cs=[]; r=n
        while r>0:
            a=s.tokens.numel()-s.pos
            if a<=0: s._adv(); continue
            k=min(r,a); cs.append(s.tokens[s.pos:s.pos+k]); s.pos+=k; r-=k
        return cs[0] if len(cs)==1 else torch.cat(cs)

class DistTokenLoader:
    def __init__(s,pat,rk,ws,dev): s.rk=rk; s.ws=ws; s.dev=dev; s.stream=TokenStream(pat)
    def next_batch(s,gt,sl,acc=1):
        lt=gt//(s.ws*acc); sp=lt+1; ch=s.stream.take(sp*s.ws)
        lo=ch[s.rk*sp:s.rk*sp+sp].to(dtype=torch.int64)
        x,y=lo[:-1].reshape(-1,sl),lo[1:].reshape(-1,sl)
        return x.to(s.dev,non_blocking=True),y.to(s.dev,non_blocking=True)

def load_val_tokens(pat,sl):
    toks=torch.cat([load_shard(f) for f in sorted(glob.glob(pat))]).contiguous()
    u=((toks.numel()-1)//sl)*sl; return toks[:u+1]

# ── Model ─────────────────────────────────────────────────────────────────────
class CastedLinear(nn.Linear):
    def forward(s,x): return F.linear(x,s.weight.to(x.dtype),s.bias.to(x.dtype) if s.bias is not None else None)

class Rotary(nn.Module):
    def __init__(s,dim,base=10000.0):
        super().__init__()
        s.register_buffer("inv_freq",1.0/(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim)),persistent=False)
        s._cache=None
    def forward(s,sl,dev,dt):
        if s._cache is None or s._cache[2]!=sl or s._cache[3]!=dev:
            t=torch.arange(sl,device=dev,dtype=s.inv_freq.dtype)
            f=torch.outer(t,s.inv_freq.to(dev))
            s._cache=(f.cos()[None,None].detach(),f.sin()[None,None].detach(),sl,dev)
        return s._cache[0].to(dt),s._cache[1].to(dt)

def apply_rope_partial(x,cos,sin,rd):
    if rd<=0: return x
    xr,xp=x[...,:rd],x[...,rd:]
    half=rd//2; x1,x2=xr[...,:half],xr[...,half:]
    rot=torch.cat([x1*cos-x2*sin,x1*sin+x2*cos],dim=-1)
    return torch.cat([rot,xp],dim=-1)

class BigramHash(nn.Module):
    def __init__(s,vs,bv,bd,md):
        super().__init__(); s.bv=bv
        s.embed=nn.Embedding(bv,bd); s.proj=CastedLinear(bd,md,bias=False)
        nn.init.normal_(s.embed.weight,std=0.005); nn.init.zeros_(s.proj.weight)
    def forward(s,ids):
        pr=torch.cat([torch.zeros_like(ids[:,:1]),ids[:,:-1]],dim=1)
        return s.proj(s.embed((pr*1024+ids)%s.bv))

class SmearGate(nn.Module):
    def __init__(s,d): super().__init__(); s.gate=nn.Parameter(torch.zeros(d))
    def forward(s,x):
        g=torch.sigmoid(s.gate.to(x.dtype))[None,None,:]
        return x*(1-g)+torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],dim=1)*g

class CausalSelfAttention(nn.Module):
    def __init__(s,dim,nh,nkv,rope_base,qk_gain,rd,use_xsa=False):
        super().__init__(); s.nh=nh; s.nkv=nkv; s.hd=dim//nh; s.rd=rd; s.use_xsa=use_xsa
        kvd=nkv*s.hd
        s.c_q=CastedLinear(dim,dim,bias=False); s.c_k=CastedLinear(dim,kvd,bias=False)
        s.c_v=CastedLinear(dim,kvd,bias=False); s.proj=CastedLinear(dim,dim,bias=False)
        nn.init.zeros_(s.proj.weight)
        s.q_gain=nn.Parameter(torch.full((nh,),qk_gain,dtype=torch.float32))
        s.rotary=Rotary(rd,base=rope_base)
    def forward(s,x,ve_embed=None):
        B,T,D=x.shape
        q=s.c_q(x).reshape(B,T,s.nh,s.hd).transpose(1,2)
        k=s.c_k(x).reshape(B,T,s.nkv,s.hd).transpose(1,2)
        v=s.c_v(x).reshape(B,T,s.nkv,s.hd).transpose(1,2)
        q,k=F.rms_norm(q,(s.hd,)),F.rms_norm(k,(s.hd,))
        cos,sin=s.rotary(T,x.device,q.dtype)
        q=apply_rope_partial(q,cos,sin,s.rd); k=apply_rope_partial(k,cos,sin,s.rd)
        q=q*s.q_gain.to(q.dtype)[None,:,None,None]
        if ve_embed is not None: v=v+ve_embed.reshape(B,T,s.nkv,s.hd).transpose(1,2)
        if s.use_xsa:
            if s.nkv!=s.nh:
                k=k.repeat_interleave(s.nh//s.nkv,dim=1); v=v.repeat_interleave(s.nh//s.nkv,dim=1)
            mask=torch.triu(torch.full((T,T),float("-inf"),device=x.device,dtype=q.dtype),diagonal=1)
            mask=mask+torch.diag(torch.full((T,),float("-inf"),device=x.device,dtype=q.dtype))
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=False)
        else:
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=(s.nkv!=s.nh))
        return s.proj(y.transpose(1,2).contiguous().reshape(B,T,D))

class MLP(nn.Module):
    def __init__(s,d,m):
        super().__init__(); s.fc=CastedLinear(d,d*m,bias=False); s.proj=CastedLinear(d*m,d,bias=False)
        nn.init.zeros_(s.proj.weight)
    def forward(s,x): return s.proj(F.leaky_relu(s.fc(x),negative_slope=0.5).square())

class Block(nn.Module):
    def __init__(s,dim,nh,nkv,mult,rope_base,qk_gain,li,rd,use_xsa=False):
        super().__init__(); s.li=li
        s.attn=CausalSelfAttention(dim,nh,nkv,rope_base,qk_gain,rd,use_xsa=use_xsa)
        s.mlp=MLP(dim,mult)
        s.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32))
        s.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32))
        s.resid_mix=nn.Parameter(torch.stack([torch.ones(dim),torch.zeros(dim)]).float())
    def forward(s,x,x0,ve_embed=None):
        mix=s.resid_mix.to(x.dtype); x=mix[0][None,None,:]*x+mix[1][None,None,:]*x0
        ln_s=1.0/math.sqrt(s.li+1) if H.ln_scale else 1.0
        ao=s.attn(F.rms_norm(x,(x.size(-1),)),ve_embed=ve_embed)
        x=x+s.attn_scale.to(x.dtype)[None,None,:]*ao*ln_s
        x=x+s.mlp_scale.to(x.dtype)[None,None,:]*s.mlp(F.rms_norm(x,(x.size(-1),)))*ln_s
        return x

class GPT(nn.Module):
    def __init__(s):
        super().__init__(); a=H; dim,L=a.model_dim,a.num_layers
        s.tok_emb=nn.Embedding(a.vocab_size,dim); nn.init.normal_(s.tok_emb.weight,std=0.005)
        s.bigram=BigramHash(a.vocab_size,a.bigram_vocab,a.bigram_dim,dim) if a.bigram_vocab>0 else None
        s.smear=SmearGate(dim) if a.smear_enabled else None
        s.num_enc=L//2; s.num_dec=L-s.num_enc; s.num_skips=min(s.num_enc,s.num_dec)
        s.skip_weights=nn.Parameter(torch.ones(s.num_skips,dim,dtype=torch.float32))
        xs=L-a.xsa_last_n
        s.blocks=nn.ModuleList([Block(dim,a.num_heads,a.num_kv_heads,a.mlp_mult,a.rope_base,a.qk_gain_init,i,a.rope_dims,use_xsa=(i>=xs)) for i in range(L)])
        s.logit_softcap=a.logit_softcap
        s.ve_layers=a.ve_layer_set() if a.ve_enabled else set()
        if s.ve_layers:
            s.ve_embed=nn.Embedding(a.vocab_size,a.ve_dim)
            s.ve_projs=nn.ParameterDict({str(li):nn.Parameter(torch.empty(a.num_kv_heads*(dim//a.num_heads),a.ve_dim)) for li in s.ve_layers})
            for p in s.ve_projs.values(): nn.init.orthogonal_(p,gain=a.ve_dim**-0.5)
    def forward(s,ids,targets=None):
        B,T=ids.shape; x=s.tok_emb(ids)
        if s.bigram is not None: x=x+s.bigram(ids)
        x=F.rms_norm(x,(x.size(-1),))
        if s.smear is not None: x=s.smear(x)
        x0=x; skips=[]
        for i in range(s.num_enc):
            ve_e=F.linear(s.ve_embed(ids),s.ve_projs[str(i)]) if i in s.ve_layers else None
            x=s.blocks[i](x,x0,ve_embed=ve_e); skips.append(x)
        for i in range(s.num_dec):
            li=s.num_enc+i
            if skips: x=x+s.skip_weights[i].to(x.dtype)[None,None,:]*skips.pop()
            ve_e=F.linear(s.ve_embed(ids),s.ve_projs[str(li)]) if li in s.ve_layers else None
            x=s.blocks[li](x,x0,ve_embed=ve_e)
        x=F.rms_norm(x,(x.size(-1),)).reshape(-1,x.size(-1))
        logits=s.logit_softcap*torch.tanh(F.linear(x,s.tok_emb.weight)/s.logit_softcap)
        if targets is not None: return F.cross_entropy(logits.float(),targets.reshape(-1),reduction="mean")
        return logits

# ── Evaluation ────────────────────────────────────────────────────────────────
def eval_val(mdl,rk,ws,dev,acc,vt,bb,hs,ib):
    sl=H.train_seq_len; ls_=max(H.val_batch_size//(ws*acc*sl),1)
    ts=(vt.numel()-1)//sl; s0,s1=(ts*rk)//ws,(ts*(rk+1))//ws
    ls=torch.zeros((),device=dev,dtype=torch.float64); tc=torch.zeros((),device=dev,dtype=torch.float64); bc=torch.zeros((),device=dev,dtype=torch.float64)
    mdl.eval()
    with torch.inference_mode():
        for bs in range(s0,s1,ls_):
            be=min(bs+ls_,s1); lo=vt[bs*sl:be*sl+1].to(device=dev,dtype=torch.int64,non_blocking=True)
            x,y=lo[:-1].reshape(-1,sl),lo[1:].reshape(-1,sl)
            with torch.autocast("cuda",torch.bfloat16): bl=mdl(x,y).detach()
            n=float(y.numel()); ls+=bl.to(torch.float64)*n; tc+=n
            tb=bb[y.reshape(-1)].to(torch.int16); tb+=(hs[y.reshape(-1)]&~ib[x.reshape(-1)]).to(torch.int16); bc+=tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [ls,tc,bc]: dist.all_reduce(t,op=dist.ReduceOp.SUM)
    vl=ls/tc; mdl.train(); return float(vl.item()),float(vl.item()/math.log(2.0)*tc.item()/bc.item())

def eval_sliding(mdl,rk,ws,dev,vt,bb,hs,ib,stride=64):
    sl=H.train_seq_len; tot=vt.numel()-1; starts=list(range(0,tot-sl+1,stride)); my=starts[rk::ws]
    ls=torch.zeros((),device=dev,dtype=torch.float64); tc=torch.zeros((),device=dev,dtype=torch.float64); bc=torch.zeros((),device=dev,dtype=torch.float64)
    mdl.eval()
    with torch.inference_mode():
        for bi in range(0,len(my),16):
            ba=my[bi:bi+16]; xs=torch.stack([vt[p:p+sl] for p in ba]).to(dev,dtype=torch.int64)
            ys=torch.stack([vt[p+1:p+sl+1] for p in ba]).to(dev,dtype=torch.int64)
            with torch.autocast("cuda",torch.bfloat16): lo=mdl(xs)
            lo=lo.view(len(ba),sl,-1)
            ptl=F.cross_entropy(lo.float().reshape(-1,lo.size(-1)),ys.reshape(-1),reduction="none").reshape(len(ba),sl)
            for i,pos in enumerate(ba):
                cs=max(0,sl-stride) if pos>0 else 0
                ls+=ptl[i,cs:].double().sum(); tc+=ptl[i,cs:].numel()
                tb=bb[ys[i,cs:]].to(torch.int16); tb+=(hs[ys[i,cs:]]&~ib[xs[i,cs:]]).to(torch.int16); bc+=tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [ls,tc,bc]: dist.all_reduce(t,op=dist.ReduceOp.SUM)
    vl=ls/tc; mdl.train(); return float(vl.item()),float(vl.item()/math.log(2.0)*tc.item()/bc.item())

# ── Legal Score-First TTT (SGD, matching SOTA) ────────────────────────────────
def execute_ttt(base_mdl,rk,ws,dev,vt,bb,hs,ib):
    a=H; cs,stride,sl=a.ttt_chunk_tokens,a.eval_stride,a.train_seq_len
    mdl=GPT().to(dev).bfloat16()
    raw_sd=base_mdl.state_dict()
    clean={k.replace("_orig_mod.",""):v for k,v in raw_sd.items()}
    mdl.load_state_dict(clean,strict=False); mdl.requires_grad_(True)
    ttt_ddp=DDP(mdl,device_ids=[dev.index])
    opt=torch.optim.SGD(mdl.parameters(),lr=a.ttt_lr,momentum=a.ttt_momentum)
    ls_t=torch.zeros((),device=dev,dtype=torch.float64); tc_t=torch.zeros((),device=dev,dtype=torch.float64); bc_t=torch.zeros((),device=dev,dtype=torch.float64)
    total_len=vt.numel()-1; chunk_starts=list(range(0,total_len,cs)); total_chunks=len(chunk_starts)
    for ci,start in enumerate(chunk_starts):
        end=min(start+cs,total_len); chunk=vt[start:end+1].to(dev,dtype=torch.int64)
        mdl.eval(); starts_list=list(range(0,chunk.numel()-sl,stride)); my_starts=starts_list[rk::ws]
        with torch.no_grad():
            for bi in range(0,len(my_starts),16):
                ba=my_starts[bi:bi+16]; xs=torch.stack([chunk[p:p+sl] for p in ba]); ys=torch.stack([chunk[p+1:p+sl+1] for p in ba])
                with torch.autocast("cuda",torch.bfloat16): lo=mdl(xs)
                lo=lo.view(len(ba),sl,-1)
                ptl=F.cross_entropy(lo.float().reshape(-1,H.vocab_size),ys.reshape(-1),reduction="none").reshape(len(ba),sl)
                for i,pos in enumerate(ba):
                    c0=0 if pos==0 and ci==0 else sl-stride
                    ls_t+=ptl[i,c0:].double().sum(); tc_t+=ptl[i,c0:].numel()
                    tb=bb[ys[i,c0:]].to(torch.int16); tb+=(hs[ys[i,c0:]]&~ib[xs[i,c0:]]).to(torch.int16); bc_t+=tb.to(torch.float64).sum()
        if ci==total_chunks-1: break
        mdl.train(); seq_starts=list(range(0,chunk.numel()-sl,sl))
        seq_starts=seq_starts[:(len(seq_starts)//ws)*ws]; my_seqs=seq_starts[rk::ws]
        for epoch in range(a.ttt_epochs):
            progress=(ci*a.ttt_epochs+epoch)/(total_chunks*a.ttt_epochs)
            cos_lr=0.5*(1.0+math.cos(math.pi*progress))
            for g in opt.param_groups: g["lr"]=a.ttt_lr*cos_lr
            for bi in range(0,len(my_seqs),a.ttt_batch_seqs):
                ba=my_seqs[bi:bi+a.ttt_batch_seqs]
                if not ba: continue
                xs=torch.stack([chunk[p:p+sl] for p in ba]); ys=torch.stack([chunk[p+1:p+sl+1] for p in ba])
                with torch.autocast("cuda",torch.bfloat16): loss=ttt_ddp(xs,ys)
                loss.backward(); torch.nn.utils.clip_grad_norm_(ttt_ddp.parameters(),a.ttt_grad_clip)
                opt.step(); opt.zero_grad(set_to_none=True)
    if dist.is_available() and dist.is_initialized():
        for t in [ls_t,tc_t,bc_t]: dist.all_reduce(t,op=dist.ReduceOp.SUM)
    vl=ls_t/tc_t; return float(vl.item()/math.log(2.0)*tc_t.item()/bc_t.item())

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    a=H; code=Path(__file__).read_text(encoding="utf-8")
    dd="RANK" in os.environ and "WORLD_SIZE" in os.environ
    rk=int(os.environ.get("RANK","0")); ws=int(os.environ.get("WORLD_SIZE","1"))
    lr_=int(os.environ.get("LOCAL_RANK","0")); acc=8//ws
    dev=torch.device("cuda",lr_); torch.cuda.set_device(dev)
    if dd: dist.init_process_group(backend="nccl",device_id=dev); dist.barrier()
    master=rk==0
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(True)
    logfile=None
    if master: os.makedirs("logs",exist_ok=True); logfile=f"logs/{a.run_id}.txt"
    def log(m,c=True):
        if not master: return
        if c: print(m)
        if logfile:
            with open(logfile,"a") as f: print(m,file=f)
    log(code,c=False); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    sp=spm.SentencePieceProcessor(model_file=a.tokenizer_path)
    vt=load_val_tokens(a.val_files,a.train_seq_len); bb,hs,ib=build_sp_luts(sp,a.vocab_size,dev)
    log(f"val tokens: {vt.numel()-1}")
    bm=GPT().to(dev).bfloat16()
    for m in bm.modules():
        if isinstance(m,CastedLinear): m.float()
    with torch.no_grad():
        for n,p in bm.named_parameters():
            if (p.ndim<2 or any(pt in n for pt in CONTROL_PATTERNS)) and p.dtype!=torch.float32: p.data=p.data.float()
    # EMA: deepcopy BEFORE compile, use named_parameters for safe matching
    ema=copy.deepcopy(bm); ema.requires_grad_(False)
    ema_names={n:p for n,p in ema.named_parameters()}
    compiled=torch.compile(bm,dynamic=False); model=DDP(compiled,device_ids=[lr_],broadcast_buffers=False) if dd else compiled
    # SWA (tight: warmdown only)
    swa_started=False; swa_sd=None; swa_count=0
    # Optimizers with weight decay (matching SOTA)
    bp=list(bm.blocks.named_parameters())
    mp=[p for n,p in bp if p.ndim==2 and not any(pt in n for pt in CONTROL_PATTERNS)]
    sp_=[p for n,p in bp if p.ndim<2 or any(pt in n for pt in CONTROL_PATTERNS)]
    if bm.skip_weights.numel()>0: sp_.append(bm.skip_weights)
    assigned={id(p) for p in sp_}|{id(p) for p in mp}|{id(bm.tok_emb.weight)}
    for n,p in bm.named_parameters():
        if any(x in n for x in ["bigram","smear","ve_"]):
            if id(p) not in assigned: sp_.append(p); assigned.add(id(p))
    ot=torch.optim.AdamW([{"params":[bm.tok_emb.weight],"lr":a.tied_embed_lr,"base_lr":a.tied_embed_lr}],betas=(0.9,0.95),weight_decay=a.adam_wd,fused=True)
    om=Muon(mp,lr=a.matrix_lr,momentum=a.muon_momentum,backend_steps=a.muon_backend_steps,wd=a.muon_wd)
    for g in om.param_groups: g["base_lr"]=a.matrix_lr
    os_=torch.optim.AdamW([{"params":sp_,"lr":a.scalar_lr,"base_lr":a.scalar_lr}],betas=(0.9,0.95),weight_decay=a.adam_wd,fused=True)
    opts=[ot,om,os_]
    np_=sum(p.numel() for p in bm.parameters())
    log(f"params: {np_} | layers: {a.num_layers} | dim: {a.model_dim}")
    log(f"ws: {ws} accum: {acc} | batch_tokens: {a.train_batch_tokens}")
    tl_=DistTokenLoader(a.train_files,rk,ws,dev)
    mwm=1000.0*a.max_wallclock_seconds if a.max_wallclock_seconds>0 else None; gs=1.0/acc
    def lr_mul(st,el):
        if a.warmdown_iters<=0: return 1.0
        if mwm is None:
            wd=max(a.iterations-a.warmdown_iters,0)
            return max((a.iterations-st)/max(a.warmdown_iters,1),0.0) if st>=wd else 1.0
        sm=el/max(st,1); wm=a.warmdown_iters*sm; rm=max(mwm-el,0.0)
        return rm/max(wm,1e-9) if rm<=wm else 1.0
    # Compile warmup
    if a.warmup_steps>0:
        isd={k:v.detach().cpu().clone() for k,v in bm.state_dict().items()}
        ios=[copy.deepcopy(o.state_dict()) for o in opts]; model.train()
        for wi in range(a.warmup_steps):
            for o in opts: o.zero_grad(set_to_none=True)
            for ms in range(acc):
                if dd: model.require_backward_grad_sync=ms==acc-1
                x,y=tl_.next_batch(a.train_batch_tokens,a.train_seq_len,acc)
                with torch.autocast("cuda",torch.bfloat16): wl=model(x,y)
                (wl*gs).backward()
            for o in opts: o.step(); o.zero_grad(set_to_none=True)
        bm.load_state_dict(isd,strict=True)
        for o,s in zip(opts,ios): o.load_state_dict(s)
        for o in opts: o.zero_grad(set_to_none=True)
        if dd: model.require_backward_grad_sync=True
        tl_=DistTokenLoader(a.train_files,rk,ws,dev)
        ema.load_state_dict(isd,strict=True); ema_names={n:p for n,p in ema.named_parameters()}
        log(f"warmup done ({a.warmup_steps} steps)")
    # Training loop
    trms=0.0; stop=None; torch.cuda.synchronize(); t0=time.perf_counter(); step=0
    while True:
        last=step==a.iterations or (stop is not None and step>=stop)
        if last or (a.val_loss_every>0 and step%a.val_loss_every==0 and step>0):
            torch.cuda.synchronize(); trms+=1000.0*(time.perf_counter()-t0)
            vl,vb=eval_val(model,rk,ws,dev,acc,vt,bb,hs,ib)
            log(f"step:{step}/{a.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} time:{trms:.0f}ms avg:{trms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0=time.perf_counter()
        if last:
            if stop is not None and step<a.iterations: log(f"early stop at {step}")
            break
        el=trms+1000.0*(time.perf_counter()-t0); sc=lr_mul(step,el)
        frac=min(step/a.muon_warmup_steps,1.0) if a.muon_warmup_steps>0 else 1.0
        cm=(1-frac)*a.muon_warmup_start+frac*a.muon_momentum
        for g in om.param_groups: g["momentum"]=cm
        for o in opts:
            for g in o.param_groups: g["lr"]=g["base_lr"]*sc
        for o in opts: o.zero_grad(set_to_none=True)
        tl=torch.zeros((),device=dev)
        for ms in range(acc):
            if dd: model.require_backward_grad_sync=ms==acc-1
            x,y=tl_.next_batch(a.train_batch_tokens,a.train_seq_len,acc)
            with torch.autocast("cuda",torch.bfloat16): loss=model(x,y)
            tl+=loss.detach(); (loss*gs).backward()
        tl/=acc
        for o in opts: o.step(); o.zero_grad(set_to_none=True)
        # EMA update (name-matched, safe across compile)
        if a.ema_enabled:
            with torch.no_grad():
                for n,pb in bm.named_parameters():
                    cn=n.replace("_orig_mod.","")
                    if cn in ema_names: ema_names[cn].mul_(a.ema_decay).add_(pb,alpha=1.0-a.ema_decay)
        # Tight SWA
        if a.swa_enabled and sc<1.0:
            if not swa_started:
                swa_sd={k:v.clone().double() for k,v in bm.state_dict().items()}; swa_count=0; swa_started=True
            if step%a.swa_every==0:
                with torch.no_grad():
                    swa_count+=1
                    for k,v in bm.state_dict().items(): swa_sd[k]+=v.double()
        # Late QAT (wallclock-based)
        approx_now=trms+1000.0*(time.perf_counter()-t0)
        qat_on=a.late_qat and mwm is not None and (mwm-approx_now)<mwm*a.late_qat_frac
        if not qat_on and a.late_qat: qat_on=step>a.iterations*(1.0-a.late_qat_frac)
        if qat_on:
            with torch.no_grad():
                for n,p in bm.named_parameters():
                    if p.ndim>=2 and not any(pt in n for pt in CONTROL_PATTERNS): p.data=_fake_quant_int6(p.data)
        step+=1; approx=trms+1000.0*(time.perf_counter()-t0)
        if a.train_log_every>0 and (step<=10 or step%a.train_log_every==0):
            log(f"step:{step}/{a.iterations} loss:{tl.item():.4f} time:{approx:.0f}ms avg:{approx/step:.2f}ms")
        reached=mwm is not None and approx>=mwm
        if dd and mwm is not None:
            rt=torch.tensor(int(reached),device=dev); dist.all_reduce(rt,op=dist.ReduceOp.MAX); reached=bool(rt.item())
        if stop is None and reached: stop=step
    log(f"peak mem: {torch.cuda.max_memory_allocated()//1024//1024} MiB")
    # Pick best model
    eval_model=ema if a.ema_enabled else bm
    if a.swa_enabled and swa_started and swa_count>0:
        swa_f={}
        with torch.no_grad():
            for k in swa_sd: swa_f[k]=(swa_sd[k]/(swa_count+1)).to(bm.state_dict()[k].dtype)
        swa_m=copy.deepcopy(bm); swa_m.load_state_dict(swa_f,strict=True)
        # Compare SWA vs EMA
        bm.load_state_dict(swa_f,strict=True)
        swa_vl,swa_vb=eval_val(model,rk,ws,dev,acc,vt,bb,hs,ib)
        log(f"SWA val_loss:{swa_vl:.4f} val_bpb:{swa_vb:.4f} (count={swa_count})")
        ema_sd={k.replace("_orig_mod.",""):v for k,v in ema.state_dict().items()}
        bm.load_state_dict(ema_sd,strict=True)
        ema_vl,ema_vb=eval_val(model,rk,ws,dev,acc,vt,bb,hs,ib)
        log(f"EMA val_loss:{ema_vl:.4f} val_bpb:{ema_vb:.4f}")
        if swa_vb<ema_vb: log("Using SWA model (better)"); eval_model=swa_m
        else: log("Using EMA model (better)")
    # Serialize
    log("Serializing model...")
    sd={k.replace("_orig_mod.",""):v for k,v in eval_model.state_dict().items()}
    meta,raw=quantize_state_dict(sd); compressed=compress_bytes(raw)
    payload=pickle.dumps({"m":meta,"c":compressed})
    if master:
        Path("final_model.ptz").write_bytes(payload); cb=len(code.encode("utf-8"))
        log(f"Model: {len(payload)} bytes | Code: {cb} bytes | Total: {len(payload)+cb} bytes")
        log(f"Target < 16,000,000 bytes: {'PASS' if len(payload)+cb<16_000_000 else 'FAIL'}")
    # Round-trip
    log("Round-trip validation..."); loaded=pickle.loads(payload)
    rt_sd=dequantize_state_dict(loaded["m"],decompress_bytes(loaded["c"]))
    bm.load_state_dict(rt_sd,strict=True); torch.cuda.synchronize(); te=time.perf_counter()
    qvl,qvb=eval_val(model,rk,ws,dev,acc,vt,bb,hs,ib); torch.cuda.synchronize()
    log(f"int6+{COMPRESSOR} roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} eval:{1000*(time.perf_counter()-te):.0f}ms")
    log("Sliding window eval..."); torch.cuda.synchronize(); ts=time.perf_counter()
    swvl,swvb=eval_sliding(model,rk,ws,dev,vt,bb,hs,ib,stride=a.eval_stride); torch.cuda.synchronize()
    log(f"sliding_window val_loss:{swvl:.4f} val_bpb:{swvb:.4f} eval:{1000*(time.perf_counter()-ts):.0f}ms")
    if a.ttt_enabled:
        log("Starting Legal Score-First TTT...")
        eval_model.load_state_dict(rt_sd,strict=True); torch.cuda.synchronize(); tt=time.perf_counter()
        ttt_bpb=execute_ttt(eval_model,rk,ws,dev,vt,bb,hs,ib); torch.cuda.synchronize()
        log(f"TTT val_bpb:{ttt_bpb:.4f} time:{1000*(time.perf_counter()-tt):.0f}ms")
        log(f"FINAL val_bpb:{ttt_bpb:.4f}")
    else: log(f"FINAL val_bpb:{swvb:.4f}")
    if dd: dist.barrier(); dist.destroy_process_group()

if __name__=="__main__": main()
