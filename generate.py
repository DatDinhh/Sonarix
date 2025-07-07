CACHE_NPY   = "Your/Files/Pathway"
MODEL_DIR   = "Your/Files/Pathway"     
OUT_DIR     = "Your/Files/Pathway"
TEMPO_MULT  = 3.5      # 2.0 = half‑speed, 1.5 = ⅔ speed, 0.5 = double speed


# Optional Knobs
PPQ               = 960
MAX_SHIFT_BEATS   = 8           
NOTE_VELOCITY     = 80
FALLBACK_NOTE_LEN = 0.5        
SEED_TOKEN        = "NOTE_ON_64"
SAMPLE_LEN        = 1500
TOP_K             = 8
TEMPERATURE       = 1.0
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# IMPORTS 
import os, math, numpy as np, torch, torch.nn as nn, pathlib, pretty_midi, warnings
print("Using device:", DEVICE)
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 1.  VOCAB 
vocab = sorted(set(np.load(CACHE_NPY, allow_pickle=True)))
t2i   = {s:i for i,s in enumerate(vocab)}
i2t   = {i:s for s,i in t2i.items()}
V     = len(vocab)
print("✓ vocab:", V, "tokens")

# 2.  MODEL  
def rotate_half(x): h=x.shape[-1]//2; return torch.cat([-x[...,h:],x[...,:h]],-1)
def rope(q,k,s,c): return (q*c)+(rotate_half(q)*s), (k*c)+(rotate_half(k)*s)
def sincos(pos,dim,b=10000):
    inv=1/b**(torch.arange(0,dim,2,device=pos.device)/dim)
    ang=pos*inv; sin=torch.sin(ang).repeat_interleave(2,-1)
    cos=torch.cos(ang).repeat_interleave(2,-1); return sin,cos
class RMSNorm(nn.Module):
    def __init__(s,d,eps=1e-6): super().__init__(); s.w=nn.Parameter(torch.ones(d)); s.eps=eps
    def forward(s,x): return s.w*x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+s.eps)
class SwiGLU(nn.Module):
    def __init__(s,d,m=4,p=0.1): super().__init__(); s.lin=nn.Linear(d,d*m*2); s.out=nn.Linear(d*m,d); s.drop=nn.Dropout(p)
    def forward(s,x): u,v=s.lin(x).chunk(2,-1); return s.out(s.drop(u*torch.sigmoid(v)))
class Attn(nn.Module):
    def __init__(s,d,h,p=0.1):
        super().__init__(); s.h,s.dh=h,d//h
        s.q=nn.Linear(d,d); s.k=nn.Linear(d,d); s.v=nn.Linear(d,d); s.o=nn.Linear(d,d); s.drop=nn.Dropout(p)
    def forward(s,x,m):
        b,t,_=x.shape
        q=s.q(x).view(b,t,s.h,s.dh).transpose(1,2)
        k=s.k(x).view(b,t,s.h,s.dh).transpose(1,2)
        v=s.v(x).view(b,t,s.h,s.dh).transpose(1,2)
        sin,cos=sincos(torch.arange(t,device=x.device).unsqueeze(-1),s.dh)
        q,k=rope(q,k,sin,cos)
        a=(q@k.transpose(-2,-1))/math.sqrt(s.dh)
        if m is not None: a=a.masked_fill(m,-1e4)
        a=s.drop(torch.softmax(a,-1)); out=(a@v).transpose(1,2).reshape(b,t,-1)
        return s.o(s.drop(out))
class Block(nn.Module):
    def __init__(s,d=512,h=8,p=0.1): super().__init__(); s.n1=RMSNorm(d); s.att=Attn(d,h,p); s.n2=RMSNorm(d); s.ff=SwiGLU(d,4,p)
    def forward(s,x,m): x=x+s.att(s.n1(x),m); return x+s.ff(s.n2(x))
class GPT(nn.Module):
    def __init__(s,V,L=512,d=512,n=6,h=8,p=0.1):
        super().__init__(); s.tok=nn.Embedding(V,d)
        s.blocks=nn.ModuleList(Block(d,h,p) for _ in range(n))
        s.norm=RMSNorm(d); s.head=nn.Linear(d,V,bias=False)
        s.register_buffer("mask", torch.triu(torch.ones(L,L),1).bool())
    def forward(s,x):
        b,t=x.shape; h=s.tok(x); m=s.mask[:t,:t]
        for blk in s.blocks: h=blk(h,m)
        return s.head(s.norm(h))

model=GPT(V).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR,"best.pt"),map_location=DEVICE))
model.eval(); print(" model loaded")

# 3.  SAMPLER 
@torch.inference_mode()
def sample(seed_id,n=SAMPLE_LEN,T=TEMPERATURE,k=TOP_K):
    seq=torch.tensor([[seed_id]],device=DEVICE)
    for _ in range(n):
        logits=model(seq[:,-512:])[:,-1,:]/T
        topv,topi=torch.topk(torch.softmax(logits,-1).squeeze(0),k)
        nxt=topi[torch.multinomial(topv/topv.sum(),1)]
        seq=torch.cat([seq,nxt.view(1,1)],1)
    return seq.squeeze(0).cpu().tolist()

# 4.  DECODER 
def to_midi(ids,out):
    pm   = pretty_midi.PrettyMIDI(resolution=PPQ)
    inst = pretty_midi.Instrument(0); pm.instruments.append(inst)

    bpm           = 120        
    sec_per_tick  = 60.0 / (bpm * PPQ) * TEMPO_MULT
    cur_sec       = 0.0
    active_notes  = {}          

    def finish(p,end):
        if p in active_notes and end>active_notes[p]:
            inst.notes.append(pretty_midi.Note(NOTE_VELOCITY,p,active_notes[p],end))
        active_notes.pop(p,None)

    for tid in ids:
        tok = i2t[tid]

        if tok.startswith("TIME_SHIFT_"):
            ticks = int(tok.split("_")[-1])
            ticks = min(ticks, int(MAX_SHIFT_BEATS * PPQ))  
            cur_sec += ticks * sec_per_tick

        elif tok.startswith("NOTE_ON_"):
            pitch = int(tok.split("_")[-1]) & 0x7F
            finish(pitch, cur_sec)         
            active_notes[pitch] = cur_sec

        elif tok.startswith("NOTE_OFF_"):
            pitch = int(tok.split("_")[-1]) & 0x7F
            finish(pitch, cur_sec)

        elif tok.startswith("TEMPO_") and tok.endswith("_BPM"):
            bpm = max(40, min(240, int(tok.split("_")[1])))
            sec_per_tick = 60.0 / (bpm * PPQ) * TEMPO_MULT

    for p,s in active_notes.items():
        inst.notes.append(pretty_midi.Note(NOTE_VELOCITY,p,s,s+FALLBACK_NOTE_LEN*TEMPO_MULT))

    pm.write(out)
    print(f" MIDI written → {out}\n   tempo×{TEMPO_MULT}, max_rest≤{MAX_SHIFT_BEATS} beats")

# 5.  RUN 
seed_id = t2i.get(SEED_TOKEN,0)
ids     = sample(seed_id)
to_midi(ids, f"{OUT_DIR}/generated_{TEMPO_MULT}x.mid")
