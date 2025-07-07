# Path configuration
CSV_DIR   = "Your/Files/Pathway"
MODEL_DIR = "Your/Files/Pathway"
OUT_DIR   = "Your/Files/Pathway"

SEQ_LEN   = 512
BATCH_SZ  = 4
EPOCHS    = 15
LR0       = 1e-4
LR_MIN    = 2e-5
PATIENCE  = 3

# Imports
import os, math, glob, pathlib, tqdm, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pretty_midi, mido

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

for p in (MODEL_DIR, OUT_DIR):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# 1.  Tokeniser & Dataset
CACHE_NPY = "Your/Files/Pathway"

def row_to_tokens(r):
    toks = []
    dt = int(r["delta_ticks"])
    if dt: toks.append(f"TIME_SHIFT_{dt}")

    typ = r["type"]
    if typ == "note_on":
        toks.append(f"NOTE_ON_{int(r['note'])}")
    elif typ == "note_off":
        toks.append(f"NOTE_OFF_{int(r['note'])}")
    elif typ == "set_tempo":
        bpm = round(60_000_000 / int(r["tempo"]))
        toks.append(f"TEMPO_{bpm}_BPM")
    elif typ == "time_signature":
        toks.append(f"TIME_SIGNATURE_{int(r['numerator'])}_{int(r['denominator'])}")
    elif typ == "key_signature":
        toks.append(f"KEY_SIGNATURE_{r['key']}")
    elif typ == "program_change":
        toks.append(f"PROGRAM_{int(r['program'])}")
    elif typ == "control_change":
        toks.append(f"CTRL_{int(r['control'])}_{int(r['value'])}")
    elif typ in ("track_name", "marker", "text"):
        txt = str(r["text"])[:20].replace(" ","_").replace(",","")
        toks.append(f"{typ.upper()}_{txt}")
    elif typ == "smpte_offset":
        hh,mm,ss,fr = map(int,[r["hours"],r["minutes"],r["seconds"],r["frames"]])
        toks.append(f"SMPTE_{hh:02d}_{mm:02d}_{ss:02d}_{fr:02d}")
    else:
        toks.append(typ.upper())
    return toks

def load_tokens(csv_dir):
    if os.path.exists(CACHE_NPY):
        print("[Data] cached tokens found.")
        return np.load(CACHE_NPY, allow_pickle=True)

    stream=[]
    for p in tqdm.tqdm(sorted(glob.glob(f"{csv_dir}/*.csv")), unit="csv"):
        df=pd.read_csv(p)
        for _,r in df.iterrows():
            stream.extend(row_to_tokens(r))
    arr=np.array(stream,dtype=object)
    np.save(CACHE_NPY, arr); print("[Data] token cache saved.")
    return arr

class CSVTokenDataset(Dataset):
    def __init__(self, csv_dir, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.tokens = load_tokens(csv_dir)
        uniq = sorted(set(self.tokens))
        self.t2i = {s:i for i,s in enumerate(uniq)}
        self.i2t = {i:s for s,i in self.t2i.items()}
        self.ids  = np.vectorize(self.t2i.get)(self.tokens)
        self.slices = len(self.ids)-seq_len
        print(f"tokens={len(self.ids)}  vocab={len(uniq)}")

    def __len__(self): return self.slices
    def __getitem__(self, idx):
        x=self.ids[idx:idx+SEQ_LEN]
        y=self.ids[idx+1:idx+SEQ_LEN+1]
        return torch.as_tensor(x), torch.as_tensor(y)

# 2.  Transformer (RoPE ‑ SwiGLU)
def rotate_half(x): h=x.shape[-1]//2; return torch.cat([-x[...,h:],x[...,:h]],-1)
def rope(q,k,sin,cos): return (q*cos)+(rotate_half(q)*sin), (k*cos)+(rotate_half(k)*sin)
def sincos(pos,dim,base=10000):
    inv=1/base**(torch.arange(0,dim,2,device=pos.device)/dim)
    ang=pos*inv
    sin=torch.sin(ang).repeat_interleave(2,-1)
    cos=torch.cos(ang).repeat_interleave(2,-1)
    return sin,cos

class RMSNorm(nn.Module):
    def __init__(self,d,eps=1e-6):
        super().__init__(); self.w=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x): return self.w*x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

class SwiGLU(nn.Module):
    def __init__(self,d,m=4,p=0.0):
        super().__init__(); self.lin=nn.Linear(d,d*m*2); self.out=nn.Linear(d*m,d); self.drop=nn.Dropout(p)
    def forward(self,x):
        u,v=self.lin(x).chunk(2,-1)
        return self.out(self.drop(u*torch.sigmoid(v)))

class Attn(nn.Module):
    def __init__(self,d,h,p=0.0):
        super().__init__(); self.h,self.dh=h,d//h
        self.q=nn.Linear(d,d); self.k=nn.Linear(d,d); self.v=nn.Linear(d,d); self.o=nn.Linear(d,d); self.drop=nn.Dropout(p)
    def forward(self,x,mask):
        b,t,_=x.shape
        q=self.q(x).view(b,t,self.h,self.dh).transpose(1,2)
        k=self.k(x).view(b,t,self.h,self.dh).transpose(1,2)
        v=self.v(x).view(b,t,self.h,self.dh).transpose(1,2)
        sin,cos=sincos(torch.arange(t,device=x.device).unsqueeze(-1),self.dh)
        q,k=rope(q,k,sin,cos)
        a=(q@k.transpose(-2,-1))/math.sqrt(self.dh)
        if mask is not None: a=a.masked_fill(mask,-1e4)
        a=self.drop(torch.softmax(a,-1))
        out=(a@v).transpose(1,2).reshape(b,t,-1)
        return self.o(self.drop(out))

class Block(nn.Module):
    def __init__(self,d=512,h=8,p=0.1):
        super().__init__(); self.n1=RMSNorm(d); self.att=Attn(d,h,p); self.n2=RMSNorm(d); self.ff=SwiGLU(d,4,p)
    def forward(self,x,m): x=x+self.att(self.n1(x),m); return x+self.ff(self.n2(x))

class GPT(nn.Module):
    def __init__(self,V,L=SEQ_LEN,d=512,n=6,h=8,p=0.1):
        super().__init__(); self.tok=nn.Embedding(V,d); self.max=L
        self.blocks=nn.ModuleList(Block(d,h,p) for _ in range(n))
        self.norm=RMSNorm(d); self.head=nn.Linear(d,V,bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(L,L),1).bool())
    def forward(self,x):
        b,t=x.shape; h=self.tok(x); m=self.mask[:t,:t]
        for blk in self.blocks: h=blk(h,m)
        return self.head(self.norm(h))

# 3.  Train utils
def train(model,tr_loader,val_loader):
    opt=torch.optim.AdamW(model.parameters(), lr=LR0)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS*len(tr_loader),LR_MIN)
    crit=nn.CrossEntropyLoss(); best=1e9; noimp=0

    for ep in range(1,EPOCHS+1):
        model.train(); tot=0
        for xb,yb in tqdm.tqdm(tr_loader,desc=f"train {ep}/{EPOCHS}",leave=False):
            xb,yb=xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); out=model(xb); b,t,v=out.shape
            loss=crit(out.view(b*t,v), yb.view(b*t)); loss.backward(); opt.step(); sched.step()
            tot+=loss.item()
        tr_loss=tot/len(tr_loader)

        model.eval(); tot=0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb=xb.to(DEVICE), yb.to(DEVICE)
                out=model(xb); b,t,v=out.shape
                tot+=crit(out.view(b*t,v), yb.view(b*t)).item()
        val_loss=tot/len(val_loader)
        print(f"Ep{ep:02d} train={tr_loss:.4f} val={val_loss:.4f}")

        if val_loss<best:
            best,noimp=val_loss,0
            torch.save(model.state_dict(), f"{MODEL_DIR}/best.pt"); print(" best saved")
        else:
            noimp+=1
            if noimp>=PATIENCE:
                print("  early‑stopping"); break

# 4.  Sampling & MIDI decode
@torch.inference_mode()
def sample(model,start,n=1500,temp=1.0,top_k=8):
    model.eval(); seq=torch.tensor([start],device=DEVICE).unsqueeze(0)
    for _ in range(n):
        logits=model(seq[:,-SEQ_LEN:])[:,-1,:]/temp
        probs=torch.softmax(logits,-1); topv,topi=torch.topk(probs,top_k)
        nxt=topi[torch.multinomial(topv/topv.sum(),1)]
        seq=torch.cat([seq,nxt.view(1,1)],1)
    return seq.squeeze(0).cpu().tolist()

def ids_to_midi(ids,i2t,outfile,tempo_x=4):
    pm=pretty_midi.PrettyMIDI(); inst=pretty_midi.Instrument(0)
    cur=0.0; sec_per_tick=0.5/480
    for i in ids:
        tok=i2t[i]
        if tok.startswith("NOTE_ON_"):
            n=int(tok.split("_")[-1]); inst.notes.append(pretty_midi.Note(80,cur,cur+0.5,n))
        elif tok.startswith("TIME_SHIFT_"):
            cur+=int(tok.split("_")[-1])*sec_per_tick*tempo_x
    pm.instruments.append(inst); pm.write(outfile); print("→",outfile)

# 5.  Main
def main():
    ds=CSVTokenDataset(CSV_DIR)
    val_len=int(0.1*len(ds)); tr_len=len(ds)-val_len
    tr_ds,val_ds=random_split(ds,[tr_len,val_len],generator=torch.Generator().manual_seed(42))
    tr_ld=DataLoader(tr_ds,BATCH_SZ,shuffle=True,drop_last=True,pin_memory=True)
    val_ld=DataLoader(val_ds,BATCH_SZ,shuffle=False,drop_last=True,pin_memory=True)

    model=GPT(len(ds.t2i)).to(DEVICE)
    ck=f"{MODEL_DIR}/best.pt"
    if os.path.exists(ck):
        model.load_state_dict(torch.load(ck,map_location=DEVICE)); print("✓ checkpoint loaded")

    train(model,tr_ld,val_ld)
    model.load_state_dict(torch.load(ck,map_location=DEVICE))

    seed=ds.t2i.get("NOTE_ON_64",0)
    for idx,(t,tf) in enumerate([(1.0,4),(1.3,4)],1):
        ids=sample(model,seed,1500,t); midi=f"{OUT_DIR}/gen_{idx}.mid"
        ids_to_midi(ids,ds.i2t,midi,tf)

main()
