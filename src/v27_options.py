"""Score candidate CYP2D6 arms for v27 against v26 on the blend, CYP2D6 column only.

Usage: python v27_options.py name=arm1.npz[,arm2.npz...] name2=... 
A candidate with several files uses the mean of their oof / hold / test arrays (same rows).
The first candidate is the reference; it should be run `a` alone (v26's arm).
Prints, per candidate: blend weights, dev and holdout ST-RAE, paired bootstrap of the holdout
ST-RAE change against the reference (CYP2D6 only, and as a share of macro), and the Spearman of
its blind column with the reference's. The other three isoforms are v26's and do not change."""
from __future__ import annotations
import itertools, os, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
C=Path("clean/cache"); P1=Path(os.environ.get("P1_DIR","/tmp/part1")); N5=Path("/tmp/nb5")
N7=Path(os.environ.get("N7_DIR","/tmp/nb7")); N15=Path("/tmp/nb15"); DS=np.arange(-0.6,0.81,0.05)
ISO=os.environ.get("ISO","CYP2D6"); KEYS=["tab","raw","n15","mtl13","che"]
BLIND={"CYP1A2":(4.412,1.553),"CYP2C9":(4.830,1.101),"CYP2D6":(3.420,1.599),"CYP3A4":(4.880,1.272)}[ISO]
soft=lambda p,lo,hi: np.clip(p-hi,0,None)+np.clip(lo-p,0,None)
def st_rae(y,p,lo,hi): return float(soft(p,lo,hi).sum()/soft(np.full_like(y,y.mean()),lo,hi).sum())
def cal(yd,bd,lod,hid):
    k=float(np.clip(pearsonr(yd,bd)[0],0.3,0.95)); u=(bd-bd.mean())/bd.std()
    d=float(DS[int(np.argmin([st_rae(yd,yd.mean()+q+u*k*yd.std(),lod,hid) for q in DS]))])
    return k,d
meta=pd.read_csv(C/"meta_train.csv.gz"); c=f"{ISO}_pIC50_direct_inhibition"
y=meta[c].to_numpy(float); lo=meta[f"{c}_conf_low"].to_numpy(float); hi=meta[f"{c}_conf_high"].to_numpy(float)
def load_arm(files):
    Z=[np.load(f) for f in files]
    for z in Z[1:]:
        assert np.array_equal(z["dev"],Z[0]["dev"]) and np.array_equal(z["ho"],Z[0]["ho"]), "rows differ"
    return dict(dev=Z[0]["dev"],ho=Z[0]["ho"],oof=np.mean([z["oof"] for z in Z],0),
                hold=np.mean([z["hold"] for z in Z],0),test=np.mean([z["test"] for z in Z],0))
fixed={"tab":C/f"tabqhts_{ISO}.npz","raw":P1/f"full_raw_sub_{ISO}.npz","mtl13":N5/f"mtl_{ISO}.npz","che":N7/f"mtlche_{ISO}.npz"}
REFW={}
def run(arm,force=None):
    A={k:dict(np.load(p)) for k,p in fixed.items()}; A["n15"]=arm
    dev,ho=A["tab"]["dev"],A["tab"]["ho"]
    for a in A.values(): dev=np.intersect1d(dev,a["dev"]); ho=np.intersect1d(ho,a["ho"])
    col=lambda a,k,idx: a[k][np.isin(a["dev"] if k=="oof" else a["ho"],idx)]
    O=[col(A[k],"oof",dev) for k in KEYS]; H=[col(A[k],"hold",ho) for k in KEYS]; T=[A[k]["test"] for k in KEYS]
    yd,lod,hid=y[dev],lo[dev],hi[dev]; g=np.round(np.arange(0,1.0001,0.1),2); best=None
    for w in itertools.product(g,repeat=len(KEYS)-1):
        if sum(w)>1.0001: continue
        ws=list(w)+[1-sum(w)]; bd=sum(q*o for q,o in zip(ws,O)); k_,d_=cal(yd,bd,lod,hid)
        s=st_rae(yd,yd.mean()+d_+(bd-bd.mean())/bd.std()*k_*yd.std(),lod,hid)
        if best is None or s<best[0]: best=(s,ws)
    s_dev,ws=best
    if force is not None:   # pseudo-arm weight set by hand; the other four keep the reference blend's proportions
        r=REFW['w']; o=[i for i,k in enumerate(KEYS) if k!='n15']; tot=sum(r[i] for i in o)
        ws=[force if k=='n15' else r[i]*(1-force)/tot for i,k in enumerate(KEYS)]
        bd0=sum(q*o_ for q,o_ in zip(ws,O)); k0,d0=cal(yd,bd0,lod,hid)
        s_dev=st_rae(yd,yd.mean()+d0+(bd0-bd0.mean())/bd0.std()*k0*yd.std(),lod,hid)
    else: REFW.setdefault('w',ws)
    bd=sum(q*o for q,o in zip(ws,O)); bh=sum(q*h for q,h in zip(ws,H)); bt=sum(q*t for q,t in zip(ws,T))
    k_,d_=cal(yd,bd,lod,hid); v=(bh-bh.mean())/bh.std()
    ph=y[ho].mean()+d_+v*k_*y[ho].std(); test=BLIND[0]+d_+(bt-bt.mean())*(k_*BLIND[1]/bt.std())
    return dict(ws=dict(zip(KEYS,ws)),s_dev=s_dev,y=y[ho],p=ph,lo=lo[ho],hi=hi[ho],test=test)
cands=[a.split("=",1) for a in sys.argv[1:]]; R={}
for name,files in cands:
    fw=None
    head,_,tail=files.rpartition("@")
    try: fw=float(tail); files=head
    except ValueError: pass
    R[name]=run(load_arm([Path(f) for f in files.split(",")]),fw)
ref=next(iter(R)); rng=np.random.default_rng(0); idx=[rng.integers(0,len(R[ref]["y"]),len(R[ref]["y"])) for _ in range(6000)]
print(f"{'candidate':14s}{'dev':>8s}{'holdout':>9s}{'d vs ref':>10s}{'CI95':>20s}{'P(better)':>10s}{'rho blind':>10s}  weights")
for n,r in R.items():
    h=st_rae(r["y"],r["p"],r["lo"],r["hi"]); a=R[ref]
    d=np.array([st_rae(r["y"][k],r["p"][k],r["lo"][k],r["hi"][k])-st_rae(a["y"][k],a["p"][k],a["lo"][k],a["hi"][k]) for k in idx])
    print(f"{n:14s}{r['s_dev']:8.4f}{h:9.4f}{d.mean():+10.4f}  [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]{np.mean(d<0):10.2f}"
          f"{spearmanr(r['test'],a['test']).statistic:10.3f}  "+", ".join(f"{k} {v:.1f}" for k,v in r["ws"].items() if v>0))
