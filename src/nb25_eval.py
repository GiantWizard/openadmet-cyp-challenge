"""nb25 (Atom-JEPA) against the shipped v26 blend: added as a sixth arm, and swapped in for mtl13.
Same weight search, calibration and paired bootstrap as nb24_eval.py / make_v27.py.

Usage: python nb25_eval.py <nb25 arm dir from nb25_arm.py> [nb24 CYP2D6 arm npz = v26's]
All configurations are fitted on the same development rows (the nb25 arm's), so a partial set of
folds still gives a fair comparison, only a noisier one."""
from __future__ import annotations
import itertools, os, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
C=Path("clean/cache"); P1=Path(os.environ.get("P1_DIR","/tmp/part1")); N5=Path("/tmp/nb5")
N7=Path(os.environ.get("N7_DIR","/tmp/nb7")); N15=Path("/tmp/nb15")
AJ=Path(sys.argv[1])
D6=Path(sys.argv[2] if len(sys.argv)>2 else
  "/private/tmp/claude-504/-Users-b-Documents-College-Applications/7ae4c836-0a4d-451b-8e20-262b30e8a696/scratchpad/nb24res/nb24_pseudo_CYP2D6.npz")
ISO=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; DS=np.arange(-0.6,0.81,0.05)
PATH={"tab":lambda i:C/f"tabqhts_{i}.npz","raw":lambda i:P1/f"full_raw_sub_{i}.npz",
      "n15":lambda i:D6 if i=="CYP2D6" else N15/f"nb15_{i}.npz","mtl13":lambda i:N5/f"mtl_{i}.npz",
      "che":lambda i:N7/f"mtlche_{i}.npz","aj":lambda i:AJ/f"nb25_{i}.npz"}
V=["tab","raw","n15","mtl13","che"]
CFG={"v26: tab raw n15 mtl13 che":V,
     "swap mtl13 -> nb25":["tab","raw","n15","aj","che"],
     "add nb25 as a sixth arm":V+["aj"]}
soft=lambda p,lo,hi: np.clip(p-hi,0,None)+np.clip(lo-p,0,None)
def st_rae(y,p,lo,hi): return float(soft(p,lo,hi).sum()/soft(np.full_like(y,y.mean()),lo,hi).sum())
def cal(yd,bd,lod,hid):
    k=float(np.clip(pearsonr(yd,bd)[0],0.3,0.95)); u=(bd-bd.mean())/bd.std()
    d=float(DS[int(np.argmin([st_rae(yd,yd.mean()+q+u*k*yd.std(),lod,hid) for q in DS]))])
    return k,d
meta=pd.read_csv(C/"meta_train.csv.gz")
def run(keys):
    out={}
    for iso in ISO:
        c=f"{iso}_pIC50_direct_inhibition"
        y=meta[c].to_numpy(float); lo=meta[f"{c}_conf_low"].to_numpy(float)
        hi=meta[f"{c}_conf_high"].to_numpy(float)
        A=[np.load(PATH[k](iso)) for k in keys]; aj=np.load(PATH["aj"](iso))
        dev,ho=aj["dev"],aj["ho"]                      # every config uses the nb25 arm's rows
        for a in A:
            if "dev" in a.files: dev=np.intersect1d(dev,a["dev"]); ho=np.intersect1d(ho,a["ho"])
        def col(a,key,idx):
            src=a[{"oof":"dev","hold":"ho"}[key]] if "dev" in a.files else (
                np.load(PATH["tab"](iso))["dev"] if key=="oof" else np.load(PATH["tab"](iso))["ho"])
            return a[key][np.isin(src,idx)]
        O=[col(a,"oof",dev) for a in A]; H=[col(a,"hold",ho) for a in A]
        yd,lod,hid=y[dev],lo[dev],hi[dev]
        g=np.round(np.arange(0,1.0001,0.1),2); best=None
        for w in itertools.product(g,repeat=len(keys)-1):
            s_=sum(w)
            if s_>1.0001: continue
            ws=list(w)+[1-s_]
            bd=sum(q*o for q,o in zip(ws,O))
            k_,d_=cal(yd,bd,lod,hid); u=(bd-bd.mean())/bd.std()
            s=st_rae(yd,yd.mean()+d_+u*k_*yd.std(),lod,hid)
            if best is None or s<best[0]: best=(s,ws)
        s_dev,ws=best
        bd=sum(q*o for q,o in zip(ws,O)); bh=sum(q*h for q,h in zip(ws,H))
        k_,d_=cal(yd,bd,lod,hid); v=(bh-bh.mean())/bh.std()
        out[iso]=(y[ho], y[ho].mean()+d_+v*k_*y[ho].std(), lo[ho], hi[ho],
                  dict(zip(keys,[round(x,2) for x in ws])), s_dev, ho, len(dev))
    return out
R={n:run(k) for n,k in CFG.items()}
print("development rows used:", {i:R[list(CFG)[0]][i][7] for i in ISO})
print(f"{'config':32s} " + " ".join(f"{i[3:]:>8s}" for i in ISO) + "  ho macro dev macro")
for n,r in R.items():
    v=[st_rae(*r[i][:4]) for i in ISO]
    print(f"{n:32s} " + " ".join(f"{x:8.4f}" for x in v) + f" {np.mean(v):9.4f} {np.mean([r[i][5] for i in ISO]):9.4f}")
for n in list(CFG)[1:]:
    print(f"\nweights, {n}:")
    for iso in ISO: print(f"  {iso}: " + ", ".join(f"{k} {v:.2f}" for k,v in R[n][iso][4].items() if v>0))
base=R[list(CFG)[0]]; print()
for n,r in list(R.items())[1:]:
    for iso in ISO: assert np.array_equal(base[iso][6], r[iso][6]), "holdout rows differ"
    rng=np.random.default_rng(0); N=6000; d=np.zeros(N)
    for t in range(N):
        sa=[];sb=[]
        for iso in ISO:
            ya,pa,loa,hia=base[iso][:4]; yb,pb,lob,hib=r[iso][:4]
            k=rng.integers(0,len(ya),len(ya))
            sa.append(st_rae(ya[k],pa[k],loa[k],hia[k])); sb.append(st_rae(yb[k],pb[k],lob[k],hib[k]))
        d[t]=np.mean(sb)-np.mean(sa)
    print(f"  {n:32s} {d.mean():+.4f}  CI [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]  P(better) {np.mean(d<0):.2f}")
