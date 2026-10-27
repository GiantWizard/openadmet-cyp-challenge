"""Write submission_v27.csv: v26 with ONLY the CYP2D6 pseudo-label arm swapped for another nb24 run.

Usage: python make_v27.py <path to nb24_<RUN>_pseudo_CYP2D6.npz>

Was: Write submission_v25.csv: the five-arm shape, tab raw n15 mtl13 che.

Holdout 0.5827 against v21's 0.5891, paired bootstrap -0.0064, P(better) 0.89, which is
the same evidence strength v21 itself shipped on (-0.0076, P 0.91). The gain is isolated
to n15 by a control that drops it from the same small pool (-0.0014 instead of -0.0043).
"""
from __future__ import annotations
import itertools, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import sys, os
CHECK=bool(os.environ.get("V27_CHECK"))   # reproduce v26 from its own arm: all four columns must match, nothing written
_files=[Path(f) for f in sys.argv[1].split(",")]; assert all(f.exists() for f in _files), _files
_Z=[np.load(f) for f in _files]
for _z in _Z[1:]: assert np.array_equal(_z["dev"],_Z[0]["dev"]) and np.array_equal(_z["ho"],_Z[0]["ho"])
import tempfile
ARM=Path(tempfile.mkdtemp())/"arm.npz"
np.savez(ARM,dev=_Z[0]["dev"],ho=_Z[0]["ho"],oof=np.mean([z["oof"] for z in _Z],0),
         hold=np.mean([z["hold"] for z in _Z],0),test=np.mean([z["test"] for z in _Z],0))
N15_W=float(os.environ["N15_W"]) if os.environ.get("N15_W") else None   # CYP2D6 pseudo-arm weight set by hand
C=Path("clean/cache"); P1=Path(os.environ.get("P1_DIR","/tmp/part1")); N5=Path("/tmp/nb5"); N7=Path(os.environ.get("N7_DIR","/tmp/nb7"))
N15=Path("/tmp/nb15")
OUT=Path("<repo>/01-projects/portfolio/"
         "openadmet-cyp-challenge/submissions")
ISO=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; DS=np.arange(-0.6,0.81,0.05)
BLIND={"CYP1A2":(4.412,1.553),"CYP2C9":(4.830,1.101),"CYP2D6":(3.420,1.599),"CYP3A4":(4.880,1.272)}
KEYS=["tab","raw","n15","mtl13","che"]
PATH={"tab":lambda i:C/f"tabqhts_{i}.npz","raw":lambda i:P1/f"full_raw_sub_{i}.npz",
      "n15":lambda i:(ARM if i=="CYP2D6" else N15/f"nb15_{i}.npz"),"mtl13":lambda i:N5/f"mtl_{i}.npz",
      "che":lambda i:N7/f"mtlche_{i}.npz"}
soft=lambda p,lo,hi: np.clip(p-hi,0,None)+np.clip(lo-p,0,None)
def st_rae(y,p,lo,hi): return float(soft(p,lo,hi).sum()/soft(np.full_like(y,y.mean()),lo,hi).sum())
def cal(yd,bd,lod,hid):
    k=float(np.clip(pearsonr(yd,bd)[0],0.3,0.95)); u=(bd-bd.mean())/bd.std()
    d=float(DS[int(np.argmin([st_rae(yd,yd.mean()+q+u*k*yd.std(),lod,hid) for q in DS]))])
    return k,d
meta=pd.read_csv(C/"meta_train.csv.gz"); te=pd.read_csv(C/"meta_test.csv.gz")
sub=te[["SMILES","Molecule_Name"]].copy()
print(f"{'iso':8s} {'holdout ST-RAE':>15s}  weights")
hs=[]
for iso in ISO:
    c=f"{iso}_pIC50_direct_inhibition"
    y=meta[c].to_numpy(float); lo=meta[f"{c}_conf_low"].to_numpy(float)
    hi=meta[f"{c}_conf_high"].to_numpy(float)
    A=[np.load(PATH[k](iso)) for k in KEYS]
    dev,ho=A[0]["dev"],A[0]["ho"]
    for a in A[1:]:
        if "dev" in a.files: dev=np.intersect1d(dev,a["dev"]); ho=np.intersect1d(ho,a["ho"])
    def col(a,key,idx):
        src=a[{"oof":"dev","hold":"ho"}[key]] if "dev" in a.files else (
            A[0]["dev"] if key=="oof" else A[0]["ho"])
        return a[key][np.isin(src,idx)]
    O=[col(a,"oof",dev) for a in A]; H=[col(a,"hold",ho) for a in A]; T=[a["test"] for a in A]
    yd,lod,hid=y[dev],lo[dev],hi[dev]
    g=np.round(np.arange(0,1.0001,0.1),2); best=None
    for w in itertools.product(g,repeat=len(KEYS)-1):
        s_=sum(w)
        if s_>1.0001: continue
        ws=list(w)+[1-s_]
        bd=sum(q*o for q,o in zip(ws,O))
        k_,d_=cal(yd,bd,lod,hid); u=(bd-bd.mean())/bd.std()
        s=st_rae(yd,yd.mean()+d_+u*k_*yd.std(),lod,hid)
        if best is None or s<best[0]: best=(s,ws)
    ws=best[1]
    if N15_W is not None and iso=="CYP2D6":
        v26w=ws; oth=[i for i,k in enumerate(KEYS) if k!="n15"]; tot=sum(v26w[i] for i in oth)
        ws=[N15_W if k=="n15" else v26w[i]*(1-N15_W)/tot for i,k in enumerate(KEYS)]
    bd=sum(q*o for q,o in zip(ws,O)); bh=sum(q*h for q,h in zip(ws,H)); bt=sum(q*t for q,t in zip(ws,T))
    k_,d_=cal(yd,bd,lod,hid); v=(bh-bh.mean())/bh.std()
    h=st_rae(y[ho], y[ho].mean()+d_+v*k_*y[ho].std(), lo[ho], hi[ho]); hs.append(h)
    centre,sdv=BLIND[iso]
    sub[c]=centre+d_+(bt-bt.mean())*(k_*sdv/bt.std())
    print(f"{iso:8s} {h:15.4f}  " + ", ".join(f"{a} {b:.2f}" for a,b in zip(KEYS,ws) if b>0))
print(f"\nmacro holdout {np.mean(hs):.4f}  (v21 0.5891)")
assert len(sub)==750 and list(sub.columns)==["SMILES","Molecule_Name"]+[f"{i}_pIC50_direct_inhibition" for i in ISO]
assert sub.notna().all().all() and np.isfinite(sub[[f"{i}_pIC50_direct_inhibition" for i in ISO]].to_numpy()).all()
old=pd.read_csv(OUT/"submission_v21.csv")
assert (sub["SMILES"].to_numpy()==old["SMILES"].to_numpy()).all(), "row order differs from v21"
v26=pd.read_csv(OUT/"submission_v26.csv")
for i in ISO:
    c=f"{i}_pIC50_direct_inhibition"
    same=np.allclose(sub[c].to_numpy(),v26[c].to_numpy(),atol=1e-9,rtol=0)
    print(f"  column {i}: identical to v26 = {same}")
    assert same == (CHECK or i != "CYP2D6"), f"{i}: expected identical={i!='CYP2D6'}"
if CHECK: print("CHECK passed: v26 reproduced from its own arm"); sys.exit(0)
p=OUT/f"submission_{os.environ.get('OUTV','v27')}.csv"; sub.to_csv(p,index=False)
print(f"\nwrote {p}  ({len(sub)} rows)")
print("per-isoform shipped distribution (v27 vs v21):")
for i in ISO:
    c=f"{i}_pIC50_direct_inhibition"
    print(f"  {i}: mean {sub[c].mean():.3f} sd {sub[c].std():.3f}   "
          f"| v21 mean {old[c].mean():.3f} sd {old[c].std():.3f}   "
          f"| rho to v21 {spearmanr(sub[c],old[c]).statistic:.4f}")
