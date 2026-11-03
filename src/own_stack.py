"""Own-member hinge combiner: base = incumbent reference; minimise mean hinge + lam*|w|_1 over member deltas (LP).
Penalty chosen by inner folds inside each outer fold; outer-fold scores are the honest estimate."""
import os,glob,pickle,json,numpy as np,pandas as pd
from pathlib import Path
from scipy.optimize import linprog
D=Path(os.environ['DRIVE']);S=Path(os.environ['S']);OUT=Path(os.environ['OUT'])
M=pd.read_csv('work/restored/meta_train.csv.gz');H=np.load('work/restored/holdout_mask.npy').astype(bool);R=np.load('work/v31_reference_three.npz');RF=Path('references')
match=lambda p,r:r.mean()+(p-p.mean())/p.std()*r.std()
names=dict(zip(M.Molecule_Name,range(len(M))))
NB={'nb33':D/'nb33_a_wide_interval','nb35b':S/'nb35res','nb35c':S/'nb35cres','nb36':S/'nb36full'}
fold=np.full(len(M),-1);nbo={};nbt={}
for key,d in NB.items():
    o={};t=[]
    for k in range(5):
        z=np.load(d/f'fold{k}/development.npz');h=np.load(d/f'fold{k}/hold_test.npz');tg=list(z['targets']);t.append(h['test'])
        for i,n in enumerate(z['names']):
            if n in names:fold[names[n]]=k;o[names[n]]=z['pred'][i]
    nbo[key]=(o,tg);nbt[key]=np.mean(t,0)
LAMS=[1e-3,3e-3,1e-2,2e-2,4e-2,8e-2]
def fit(base,X,lo,hi,lam):
    n,m=X.shape
    # vars: w+ (m), w- (m), s (n); s >= lo-base-Xw ; s >= base+Xw-hi
    c=np.r_[np.full(2*m,lam),np.full(n,1/n)]
    A1=np.hstack([-X,X,-np.eye(n)]);b1=-(lo-base)
    A2=np.hstack([X,-X,-np.eye(n)]);b2=hi-base
    r=linprog(c,A_ub=np.vstack([A1,A2]),b_ub=np.r_[b1,b2],bounds=[(0,None)]*(2*m+n),method='highs')
    assert r.status==0,r.message
    return r.x[:m]-r.x[m:2*m]
res={};cols={}
for iso in ['CYP1A2','CYP2C9','CYP2D6','CYP3A4']:
    c=iso+'_pIC50_direct_inhibition';y=M[c].to_numpy(float);lo=M[c+'_conf_low'].to_numpy(float);hi=M[c+'_conf_high'].to_numpy(float)
    if iso!='CYP2D6':
        dev=R[iso+'_dev'];base=R[iso+'_oof_ref'];bt=R[iso+'_test_ref']
        if iso=='CYP2C9':z=np.load(RF/'nb29_a_control_CYP2C9.npz');base=match(.8*R[iso+'_oof_raw']+.2*z['oof'],base);bt=match(.8*R[iso+'_test_raw']+.2*z['test'],bt)
    else:
        dev=np.flatnonzero(~H&np.isfinite(y));zs=[np.load(RF/f'nb24_{a}_pseudo_CYP2D6.npz') for a in 'abce'];a=[np.load(p) for p in [Path('work/restored/tabqhts_CYP2D6.npz'),RF/'mtl_CYP2D6.npz',RF/'mtlche_CYP2D6.npz']]
        comb=lambda k:.85*np.mean([z[k] for z in zs],0)+.1*a[0][k]+.025*a[1][k]+.025*a[2][k]
        base=1.690096383627818+.714575767159324*comb('oof');bt=1.690096383627818+.714575767159324*comb('test')
    mem={};memt={}
    for f in sorted(glob.glob(str(D/f'*_{iso}.npz'))):
        try:z=np.load(f)
        except Exception:continue
        if not {'oof','dev','test'}<=set(z.files) or len(z['dev'])!=len(dev) or not np.array_equal(z['dev'],dev):continue
        if not (np.isfinite(z['oof']).all() and np.isfinite(z['test']).all() and z['test'].shape==(750,)):continue
        mem[Path(f).stem]=z['oof'];memt[Path(f).stem]=z['test']
    for key,(o,tg) in nbo.items():
        j=tg.index(iso.lower()+'_pic50_direct_inhibition');v=np.array([o[i][j] if i in o else np.nan for i in dev])
        if np.isfinite(v).all():mem[key]=v;memt[key]=nbt[key][:,j]
    # drop near-duplicates of the base and exact duplicates
    keys=[k for k in mem if np.std(mem[k]-base)>1e-6]
    uniq=[];seen=[]
    for k in keys:
        if any(np.corrcoef(mem[k],mem[u])[0,1]>0.9995 for u in uniq):continue
        uniq.append(k)
    X=np.column_stack([mem[k]-base for k in uniq]);Xt=np.column_stack([memt[k]-bt for k in uniq]);f=fold[dev]
    assert (f>=0).all()
    err=lambda p,r:(np.maximum(lo[r]-p,0)+np.maximum(p-hi[r],0)).sum()
    sc=lambda p,r:err(p,r)/err(np.full(len(r),y[r].mean()),r)
    oof=np.empty(len(dev));chosen=[]
    for k in range(5):
        tr=f!=k;te=~tr;inner=[]
        for lam in LAMS:
            e=0
            for kk in set(range(5))-{k}:
                a_=tr&(f!=kk);b_=f==kk;w=fit(base[a_],X[a_],lo[dev][a_],hi[dev][a_],lam);e+=err(base[b_]+X[b_]@w,dev[b_])
            inner.append(e)
        lam=LAMS[int(np.argmin(inner))];chosen.append(lam)
        w=fit(base[tr],X[tr],lo[dev][tr],hi[dev][tr],lam);oof[te]=base[te]+X[te]@w
    fd=[sc(oof[f==k],dev[f==k])-sc(base[f==k],dev[f==k]) for k in range(5)]
    lam=float(pd.Series(chosen).mode().iloc[0]);w=fit(base,X,lo[dev],hi[dev],lam)
    res[iso]=dict(members=len(uniq),base=sc(base,dev),stack=sc(oof,dev),fold_deltas=fd,improving=int(sum(v<0 for v in fd)),lams=chosen,final_lam=lam,top={uniq[i]:round(float(w[i]),3) for i in np.argsort(-np.abs(w))[:8] if abs(w[i])>1e-4})
    cols[iso]=dict(names=M.Molecule_Name.iloc[dev].to_numpy(str),oof_stack=oof,oof_base=base,test_stack=bt+Xt@w,test_base=bt)
    print(iso,json.dumps(default=str,obj={k:(round(v,4) if isinstance(v,float) else v) for k,v in res[iso].items() if k!='fold_deltas'}),[round(v,4) for v in fd],flush=True)
pickle.dump(cols,open(OUT,'wb'))
