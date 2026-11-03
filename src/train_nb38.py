"""NB38 arm 1: random-initialised wide D-MPNN (2048 hidden, depth 4), MAE on all 22 heads, per-compound precision weights.
Design follows Scigantic's published arm B (random init, MAE, inverse-variance weights); this project's own folds, purge,
weights and code. Weights: for compounds with scored labels, mean inverse variance of their credible intervals
(sd = width/3.92), divided by the median, clipped to [0.25, 4]; auxiliary-only compounds weigh 1.
"""
import argparse,json,time,resource,hashlib,os,sys
from importlib.metadata import version
from pathlib import Path
import numpy as np,pandas as pd,torch,lightning as L
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint,Callback
from chemprop import data,featurizers,models,nn

class Progress(Callback):
    def __init__(self,out):self.out=out;self.start=time.time();self.rows=json.loads(out.read_text()) if out.exists() else []
    def on_train_epoch_end(self,trainer,pl_module):
        row=dict(epoch=trainer.current_epoch,minutes=(time.time()-self.start)/60,metrics={k:float(v.cpu()) for k,v in trainer.callback_metrics.items()},rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**30 if sys.platform=='darwin' else 2**20))
        if torch.cuda.is_available():row['gpu_peak_gib']=torch.cuda.max_memory_allocated()/2**30
        self.rows.append(row);self.out.write_text(json.dumps(self.rows,indent=2));print(json.dumps(row),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--fold',type=int,default=0);ap.add_argument('--batch-size',type=int,default=128);args=ap.parse_args();assert args.batch_size in (32,64,128)
    O=Path(os.environ['NB33_OUT']);audit=json.loads((O/'data_audit.json').read_text());assert audit['scaffold_disjoint']
    torch.set_num_threads(2);device=os.environ.get('NB33_DEVICE','gpu');assert device=='cpu' or torch.cuda.is_available()
    torch.set_float32_matmul_precision('high')
    L.seed_everything(int(os.environ.get('NB33_SEED','42'))+args.fold,workers=True)
    P=pd.read_csv(O/'train.csv');B=pd.read_csv(O/'bounds.csv');F=np.load(O/'folds.npy')
    targets=[c for c in P if c not in ('Molecule_Name','smiles','InChIKey')];nt=len(targets);assert nt==22
    scored=[c for c in targets if c.endswith('_pic50_direct_inhibition')];sc=[targets.index(c) for c in scored]
    y=P[targets].to_numpy(float);extras=[B[c.split('_')[0].upper()+'_'+side].to_numpy(float) for side in ['lo','hi'] for c in scored]
    lo=np.column_stack(extras[:4]);hi=np.column_stack(extras[4:]);lab=np.isfinite(y[:,sc])
    for k in range(4):assert np.isfinite(lo[lab[:,k],k]).all() and (lo[lab[:,k],k]<=hi[lab[:,k],k]).all()
    iv=np.where(lab,1/np.maximum((hi-lo)/3.92,0.02)**2,np.nan);has=lab.any(1)
    raw=np.nanmean(np.where(lab,iv,np.nan)[has],axis=1);sw=np.ones(len(y));sw[has]=np.clip(raw/np.median(raw),0.25,4.0)
    tr=np.flatnonzero((F!=args.fold)&(F!=(args.fold+1)%5));va=np.flatnonzero(F==(args.fold+1)%5);ev=np.flatnonzero(F==args.fold)
    if args.smoke:tr=np.unique(np.r_[tr[:128],*[tr[np.isfinite(y[tr,j])][:6] for j in range(y.shape[1])]]);va=va[:8];ev=ev[:8]
    feat=featurizers.SimpleMoleculeMolGraphFeaturizer()
    def dataset(rows):return data.MoleculeDataset([data.MoleculeDatapoint.from_smi(P.smiles.iloc[i],y[i],weight=float(sw[i])) for i in rows],feat)
    td=dataset(tr);scaler=td.normalize_targets();vd=dataset(va);vd.normalize_targets(scaler)
    mp=nn.BondMessagePassing(d_h=2048,depth=4,dropout=.2)
    ffn=nn.RegressionFFN(n_tasks=nt,input_dim=2048,hidden_dim=1024,n_layers=2,dropout=.2,criterion=nn.metrics.MAE(),output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    model=models.MPNN(mp,nn.NormAggregation(),ffn,batch_norm=True,metrics=[nn.metrics.MAE()],warmup_epochs=2,init_lr=1e-4,max_lr=1e-3,final_lr=1e-4)
    # Batch128 matches the published recipe; smoke tests use smaller batches.
    tl=data.build_dataloader(td,batch_size=args.batch_size,num_workers=0);vl=data.build_dataloader(vd,batch_size=args.batch_size,num_workers=0,shuffle=False)
    dest=O/('smoke' if args.smoke else (f'benchmark{args.batch_size}' if args.benchmark else f'fold{args.fold}'));dest.mkdir(exist_ok=True)
    signature=dict(implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions={p:version(p) for p in ['chemprop','lightning','torch','numpy','rdkit']},precision='32-true; matmul high',weights=hashlib.sha256(sw.tobytes()).hexdigest(),fold=args.fold,batch=args.batch_size,accumulation=128//args.batch_size,table=hashlib.sha256((O/'train.csv').read_bytes()).hexdigest(),folds=hashlib.sha256((O/'folds.npy').read_bytes()).hexdigest(),plan=hashlib.sha256((O/'plan.json').read_bytes()).hexdigest())
    if (dest/'signature.json').exists():assert json.loads((dest/'signature.json').read_text())==signature
    else:(dest/'signature.json').write_text(json.dumps(signature,indent=2))
    if (dest/'development.npz').exists() and (dest/'hold_test.npz').exists():
        print('Completed fold: skipping',args.fold,flush=True);return
    progress=Progress(dest/'progress.json');checkpoint=ModelCheckpoint(dirpath=str(dest),monitor='val_loss',mode='min',save_top_k=1,save_last=True)
    opts=dict(accelerator=device,devices=1,max_epochs=200,accumulate_grad_batches=128//args.batch_size,logger=False,enable_progress_bar=False,callbacks=[checkpoint,EarlyStopping(monitor='val_loss',patience=30,mode='min'),progress],num_sanity_val_steps=0)
    if args.benchmark:opts.update(max_epochs=3,limit_train_batches=12,limit_val_batches=4)
    if args.smoke:opts.update(max_epochs=3,limit_train_batches=1,limit_val_batches=1,accumulate_grad_batches=1)
    trainer=L.Trainer(**opts);start=time.time()
    resume=str(dest/'last.ckpt') if (dest/'last.ckpt').exists() and not args.benchmark else None
    if not (dest/'result.json').exists():trainer.fit(model,tl,vl,ckpt_path=resume,weights_only=False)
    else:checkpoint.best_model_path=json.loads((dest/'result.json').read_text())['best_checkpoint']
    result=dict(seconds=time.time()-start,train_rows=len(tr),val_rows=len(va),eval_rows=len(ev),benchmark=args.benchmark,batch_size=args.batch_size,accumulation=128//args.batch_size,best_checkpoint=checkpoint.best_model_path,epochs=trainer.current_epoch,device=device,smoke=args.smoke)
    if not (dest/'result.json').exists():(dest/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if args.benchmark:return
    best=models.MPNN.load_from_checkpoint(checkpoint.best_model_path,map_location='cpu')
    ed=dataset(ev);el=data.build_dataloader(ed,batch_size=args.batch_size,num_workers=0,shuffle=False)
    pr=L.Trainer(accelerator=device,devices=1,logger=False,enable_progress_bar=False)
    pred=np.concatenate(pr.predict(best,el));np.savez_compressed(dest/'development.npz',names=P.Molecule_Name.iloc[ev].to_numpy(str),pred=pred[:,:nt],targets=np.asarray(targets))
    meta=pd.read_csv('work/restored/meta_train.csv.gz');test=pd.read_csv('work/restored/meta_test.csv.gz');hold=np.flatnonzero(np.load('work/restored/holdout_mask.npy').astype(bool))
    if args.smoke:hold=hold[:4];test=test.iloc[:4]
    smiles=list(meta.SMILES.iloc[hold])+list(test.SMILES)
    hd=data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smiles],feat);hl=data.build_dataloader(hd,batch_size=args.batch_size,num_workers=0,shuffle=False)
    hp=np.concatenate(pr.predict(best,hl));assert np.isfinite(hp).all()
    np.savez_compressed(dest/'hold_test.npz',hold_rows=hold,hold=hp[:len(hold),:nt],test=hp[len(hold):,:nt],targets=np.asarray(targets))
    print('Fold saved including unscored holdout/blind predictions:',args.fold,flush=True)

if __name__=='__main__':main()
