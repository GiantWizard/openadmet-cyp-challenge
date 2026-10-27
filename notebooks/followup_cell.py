# ============================================================================
# Paste into a new cell in the running notebook. The session is already warm,
# so this reuses CACHE, dev_index, make, record and OUT from the setup cell.
#
# Part 1 is the one that matters: `full_raw` is the arm the blend kept, but the
# notebook only produced development out-of-fold predictions. Shipping it needs
# holdout predictions, to check the gain survives rows nothing was fitted to,
# and blind-test predictions to build the submission. About five minutes.
#
# Parts 2 and 3 are the follow-ups today's result implies. Skip them if the
# runtime is close to timing out; Part 1 alone is enough to submit.
# ============================================================================
import numpy as np, time
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

hold_mask = np.load(CACHE / 'holdout_mask.npy')
EMB_TE = np.load(CACHE / 'emb_medium_test.npy').astype(np.float32)
QH_TE = np.load(CACHE / 'qhts_te.npy')
AUX_TE = np.load(CACHE / 'aux_test.npy')

RAW = dict(device=DEVICE, random_state=42, ignore_pretraining_limits=True)


def full_arm(name, Xtr, Xte, n_est=None):
    """Dev out-of-fold, holdout and blind test, from one feature matrix."""
    kw = dict(RAW, **({'n_estimators': n_est} if n_est else {}))
    for iso in ISOFORMS:
        y = meta[f'{iso}_pIC50_direct_inhibition'].to_numpy(float)
        obs = np.isfinite(y)
        dev = np.flatnonzero(obs & ~hold_mask)
        ho = np.flatnonzero(obs & hold_mask)
        t0 = time.time()

        oof = np.full(len(dev), np.nan)
        for tr, va in GroupKFold(N_FOLDS).split(dev, dev, G[dev]):
            m = TabPFNRegressor(**kw).fit(Xtr[dev[tr]], y[dev[tr]])
            oof[va] = m.predict(Xtr[dev[va]])

        m = TabPFNRegressor(**kw).fit(Xtr[dev], y[dev])
        hp = m.predict(Xtr[ho])                      # holdout, fitted on dev only
        allr = np.flatnonzero(obs)
        m2 = TabPFNRegressor(**kw).fit(Xtr[allr], y[allr])
        tp = m2.predict(Xte)                         # blind test, fitted on everything

        np.savez_compressed(OUT / f'{name}_{iso}.npz',
                            oof=oof, dev=dev, hold=hp, ho=ho, test=tp)
        d = float(spearmanr(y[dev], oof).statistic)
        h = float(spearmanr(y[ho], hp).statistic)
        record(experiment='D', arm=name, iso=iso, spearman=d, holdout_spearman=h,
               shipped=SHIPPED[iso], seconds=round(time.time() - t0, 1))
        print(f'  {name:12s} {iso} dev {d:.4f} | holdout {h:.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)


# --- Part 1, the shippable arm -------------------------------------------
S_TR = np.hstack([EMB, QH]).astype(np.float32)
S_TE = np.hstack([EMB_TE, QH_TE]).astype(np.float32)
print('part 1: full_raw with holdout and test', S_TR.shape, flush=True)
full_arm('full_raw_sub', S_TR, S_TE)

# --- Part 2, raw descriptors ---------------------------------------------
# We concluded last week that descriptors hurt TabPFN. That test compressed
# them to 100 components, and compression is exactly what today's result shows
# was doing the damage, so the conclusion was about the PCA rather than the
# descriptors. Rebuilt here from SMILES so nothing extra has to be uploaded.
!pip -q install rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
RDLogger.DisableLog('rdApp.*')

names = [d[0] for d in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)

def descriptors(smiles):
    out = np.zeros((len(smiles), 2048 + len(names)), np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        out[i, :2048] = gen.GetFingerprintAsNumPy(m)
        out[i, 2048:] = np.nan_to_num(np.array(calc.CalcDescriptors(m), np.float32),
                                      nan=0.0, posinf=0.0, neginf=0.0)
    return out

SM_TE = pd.read_csv(CACHE / 'meta_test.csv.gz')['SMILES'].to_numpy()
D_TR, D_TE = descriptors(SMILES_TR), descriptors(SM_TE)
print('descriptors', D_TR.shape, flush=True)
full_arm('raw_desc',
         np.hstack([EMB, D_TR, QH, AUX]).astype(np.float32),
         np.hstack([EMB_TE, D_TE, QH_TE, AUX_TE]).astype(np.float32))

# --- Part 3, three representations in one model --------------------------
# The language models were tested as separate LightGBM arms whose predictions
# were mixed, worth +0.0014 and +0.0011. Handing every column to one model that
# we now know prefers raw width is a different question.
!pip -q install transformers
from transformers import AutoTokenizer, AutoModel

def embed(repo, smiles, remote, bs=64):
    tk = AutoTokenizer.from_pretrained(repo, trust_remote_code=remote)
    md = AutoModel.from_pretrained(repo, trust_remote_code=remote).to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), bs):
            b = tk(list(smiles[i:i + bs]), padding=True, truncation=True,
                   max_length=202, return_tensors='pt').to(DEVICE)
            h = md(**b).last_hidden_state
            m = b['attention_mask'].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    del md; torch.cuda.empty_cache()
    return np.vstack(out).astype(np.float32)

REPOS = [('ibm-research/MoLFormer-XL-both-10pct', True),
         ('DeepChem/ChemBERTa-77M-MTR', False)]
extra_tr = [embed(r, SMILES_TR, k) for r, k in REPOS]
extra_te = [embed(r, SM_TE, k) for r, k in REPOS]
full_arm('three_rep',
         np.hstack([EMB] + extra_tr + [QH, AUX]).astype(np.float32),
         np.hstack([EMB_TE] + extra_te + [QH_TE, AUX_TE]).astype(np.float32))

# Re-download so the new .npz files come back too.
import shutil
from google.colab import files
shutil.make_archive('/content/cyp_gpu_results2', 'zip', OUT)
files.download('/content/cyp_gpu_results2.zip')
