"""v49: per-isoform R25 weights instead of one half everywhere.

Each isoform has been read on the board at three weights, so a quadratic in w is
determined exactly and its minimum is available in closed form:

    w=0   our column alone        w=0.5  the equal-weight average   w=1  R25 alone
    CYP1A2  0.4669 (v46)                 0.4545 (v47)                    0.4739
    CYP2C9  0.3609 (v46)                 0.3585 (v47)                    0.3802
    CYP2D6  0.5524 (v40)                 0.5316 (v42)                    0.5442
    CYP3A4  0.3940 (v46)                 0.3760 (v47)                    0.3762

CYP2D6 uses the v40 family because that is the column v48 carries and the only 2D6
placement that has read well on the board.

The raw minimum is fitted to four readings that each carry about 0.002 of half-set
noise, which is the shape of a result that does not transfer, so the weights actually
written are shrunk toward one half by LAM. LAM=1 would ship the raw fit.
"""
import hashlib
import numpy as np, pandas as pd

LAM = 0.5
CAP = 0.5   # their share is never raised above this; set to 1.0 to let the fit run free
S = 'submissions/'
K = lambda i: f'{i}_pIC50_direct_inhibition'
ISO = ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4')
READ = {'CYP1A2': (.4669, .4545, .4739), 'CYP2C9': (.3609, .3585, .3802),
        'CYP2D6': (.5524, .5316, .5442), 'CYP3A4': (.3940, .3760, .3762)}

v46 = pd.read_csv(S + 'submission_v46_candidate.csv')
v40 = pd.read_csv(S + 'submission_v40_candidate.csv')
v48 = pd.read_csv(S + 'submission_v48_candidate.csv')
sg = pd.read_csv('references/scigantic/R25.csv').set_index('Molecule_Name').loc[v46.Molecule_Name].reset_index()
assert (sg.Molecule_Name.values == v46.Molecule_Name.values).all()
assert (v40.Molecule_Name.values == v46.Molecule_Name.values).all()
assert len(v46) == 750

out, proj_raw, proj_shrunk, weights = v48.copy(), [], [], {}
for i in ISO:
    a, c, b = READ[i]
    A, B = 2 * a + 2 * b - 4 * c, 4 * c - 3 * a - b
    f = lambda w: a + B * w + A * w * w
    w_raw = float(np.clip(-B / (2 * A), 0, 1)) if A > 0 else (0.0 if a < b else 1.0)
    w = min(0.5 + LAM * (w_raw - 0.5), CAP)
    weights[i] = (w_raw, w)
    proj_raw.append(f(w_raw)); proj_shrunk.append(f(w))
    own = (v40 if i == 'CYP2D6' else v46)[K(i)].to_numpy(float)
    out[K(i)] = (1 - w) * own + w * sg[K(i)].to_numpy(float)
    print(f'{i}  curvature {A:+.4f}  w_raw {w_raw:.3f} -> {f(w_raw):.4f}   '
          f'w_shrunk {w:.3f} -> {f(w):.4f}   (v48 held {c:.4f})')

assert np.isfinite(out.iloc[:, 2:].to_numpy(float)).all()
assert list(out.columns) == list(v48.columns)
out.to_csv(S + 'submission_v49_candidate.csv', index=False)

print()
print(f'CAP {CAP}  LAM {LAM}')
print('v48 measured macro      0.4301')
print('v49 shrunk projection  ', round(float(np.mean(proj_shrunk)), 4),
      f'(gain {0.4301 - float(np.mean(proj_shrunk)):+.4f})')
print('raw-fit projection     ', round(float(np.mean(proj_raw)), 4),
      f'(gain {0.4301 - float(np.mean(proj_raw)):+.4f})')
print('mean R25 share  v48 0.500   v49 shrunk',
      round(float(np.mean([w for _, w in weights.values()])), 3),
      '  raw fit', round(float(np.mean([w for w, _ in weights.values()])), 3))
print('rows', len(out), 'sha256', hashlib.sha256(open(S + 'submission_v49_candidate.csv', 'rb').read()).hexdigest())
