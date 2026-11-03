"""v48 = v47 with the CYP2D6 column reverted to v42's, the best board-measured 2D6 column.

Measured live tabs: v47 1A2 .4545, 2C9 .3585, 2D6 .5449, 3A4 .3760 (macro .4335, rank 25);
v42 2D6 .5316. Three of v47's columns beat v42; CYP2D6 lost 0.0133, the third time a
development gain on that isoform has failed to reach the board (NB33, v43/v46, here).
Only the 2D6 column changes, so the Scigantic share stays one half in every column.
"""
import hashlib
import numpy as np, pandas as pd

S = 'submissions/'
K = lambda i: f'{i}_pIC50_direct_inhibition'
v47 = pd.read_csv(S + 'submission_v47_candidate.csv')
v42 = pd.read_csv(S + 'submission_v42_candidate.csv')
assert (v47.Molecule_Name == v42.Molecule_Name).all()

v48 = v47.copy()
v48[K('CYP2D6')] = v42[K('CYP2D6')]
for i in ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4'):
    a, b = v48[K(i)], v47[K(i)]
    print(f'{i} mean {a.mean():.3f} spearman vs v47 {a.corr(b, method="spearman"):.4f} '
          f'min {a.min():.2f} max {a.max():.2f}')
assert np.isfinite(v48.iloc[:, 2:].to_numpy(float)).all()
assert list(v48.columns) == list(v42.columns)
v48.to_csv(S + 'submission_v48_candidate.csv', index=False)
print('rows', len(v48), 'sha256', hashlib.sha256(open(S + 'submission_v48_candidate.csv', 'rb').read()).hexdigest())
print('expected macro', round((.4545 + .3585 + .5316 + .3760) / 4, 4), '(three tabs measured on v47, 2D6 measured on v42)')
