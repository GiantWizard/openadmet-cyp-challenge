"""v43 (own members only) and v44 (v43 averaged with Scigantic R25, weight .5 fixed in advance).
Columns come from src/own_stack.py: incumbent reference plus an L1-penalised hinge-fitted combination of this
project's saved members (penalty chosen in nested folds), restored to the reference's mean and sd, then the same
board-measured placement as v40: CYP1A2 spread x1.08, CYP3A4 spread x0.92, CYP2D6 floor (3.5, 0.5), CYP2C9 none."""
import numpy as np, pandas as pd
S = 'submissions/'
K = lambda i: f'{i}_pIC50_direct_inhibition'
spread = lambda x, f: x.mean() + (x - x.mean()) * f
squash = lambda x, c, g: np.where(x < c, c - g * (c - x), x)
restore = lambda raw, ref: (raw - raw.mean()) / raw.std() * ref.std() + ref.mean()
Z = np.load('references/own_stack/own_stack_test.npz')
v40 = pd.read_csv(S + 'submission_v40_candidate.csv')
v35 = pd.read_csv(S + 'submission_v35_candidate.csv')
v43 = v40.copy()
for i in ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4'):
    # CYP2D6's shipped column was placed on the blind population (mean 3.47), so restore to v35's column, not the raw reference
    ref = v35[K(i)].to_numpy(float) if i == 'CYP2D6' else Z[f'{i}_test_base']
    x = restore(Z[f'{i}_test_stack'], ref)
    x = {'CYP1A2': lambda v: spread(v, 1.08), 'CYP3A4': lambda v: spread(v, 0.92),
         'CYP2D6': lambda v: squash(v, 3.5, 0.5), 'CYP2C9': lambda v: v}[i](x)
    v43[K(i)] = x
    print(i, 'spearman vs v40', round(pd.Series(x).corr(v40[K(i)], method='spearman'), 3), 'mean', round(x.mean(), 3), 'v40 mean', round(v40[K(i)].mean(), 3))
assert np.isfinite(v43.iloc[:, 2:].to_numpy(float)).all()
v43.to_csv(S + 'submission_v43_candidate.csv', index=False)
sg = pd.read_csv('references/scigantic/R25.csv').set_index('Molecule_Name').loc[v43.Molecule_Name].reset_index()
v44 = v43.copy()
for i in ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4'):
    v44[K(i)] = .5 * v43[K(i)] + .5 * sg[K(i)]
v44.to_csv(S + 'submission_v44_candidate.csv', index=False)
# v45 (added 2026-10-28 at Jeffery's request to reduce the R25 share): 0.7 x v43 + 0.3 x R25, built the same way as v44.
