"""v46 (own members only, with NB38 arm 1 added to the combiner) and v47 (v46 averaged with Scigantic R25 at .5).
Same construction and placement as src/make_v43.py; the only change is the stack, from own_stack.py with NB38 in NB."""
import numpy as np, pandas as pd
S = 'submissions/'
K = lambda i: f'{i}_pIC50_direct_inhibition'
spread = lambda x, f: x.mean() + (x - x.mean()) * f
squash = lambda x, c, g: np.where(x < c, c - g * (c - x), x)
restore = lambda raw, ref: (raw - raw.mean()) / raw.std() * ref.std() + ref.mean()
Z = np.load('references/own_stack/own_stack38_test.npz')
v40 = pd.read_csv(S + 'submission_v40_candidate.csv')
v35 = pd.read_csv(S + 'submission_v35_candidate.csv')
v43 = pd.read_csv(S + 'submission_v43_candidate.csv')
v46 = v40.copy()
for i in ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4'):
    ref = v35[K(i)].to_numpy(float) if i == 'CYP2D6' else Z[f'{i}_test_base']
    x = restore(Z[f'{i}_test_stack'], ref)
    x = {'CYP1A2': lambda v: spread(v, 1.08), 'CYP3A4': lambda v: spread(v, 0.92),
         'CYP2D6': lambda v: squash(v, 3.5, 0.5), 'CYP2C9': lambda v: v}[i](x)
    v46[K(i)] = x
    print(i, 'pearson vs v43', round(np.corrcoef(x, v43[K(i)])[0, 1], 4), 'mean', round(x.mean(), 3), 'v43 mean', round(v43[K(i)].mean(), 3))
assert np.isfinite(v46.iloc[:, 2:].to_numpy(float)).all()
v46.to_csv(S + 'submission_v46_candidate.csv', index=False)
sg = pd.read_csv('references/scigantic/R25.csv').set_index('Molecule_Name').loc[v46.Molecule_Name].reset_index()
v47 = v46.copy()
for i in ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4'):
    v47[K(i)] = .5 * v46[K(i)] + .5 * sg[K(i)]
v47.to_csv(S + 'submission_v47_candidate.csv', index=False)
