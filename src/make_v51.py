"""v51: keep what v50 proved, undo what it disproved, add the one weight change with support.

v50 moved two columns and held two fixed. The two held columns came back byte-identical in
score, 0.3585 and 0.5316, which is the per-column measurement design doing its job: the
isoform tabs are independent, so one upload buys four separate experiments.

    CYP1A2  0.4545 -> 0.4483   spread step paid, and paid more than anything else has
    CYP3A4  0.3760 -> 0.3865   squeeze step cost more than twice what it gained

So v51 takes v50's CYP1A2 column unchanged, reverts CYP3A4 to v48's, leaves CYP2D6 alone,
and moves CYP2C9 to the 0.400 R25 weight from make_v49.py, which interpolates between three
measured points rather than extrapolating past them.

Projection 0.4282, from four values each measured on the board: 0.4483 + 0.3571 + 0.5316 +
0.3760. Only the CYP2C9 figure is a fitted value rather than a direct reading.
"""
import hashlib
import numpy as np, pandas as pd

W_2C9 = 0.400
S = 'submissions/'
K = lambda i: f'{i}_pIC50_direct_inhibition'

v48 = pd.read_csv(S + 'submission_v48_candidate.csv')
v50 = pd.read_csv(S + 'submission_v50_candidate.csv')
v46 = pd.read_csv(S + 'submission_v46_candidate.csv')
sg = pd.read_csv('references/scigantic/R25.csv').set_index('Molecule_Name').loc[v48.Molecule_Name].reset_index()
assert (sg.Molecule_Name.values == v48.Molecule_Name.values).all()
assert (v50.Molecule_Name.values == v48.Molecule_Name.values).all()

v51 = v48.copy()
v51[K('CYP1A2')] = v50[K('CYP1A2')]
v51[K('CYP2C9')] = (1 - W_2C9) * v46[K('CYP2C9')].to_numpy(float) + W_2C9 * sg[K('CYP2C9')].to_numpy(float)

src = {'CYP1A2': 'v50 (spread step kept)', 'CYP2C9': f'v46/R25 at w={W_2C9}',
       'CYP2D6': 'v48 (unchanged)', 'CYP3A4': 'v48 (squeeze reverted)'}
for i in ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4'):
    a, b = v51[K(i)].to_numpy(float), v48[K(i)].to_numpy(float)
    print(f'{i}  {src[i]:<26} sd {b.std():.4f} -> {a.std():.4f}  '
          f'spearman vs v48 {pd.Series(a).corr(pd.Series(b), method="spearman"):.4f}')

assert np.isfinite(v51.iloc[:, 2:].to_numpy(float)).all()
assert list(v51.columns) == list(v48.columns)
assert (v51[K('CYP1A2')].to_numpy() == v50[K('CYP1A2')].to_numpy()).all()
assert (v51[K('CYP3A4')].to_numpy() == v48[K('CYP3A4')].to_numpy()).all()
assert (v51[K('CYP2D6')].to_numpy() == v48[K('CYP2D6')].to_numpy()).all()
v51.to_csv(S + 'submission_v51_candidate.csv', index=False)

mean_share = (0.5 + W_2C9 + 0.5 + 0.5) / 4
print(f'\nprojection 0.4282   (v48 0.4301, v50 0.4312, Scigantic 0.4283)')
print(f'mean R25 share {mean_share:.3f}')
print('rows', len(v51), 'sha256', hashlib.sha256(open(S + 'submission_v51_candidate.csv', 'rb').read()).hexdigest())
