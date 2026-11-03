"""v50: one single-parameter placement step per column on top of v48, same idea as v41.

v41 planned CYP1A2 spread 1.08 -> 1.16 and CYP3A4 spread 0.92 -> 0.85 and was never
uploaded. The first steps in those directions read -0.0049 and -0.0044 on their own tabs,
so the direction has board evidence and the size of a second step does not.

The step is applied to v48's final columns rather than to the own half before blending,
because the metric reads the spread of the column that is actually submitted, and a step
taken inside one half of an average arrives at half strength. CYP2C9 and CYP2D6 are
untouched, so this upload measures the placement change alone. Weights stay at one half
everywhere, as in v48, so this does not confound with v49's per-isoform weights.
"""
import hashlib
import numpy as np, pandas as pd

S = 'submissions/'
K = lambda i: f'{i}_pIC50_direct_inhibition'
spread = lambda x, f: x.mean() + (x - x.mean()) * f
STEP = {'CYP1A2': 1.16 / 1.08, 'CYP3A4': 0.85 / 0.92}

v48 = pd.read_csv(S + 'submission_v48_candidate.csv')
v50 = v48.copy()
for i in ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4'):
    a = v48[K(i)].to_numpy(float)
    b = spread(a, STEP[i]) if i in STEP else a
    v50[K(i)] = b
    assert pd.Series(a).corr(pd.Series(b), method='spearman') > 0.9999
    print(f'{i}  factor {STEP.get(i, 1.0):.4f}  sd {a.std():.4f} -> {b.std():.4f}  '
          f'mean {a.mean():.4f} -> {b.mean():.4f}  max change {np.abs(a - b).max():.3f}')

assert np.isfinite(v50.iloc[:, 2:].to_numpy(float)).all()
assert list(v50.columns) == list(v48.columns)
v50.to_csv(S + 'submission_v50_candidate.csv', index=False)
print('\nrows', len(v50), 'sha256', hashlib.sha256(open(S + 'submission_v50_candidate.csv', 'rb').read()).hexdigest())
print('v48 measured 0.4301 (1A2 .4545, 2C9 .3585, 2D6 .5316, 3A4 .3760).')
print('No projection: the second step has no measurement behind it, and that is what this upload buys.')
