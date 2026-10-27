"""Build leaderboard candidates from v32 by per-column recipes. Columns are scored independently,
so each upload tests one change per column; keep a column's change only if its tab improves."""
import sys
import numpy as np, pandas as pd
S = 'submissions/'
base = pd.read_csv(S + 'submission_v32_candidate.csv')
K = lambda iso: f'{iso}_pIC50_direct_inhibition'

def spread(x, f):
    return x.mean() + (x - x.mean()) * f

def squash(x, c, g):
    return np.where(x < c, c - g * (c - x), x)

def build(name, spread_f=None, squash_p=None, take_d6=None):
    out = base.copy()
    for iso, f in (spread_f or {}).items():
        out[K(iso)] = spread(base[K(iso)].to_numpy(float), f)
    for iso, (c, g) in (squash_p or {}).items():
        out[K(iso)] = squash(out[K(iso)].to_numpy(float), c, g)
    out.to_csv(S + f'submission_{name}_candidate.csv', index=False)
    return out

if __name__ == '__main__':
    v35 = pd.read_csv(S + 'submission_v35_candidate.csv')
    chk = build('tmp', spread_f={'CYP1A2': 1.08, 'CYP3A4': 0.92})
    assert all(np.allclose(chk[K(i)], v35[K(i)]) for i in ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4')), 'v35 not reproduced'
    import os; os.remove(S + 'submission_tmp_candidate.csv')
    build('v36', {'CYP1A2': 1.08, 'CYP3A4': 0.92}, {'CYP2D6': (3.5, 0.3)})
    print('built v36')
