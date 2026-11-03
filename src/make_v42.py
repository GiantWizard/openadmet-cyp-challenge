"""v42: equal-weight average of our v40 columns with Scigantic's public R25 submission (MIT-0,
github.com/Scigantic/open-admet-challenge-cyp, data/submissions/R25.csv). Weight 0.5 on every isoform, fixed
before looking at the board. Development check on shared out-of-fold rows: every isoform improves at 0.5 in 4 or 5
of 5 folds (their OOF is rebuilt with their in-sample fitted combiner weights, so it is somewhat optimistic)."""
import numpy as np, pandas as pd
S = 'submissions/'
K = lambda i: f'{i}_pIC50_direct_inhibition'
ours = pd.read_csv(S + 'submission_v40_candidate.csv')
sg = pd.read_csv('references/scigantic/R25.csv').set_index('Molecule_Name').loc[ours.Molecule_Name].reset_index()
assert (sg.Molecule_Name.values == ours.Molecule_Name.values).all() and len(ours) == 750
out = ours.copy()
for i in ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4'):
    a, b = ours[K(i)].to_numpy(float), sg[K(i)].to_numpy(float)
    out[K(i)] = .5 * a + .5 * b
    print(i, 'ours mean/sd', round(a.mean(), 3), round(a.std(), 3), 'R25', round(b.mean(), 3), round(b.std(), 3), 'spearman', round(pd.Series(a).corr(pd.Series(b), method='spearman'), 3))
assert np.isfinite(out.iloc[:, 2:].to_numpy(float)).all()
out.to_csv(S + 'submission_v42_candidate.csv', index=False)
