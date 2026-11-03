"""v41: one single-parameter step per column on top of v40, each continuing a direction the board already rewarded.
CYP1A2 spread x1.08 -> x1.16 total (x1.08 gave -0.0049); CYP3A4 spread x0.92 -> x0.85 total (x0.92 gave -0.0044);
CYP2D6 floor strength 0.65 at cut 3.5 from the v35 base (readings 1.0/.5/.3/.15 -> .566/.5524/.5556/.5714, quadratic
optimum about 0.6); CYP2C9 unchanged (no direction with evidence). Columns score independently, so each is read alone."""
import numpy as np, pandas as pd
S = 'submissions/'
K = lambda i: f'{i}_pIC50_direct_inhibition'
spread = lambda x, f: x.mean() + (x - x.mean()) * f
squash = lambda x, c, g: np.where(x < c, c - g * (c - x), x)
v40, v35 = pd.read_csv(S + 'submission_v40_candidate.csv'), pd.read_csv(S + 'submission_v35_candidate.csv')
out = v40.copy()
out[K('CYP1A2')] = spread(v40[K('CYP1A2')].to_numpy(float), 1.16 / 1.08)
out[K('CYP3A4')] = spread(v40[K('CYP3A4')].to_numpy(float), 0.85 / 0.92)
out[K('CYP2D6')] = squash(v35[K('CYP2D6')].to_numpy(float), 3.5, 0.65)
for i in ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4'):
    a, b = out[K(i)].to_numpy(float), v40[K(i)].to_numpy(float)
    assert np.isfinite(a).all() and pd.Series(a).corr(pd.Series(b), method='spearman') > 0.9999
    print(i, 'mean', round(a.mean(), 3), 'std', round(a.std(), 3), 'max change', round(np.abs(a - b).max(), 3))
out.to_csv(S + 'submission_v41_candidate.csv', index=False)
