"""Placement experiments on weak-compound mass. Each column is scored independently on the board, so one file tests
one setting per column. Floor map: x < c goes to c - g*(c - x), which keeps the ordering. Bases: v38 for CYP1A2,
CYP2C9, CYP3A4; v35 for CYP2D6 (v38's CYP2D6 is v35's with cut 3.5, gamma 0.3 already applied)."""
import sys, numpy as np, pandas as pd
S = 'submissions/'
K = lambda i: f'{i}_pIC50_direct_inhibition'
squash = lambda x, c, g: np.where(x < c, c - g * (c - x), x)

def build(name, params):
    v38, v35 = pd.read_csv(S + 'submission_v38_candidate.csv'), pd.read_csv(S + 'submission_v35_candidate.csv')
    out = v38.copy()
    out[K('CYP2D6')] = v35[K('CYP2D6')]
    for iso, p in params.items():
        if p:
            out[K(iso)] = squash(out[K(iso)].to_numpy(float), *p)
    for iso in params:
        a, b = out[K(iso)].to_numpy(float), v38[K(iso)].to_numpy(float)
        assert pd.Series(a).corr(pd.Series(b), method='spearman') > 0.995
    out.to_csv(S + f'submission_{name}_candidate.csv', index=False)
    return out

if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'v39'
    out = build(name, {'v39': {'CYP1A2': (3.6, 0.3), 'CYP2C9': (3.6, 0.3), 'CYP2D6': (3.5, 0.15), 'CYP3A4': (3.6, 0.3)},
                       'v40': {'CYP1A2': None, 'CYP2C9': None, 'CYP2D6': (3.5, 0.5), 'CYP3A4': None}}[name])
    v38 = pd.read_csv(S + 'submission_v38_candidate.csv')
    for i in ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4'):
        a, b = out[K(i)].to_numpy(float), v38[K(i)].to_numpy(float)
        print(i, 'share moved', round((np.abs(a - b) > 1e-9).mean(), 2), 'mean', round(a.mean(), 3), 'min', round(a.min(), 2), 'spearman vs v38', round(pd.Series(a).corr(pd.Series(b), method='spearman'), 4))
