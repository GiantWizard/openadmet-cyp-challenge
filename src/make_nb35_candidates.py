"""CYP1A2 blend of the NB33 + nb35 seed average (weight .2, chosen by the frozen development gate) on v35 and v36."""
import sys, numpy as np, pandas as pd
nb33_dir, nb35_dir = sys.argv[1], sys.argv[2]
S = 'submissions/'
K = 'CYP1A2_pIC50_direct_inhibition'
R = np.load('references/v31_reference_three.npz')
raw, ref = R['CYP1A2_test_raw'], R['CYP1A2_test_ref']
j = list(np.load(f'{nb35_dir}/fold0/hold_test.npz')['targets']).index('cyp1a2_pic50_direct_inhibition')
new = np.mean([np.load(f'{d}/fold{k}/hold_test.npz')['test'][:, j] for d in (nb33_dir, nb35_dir) for k in range(5)], axis=0)
match = lambda p, r: r.mean() + (p - p.mean()) / p.std() * r.std()
blend = match(.8 * raw + .2 * new, ref)
spread = lambda x, f: x.mean() + (x - x.mean()) * f
print('corr new vs ref', np.corrcoef(new, ref)[0, 1])
for src, name in (('v35', 'v37'), ('v36', 'v38')):
    df = pd.read_csv(S + f'submission_{src}_candidate.csv')
    assert np.allclose(df[K], spread(ref, 1.08))
    df[K] = spread(blend, 1.08)
    df.to_csv(S + f'submission_{name}_candidate.csv', index=False)
    print(name, 'rank corr with', src, pd.Series(df[K]).corr(pd.read_csv(S + f'submission_{src}_candidate.csv')[K], method='spearman'))
