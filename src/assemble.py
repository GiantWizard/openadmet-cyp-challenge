"""Assemble a submission from a column spec, or verify the spec reproduces a built file.

  python3 -I src/assemble.py v52 a2,c1s,d1m,g0
  python3 -I src/assemble.py --verify

Prints each column's provenance, its board reading if it has one, and the macro projection
over the columns that do. A spec is the whole record of what a file is, so reverting a bad
reading means assembling a different spec rather than editing anything.
"""
import hashlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from columns import ISO, K, S, READ, FIT, col, digest

KNOWN = {'v48': 'a0,c0,d0,g0', 'v50': 'a1,c0,d0,g1', 'v51': 'a1,c1,d0,g0',
         'v52': 'a1h,c1s,d1m,g0', 'v53': 'a2x,c2s,d2m,g2u',
         'v54': 'a2x,c3s,d2x,g1x',
         'v53a': 'a2,c2s,d2m,g1u', 'v55': 'a2x,c2s,d3x,g2u'}

def build(name, spec, write=True):
    ids = spec.split(',')
    assert len(ids) == 4, 'spec is four ids, one per isoform in CYP1A2,CYP2C9,CYP2D6,CYP3A4 order'
    ref = pd.read_csv(S + 'submission_v48_candidate.csv')
    order = ref.Molecule_Name.tolist()
    out = ref.copy()
    vals, measured = [], 0
    for iso, cid in zip(ISO, ids):
        x = col(cid, iso, order)
        out[K(iso)] = x
        tag = 'board' if cid in READ else ('fitted' if cid in FIT else 'UNTESTED')
        v = READ.get(cid, FIT.get(cid))
        measured += cid in READ
        vals.append(v)
        print(f'  {iso} {cid:4} {digest(x)}  sd {x.std():.4f}  mean {x.mean():.4f}  '
              f'{tag:8} {"" if v is None else f"{v:.4f}"}')
    assert np.isfinite(out.iloc[:, 2:].to_numpy(float)).all()
    assert list(out.columns) == list(ref.columns) and len(out) == 750
    if write:
        import os
        path = S + f'submission_{name}_candidate.csv'
        if os.path.exists(path):
            raise SystemExit(f'{path} already exists; a built spec is never overwritten, '
                             f'because an upload cannot be matched to it afterwards. '
                             f'Use a new name.')
        out.to_csv(path, index=False)
        print(f'  -> {path}  sha256 {hashlib.sha256(open(path, "rb").read()).hexdigest()}')
    known = [v for v in vals if v is not None]
    if len(known) == 4:
        print(f'  projection {sum(known) / 4:.4f}  ({measured}/4 tabs from the board)')
    else:
        print(f'  no projection: {4 - len(known)} column(s) have no reading; '
              f'measured part sums to {sum(known):.4f} over {len(known)}')
    return out

if sys.argv[1] == '--verify':
    for name, spec in KNOWN.items():
        print(f'{name} = {spec}')
        got = build(name, spec, write=False)
        want = pd.read_csv(S + f'submission_{name}_candidate.csv')
        same = all(np.allclose(got[K(i)], want[K(i)], atol=5e-7) for i in ISO)
        print(f'  reproduces {name}: {same}\n')
        assert same, name
else:
    print(f'{sys.argv[1]} = {sys.argv[2]}')
    build(sys.argv[1], sys.argv[2])
