"""Compare nb24 runs. Usage: python score_nb24.py <dir holding results_nb24_<RUN>.json files> ...
Prints one row per (run, iso, variant) and the AUC / Spearman deltas against that run's own base
(or against run a's base when a run has no base arm), using the gate from STATUS.md."""
import json, sys
from pathlib import Path
rows = []
for d in sys.argv[1:]:
    for f in sorted(Path(d).glob("results_nb24_*.json")):
        run = f.stem.split("_")[-1]
        for r in json.load(open(f)):
            rows.append((run, r["iso"], r["variant"], r))
base = {(run, i): r for run, i, v, r in rows if v == "base"}
print(f"{'run':4s}{'iso':8s}{'var':8s}{'devRho':>8s}{'hoRho':>8s}{'aucDev':>8s}{'aucHo':>8s}"
      f"{'dAUCd':>8s}{'dAUCh':>8s}{'dRhoD':>8s}{'dRhoH':>8s}  blind<3.8")
for run, iso, v, r in rows:
    b = base.get((run, iso))
    if v == "base" or b is None:
        print(f"{run:4s}{iso:8s}{v:8s}{r['dev_spearman']:8.4f}{r['holdout_spearman']:8.4f}{r['auc_dev']:8.3f}{r['auc_holdout']:8.3f}")
        continue
    da, dh = r["auc_dev"] - b["auc_dev"], r["auc_holdout"] - b["auc_holdout"]
    sd, sh = r["dev_spearman"] - b["dev_spearman"], r["holdout_spearman"] - b["holdout_spearman"]
    gate = "pass" if da >= .03 and dh >= .03 and sd >= -.01 and sh >= -.01 else "fail"
    bl = r.get("blind_below_3_8", r.get("blind_share_below_3_8", ""))
    print(f"{run:4s}{iso:8s}{v:8s}{r['dev_spearman']:8.4f}{r['holdout_spearman']:8.4f}{r['auc_dev']:8.3f}{r['auc_holdout']:8.3f}"
          f"{da:+8.3f}{dh:+8.3f}{sd:+8.4f}{sh:+8.4f}  {bl}  gate {gate}")
