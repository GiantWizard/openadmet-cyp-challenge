# OpenADMET CYP Blind Challenge

Code and report for `GiantWizard` in the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge), covering direct-inhibition regression and time-dependent inhibition (TDI) classification.

The full technical write-up is in [report/report.pdf](report/report.pdf) (Markdown source in [report/report.md](report/report.md)), and [METHOD.md](METHOD.md) holds the longer experiment notes. On the regression track, the scored metric (MA-ST-RAE) assigns zero error inside a molecule's experimental credible interval. Inactive molecules below pIC50 4 carry intervals around 2 log units wide, while potent hits carry intervals around 0.2 log units wide. In addition, the 750 blind test molecules were chosen by expanding hits on CYP1A2, CYP2C9, and CYP3A4 while the CYP2D6 Echo-MS assay was still in setup. The pipeline here trains and calibrates around those two properties of the data.

## Score and attribution

Latest scored regression upload (v56, 2 November 2026): **0.4192** MA-ST-RAE.

The final regression file averages my five-arm stack (`v46`) with [Scigantic's published R25 submission](https://github.com/Scigantic/open-admet-challenge-cyp) (MIT-0 license). Weights on R25 are 0.500 for CYP1A2, CYP2D6, and CYP3A4, and 0.400 for CYP2C9 (**47.5% of the submitted values come from R25**). On 29 October, I uploaded both files on the same day to measure the split: `v46` scored 0.4472, and the `v47` average scored 0.4335.

The Murphy $R^2$ decomposition and the method of solving for blind test moments from leaderboard feedback come from team `briford` (SuperCowPowers). The credits table in section 6 of `report/report.pdf` lists the credit breakdown for each part of the submission.

## Repository layout

- `report/` : Markdown source (`report.md`) and compiled PDF (`report.pdf`).
- `METHOD.md` : experiment log covering the metric math and negative results.
- `src/` : local descriptor pipeline, LightGBM interval objective, and TDI classifier.
- `notebooks/` : Colab notebooks for the Chemprop, CheMeleon, and TabPFN arms. These notebooks read preprocessed feature archives from Google Drive (omitted from git due to file size), so they document the training configurations unless rebuilt from the raw CSVs.

Git tracks only the code and write-ups; raw challenge CSVs and generated submission files stay local.

```text
src/build_data.py  merge raw challenge CSVs into modeling tables
src/features.py    ECFP4 counts and RDKit 2D descriptors on standardized SMILES
src/metric.py      MA-ST-RAE scorer and custom LightGBM interval objective
src/decision.py    interval placement and spread scaling helpers
src/pipeline.py    5-fold Bemis-Murcko scaffold CV (one isoform per run)
src/submit.py      fit on full table, apply calibration, write regression CSV
src/tdi.py         TDI LightGBM pipeline and MCC threshold sweep
```

## Running it

The dataset is not vendored. Download the challenge release into `data/` first:

```bash
for f in cyp-challenge-TRAIN_inhibition.csv cyp-challenge-TRAIN_TDI.csv \
         cyp-challenge-single-concentration-TRAIN.csv cyp-challenge-TEST-BLINDED.csv; do
  curl -sL -o "data/$f" \
    "https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/resolve/main/$f"
done
```

Then:

```bash
pip install rdkit numpy pandas scikit-learn lightgbm pyarrow
python src/build_data.py                 # featurise; writes cache/
python src/pipeline.py aux               # stacked single-concentration features
for iso in CYP1A2 CYP2C9 CYP2D6 CYP3A4; do python src/pipeline.py $iso; done
python src/submit.py                     # -> submissions/submission_regression.csv
python src/tdi.py                        # -> submissions/submission_classification.csv
```

Each isoform runs as its own process and writes its own checkpoint under `cache/`,
so a failure costs one isoform rather than the whole run.

## Validation

Cross-validation is grouped by Bemis-Murcko scaffold: this dataset's test half was
built by purchasing catalogue analogs of the top hits, so folds that let an analog
cluster straddle the split would report a number that the blind set will not honour.
Where the decision layer is scored, the residual pool for a fold excludes that fold's
own compounds.

Submission files are checked with the challenge's own validators, vendored from the
[tutorial repository](https://github.com/OpenADMET/CYP-Challenge-Tutorial), so the
gate before upload is the code the platform runs.
