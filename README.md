# OpenADMET CYP Blind Challenge

An entry for the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge),
covering the direct-inhibition regression track and the time-dependent-inhibition
classification track.

The modelling idea is in [METHOD.md](METHOD.md). In short: the scored metric charges
nothing for a prediction that lands inside a compound's fitted credible interval, and
those intervals are about ten times wider for inactive compounds than for potent
ones. Everything here follows from training and deciding against that asymmetry
rather than against pIC50 accuracy.

## Status and contents

Regression entry `GiantWizard`. Latest scored upload: macro soft-threshold error **0.4202**, rank 17 of about 283 on the live board, 1 November 2026. The full method report is in [report/report.pdf](report/report.pdf) with its LaTeX source in `report/report.tex`, and the figures in its body describe the entry as it stood on 29 October rather than today.

### What in the submission is borrowed

The uploaded regression file is a weighted average of this project's own stack with [Scigantic's published submission R25](https://github.com/Scigantic/open-admet-challenge-cyp), taken unchanged from their MIT-0 repository. The weight on their predictions is 0.500 for CYP1A2, CYP2D6 and CYP3A4 and 0.400 for CYP2C9, so **47.5% of the submitted numbers are theirs**. This project's own models were uploaded alone as a measurement on 29 October and scored 0.4472 against the blend's 0.4335 the same day.

The affine placement correction, and the idea of recovering the blind test set's label distribution from leaderboard feedback, are team briford's (SuperCowPowers), published before this project arrived at them. Section 8 of [METHOD.md](METHOD.md) sets out what is ours and what is not.

- `report/` the method report (LaTeX source and compiled PDF).
- `src/` the pipeline and analysis scripts.
- `notebooks/` the Colab notebooks that trained the neural arms. They read prepared payload archives from Google Drive that are not part of this repository, so they document the recipes and cannot be rerun without rebuilding those archives from the challenge data.
- `METHOD.md` the modelling idea in short form.

The challenge data and the submission files are not redistributed here.

## Layout

```
src/features.py    ECFP4 counts and RDKit descriptors, on standardised structures
src/metric.py      the challenge's soft-threshold RAE, and a LightGBM objective of the same shape
src/decision.py    the per-compound ST-RAE decision layer
src/pipeline.py    scaffold-grouped cross-validation, one isoform per invocation
src/submit.py      refit on all labelled data, apply the decision layer, write the regression file
src/tdi.py         the TDI classification track
src/build_data.py  assemble the modelling table from the challenge release
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
