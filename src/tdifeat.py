"""Featurise the TDI training compounds into the clean cache."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from features import featurise  # noqa: E402
from pipeline import CACHE  # noqa: E402

tdi = pd.read_csv("data/cyp-challenge-TRAIN_TDI.csv")
x, _ = featurise(tdi["SMILES"].tolist())
np.save(CACHE / "X_tdi_train.npy", x.astype(np.float32))
print("wrote X_tdi_train.npy", x.shape)
