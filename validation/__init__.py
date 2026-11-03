"""Submission validators, vendored unmodified from the challenge tutorial repository.

Source: https://github.com/OpenADMET/CYP-Challenge-Tutorial (Apache-2.0),
validation/activity_validation.py and validation/tdi_validation.py.
Vendored so the pre-upload gate in this repository is the organisers' own code.
"""
from .activity_validation import validate_activity_submission
from .tdi_validation import validate_tdi_submission

__all__ = ["validate_activity_submission", "validate_tdi_submission"]
