# ============================================================
# Model-based data processing methods
# ============================================================

import numpy as np
import pandas as pd

def arima_imputation(train: pd.DataFrame, material: str) -> pd.Series:
    y = train[material].copy()

    y = y.interpolate(method="linear") # spline
    y = y.ffill().bfill()

    return y

"""
Build the history used for Ridge lag features.

Important:
- The original target values are NOT changed.
- Only past values are used to fill short gaps (forward fill).
- Long gaps remain missing.
- No backward fill is used, because that would use a later value
  to construct a feature for an earlier date.
"""

def ridge_history_series(
    train: pd.DataFrame,
    material: str,
    fill_limit: int = 3,
) -> pd.Series:

    y_raw = train[material].astype(float).copy()
    return y_raw.ffill(limit=fill_limit)
