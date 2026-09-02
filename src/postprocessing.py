# ============================================================
# Forecast post-processing methods
# ============================================================

import numpy as np
import pandas as pd

def align_prediction_variants(predictions_raw: pd.DataFrame,) -> dict[str, pd.DataFrame]:

    # Variant 1: unchanged predictions
    raw = predictions_raw.copy()

    # Variant 2: independently restrict every fraction to [0, 1]
    clipped = predictions_raw.clip(lower=0.0, upper=1.0,)

    # I prefer this one, because the structure of prediction is near real data but if there is bad prediction in one material, influence more on other models
    # Variant 3: remove negatives and normalize the 13 fractions
    nonnegative = predictions_raw.clip(lower=0.0)
    row_sums = nonnegative.sum(axis=1)

    # check when all values are near 0, no change need
    zero_sum_mask = np.isclose(row_sums, 0.0)
    nonzero_sum_mask = ~zero_sum_mask

    normalized = nonnegative.copy()

    # normalize just non-zero rows
    normalized.loc[nonzero_sum_mask] = (nonnegative.loc[nonzero_sum_mask].div(row_sums.loc[nonzero_sum_mask], axis=0))

    return {
        "raw": raw,
        "clipped": clipped,
        "normalized": normalized,
    }