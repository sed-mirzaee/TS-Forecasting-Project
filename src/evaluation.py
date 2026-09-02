# ============================================================
# Evaluation methods
# ============================================================

import numpy as np
from src.config import SMAPE_EPS

# T: length of horizon
# SMAPEⱼₖ = 2/T Σᵢ |yᵢⱼ - ŷᵢⱼ|/max(ε, |yᵢⱼ| + |ŷᵢⱼ|)  , ε = 1/1000
# Missing values are removed from evaluation

def smape(y_true, y_pred) -> float:
    T = len(y_true)

    denominator = np.maximum(
        SMAPE_EPS,
        np.abs(y_true) + np.abs(y_pred)
    )

    return (
        2 / T
        * np.sum(
            np.abs(y_true - y_pred) / denominator
        )
    )