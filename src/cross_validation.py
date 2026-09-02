import pandas as pd

# [Split Train / Test Initial Windows]
# [ 80 Train          |      20 Test ]
#   │
#   ▼
# [Loop Starts For All Folds]
#   │
#   │   initial fold [           |    ]
#   ├── 1. Fit Model (Train only on current fold's training window)
#   ├── 2. Predict (Generate forecasts for the current fold's test window)
#   ├── 3. Evaluate Error (Compute MSE/MAE for this fold)
#   └── 4. Expand Window (Shift training/testing boundaries forward)
#   │   last fold  [           |    ]
#   │   Shift fold [            |   ]
#   ▼
# [Loop Ends]
#   │
#   ▼
# [Average Errors Across All Folds]

def expanding_cv(
    df: pd.DataFrame,
    horizon: int,
    initial_ratio: float,
):
    n = len(df)
    initial_train_size = int(n * initial_ratio)

    train_end = initial_train_size

    while train_end < n:
        test_end = min(train_end + horizon, n)

        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end:test_end].copy()

        yield train, test

        train_end += horizon
