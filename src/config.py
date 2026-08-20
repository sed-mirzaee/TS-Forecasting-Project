# ============================================================
# Project configuration
# ============================================================
# MATERIALS = [str(i) for i in range(1, 14)]  # 1, 2, .., 13
MATERIALS = [str(i) for i in range(1, 2)]

# Forecast horizons in days
WEEK_DAYS = 5 # Business Day
# HORIZONS = [10,  # Two weeks without weekends
#             22,  # One month
#             66,  # Three months
#             132] # Six months

HORIZONS = [10,  # Two weeks without weekends
             22]  # One month
            # 66,  # Three months
            # 132] # Six months

# Models included in the first version of the pipeline
# Currently is in forecasting

# The fraction of training set
INITIAL_RATIO = 0.80

OUTPUT_DIR = "outputs/tables"

