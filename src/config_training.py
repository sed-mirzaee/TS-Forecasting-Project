from src.config import MATERIALS, HORIZONS, INITIAL_RATIO

# version 01
# TRAINING_CONFIG = {
#     "run_name": "baselines_v01",
#     "models": {
#         "naive": {},
#         "moving_average": {},},
#         "horizons": HORIZONS,
#         "materials": MATERIALS,
#         "cv_method": "expanding",
#         "initial_ratio": INITIAL_RATIO,
# }

# version 02, for all arima/sarima candidates
SARIMAX_CANDIDATES = \
    {'1': [{'name': 'arima_112',
            'order': (1, 1, 2),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'sarima_112_100_5',
            'order': (1, 1, 2),
            'seasonal_order': (1, 0, 0, 5)}],
     '2': [{'name': 'arima_111',
            'order': (1, 1, 1),
            'seasonal_order': (0, 0, 0, 0)}],
     '3': [{'name': 'arima_111',
            'order': (1, 1, 1),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'sarima_111_002_5',
            'order': (1, 1, 1),
            'seasonal_order': (0, 0, 2, 5)}],
     '4': [{'name': 'arima_011',
            'order': (0, 1, 1),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'arima_012',
            'order': (0, 1, 2),
            'seasonal_order': (0, 0, 0, 0)}],
     '5': [{'name': 'arima_102',
            'order': (1, 0, 2),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'arima_202',
            'order': (2, 0, 2),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'arima_400',
            'order': (4, 0, 0),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'sarima_002_100_5',
            'order': (0, 0, 2),
            'seasonal_order': (1, 0, 0, 5)},
           {'name': 'sarima_100_100_5',
            'order': (1, 0, 0),
            'seasonal_order': (1, 0, 0, 5)}],
     '6': [{'name': 'arima_011',
            'order': (0, 1, 1),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'arima_211',
            'order': (2, 1, 1),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'sarima_011_200_5',
            'order': (0, 1, 1),
            'seasonal_order': (2, 0, 0, 5)}],
     '7': [{'name': 'arima_011',
            'order': (0, 1, 1),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'arima_012',
            'order': (0, 1, 2),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'arima_112',
            'order': (1, 1, 2),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'sarima_011_100_5',
            'order': (0, 1, 1),
            'seasonal_order': (1, 0, 0, 5)}],
     '8': [{'name': 'arima_011',
            'order': (0, 1, 1),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'arima_111',
            'order': (1, 1, 1),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'arima_113',
            'order': (1, 1, 3),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'sarima_011_001_5',
            'order': (0, 1, 1),
            'seasonal_order': (0, 0, 1, 5)}],
     '9': [{'name': 'arima_011',
            'order': (0, 1, 1),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'arima_013',
            'order': (0, 1, 3),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'arima_211',
            'order': (2, 1, 1),
            'seasonal_order': (0, 0, 0, 0)},
           {'name': 'arima_311',
            'order': (3, 1, 1),
            'seasonal_order': (0, 0, 0, 0)}],
     '10': [{'name': 'arima_012',
             'order': (0, 1, 2),
             'seasonal_order': (0, 0, 0, 0)},
            {'name': 'sarima_012_002_5',
             'order': (0, 1, 2),
             'seasonal_order': (0, 0, 2, 5)}],
     '11': [{'name': 'arima_111',
             'order': (1, 1, 1),
             'seasonal_order': (0, 0, 0, 0)},
            {'name': 'sarima_111_001_5',
             'order': (1, 1, 1),
             'seasonal_order': (0, 0, 1, 5)},
            {'name': 'sarima_112_001_5',
             'order': (1, 1, 2),
             'seasonal_order': (0, 0, 1, 5)},
            {'name': 'sarima_112_100_5',
             'order': (1, 1, 2),
             'seasonal_order': (1, 0, 0, 5)}],
     '12': [{'name': 'arima_201',
             'order': (2, 0, 1),
             'seasonal_order': (0, 0, 0, 0)},
            {'name': 'arima_400',
             'order': (4, 0, 0),
             'seasonal_order': (0, 0, 0, 0)},
            {'name': 'arima_500',
             'order': (5, 0, 0),
             'seasonal_order': (0, 0, 0, 0)}],
     '13': [{'name': 'arima_013',
             'order': (0, 1, 3),
             'seasonal_order': (0, 0, 0, 0)},
            {'name': 'arima_411',
             'order': (4, 1, 1),
             'seasonal_order': (0, 0, 0, 0)},
            {'name': 'sarima_113_002_5',
             'order': (1, 1, 3),
             'seasonal_order': (0, 0, 2, 5)},
            {'name': 'sarima_312_101_5',
             'order': (3, 1, 2),
             'seasonal_order': (1, 0, 1, 5)},
            {'name': 'sarima_412_101_5',
             'order': (4, 1, 2),
             'seasonal_order': (1, 0, 1, 5)}]}

# Keep candidates
# Best candidates for arima/sarima
# TRAINING_CONFIG = {
#     "run_name": "base_arima_v03",
#     "models": {
#         "naive": {},
#         "moving_average": {},
#         "arima/sarima": {
#             10: {
#                 "1": {"order": (1, 1, 2), "seasonal_order": (1, 0, 0, 5)},
#                 "2": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "3": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "4": {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "5": {"order": (1, 0, 2), "seasonal_order": (0, 0, 0, 0)},
#                 "6": {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "7": {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "8": {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "9": {"order": (2, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "10": {"order": (0, 1, 2), "seasonal_order": (0, 0, 0, 0)},
#                 "11": {"order": (1, 1, 2), "seasonal_order": (0, 0, 1, 5)},
#                 "12": {"order": (5, 0, 0), "seasonal_order": (0, 0, 0, 0)},
#                 "13": {"order": (1, 1, 3), "seasonal_order": (0, 0, 2, 5)},
#             },
#             22: {
#                 "1": {"order": (1, 1, 2), "seasonal_order": (0, 0, 0, 0)},
#                 "2": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "3": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "4": {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "5": {"order": (1, 0, 2), "seasonal_order": (0, 0, 0, 0)},
#                 "6": {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "7": {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "8": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "9": {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "10": {"order": (0, 1, 2), "seasonal_order": (0, 0, 0, 0)},
#                 "11": {"order": (1, 1, 2), "seasonal_order": (1, 0, 0, 5)},
#                 "12": {"order": (2, 0, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "13": {"order": (1, 1, 3), "seasonal_order": (0, 0, 2, 5)},
#             },
#             66: {
#                 "1": {"order": (1, 1, 2), "seasonal_order": (1, 0, 0, 5)},
#                 "2": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "3": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "4": {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "5": {"order": (1, 0, 2), "seasonal_order": (0, 0, 0, 0)},
#                 "6": {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "7": {"order": (0, 1, 1), "seasonal_order": (1, 0, 0, 5)},
#                 "8": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "9": {"order": (0, 1, 3), "seasonal_order": (0, 0, 0, 0)},
#                 "10": {"order": (0, 1, 2), "seasonal_order": (0, 0, 2, 5)},
#                 "11": {"order": (1, 1, 2), "seasonal_order": (1, 0, 0, 5)},
#                 "12": {"order": (4, 0, 0), "seasonal_order": (0, 0, 0, 0)},
#                 "13": {"order": (1, 1, 3), "seasonal_order": (0, 0, 2, 5)},
#             },
#             132: {
#                 "1": {"order": (1, 1, 2), "seasonal_order": (0, 0, 0, 0)},
#                 "2": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)} ,
#                 "3": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "4": {"order": (0, 1, 2), "seasonal_order": (0, 0, 0, 0)},
#                 "5": {"order": (1, 0, 2), "seasonal_order": (0, 0, 0, 0)},
#                 "6": {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "7": {"order": (0, 1, 1), "seasonal_order": (1, 0, 0, 5)},
#                 "8": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
#                 "9": {"order": (0, 1, 3), "seasonal_order": (0, 0, 0, 0)},
#                 "10": {"order": (0, 1, 2), "seasonal_order": (0, 0, 2, 5)},
#                 "11": {"order": (1, 1, 2), "seasonal_order": (1, 0, 0, 5)},
#                 "12": {"order": (4, 0, 0), "seasonal_order": (0, 0, 0, 0)},
#                 "13": {"order": (1, 1, 3), "seasonal_order": (0, 0, 2, 5)},
#             },
#         },
#     },
#     "horizons": HORIZONS,
#     "materials": MATERIALS,
#     "cv_method": "expanding",
#     "initial_ratio": INITIAL_RATIO,
# }

#each step create a new combination of parameters for horizon
TRAINING_HORIZON = 132
H132_RIDGE_FROZEN_PROFILES = [
    {
        "material": "1", "rolling_windows": [22], "alpha": 0.0001,
        "use_day_of_week": False, "use_holiday": False, "use_school_holiday": True,
    },
    {
        "material": "2", "rolling_windows": [66], "alpha": 0.0001,
        "use_day_of_week": False, "use_holiday": False, "use_school_holiday": False,
    },
    {
        "material": "3", "rolling_windows": [132], "alpha": 0.0001,
        "use_day_of_week": True, "use_holiday": False, "use_school_holiday": False,
    },
    {
        "material": "4", "rolling_windows": [132], "alpha": 0.0001,
        "use_day_of_week": False, "use_holiday": False, "use_school_holiday": True,
    },
    {
        "material": "5", "rolling_windows": [22], "alpha": 0.0001,
        "use_day_of_week": True, "use_holiday": False, "use_school_holiday": False,
    },
    {
        "material": "6", "rolling_windows": [22], "alpha": 0.0001,
        "use_day_of_week": False, "use_holiday": False, "use_school_holiday": False,
    },
    {
        "material": "7", "rolling_windows": [132], "alpha": 0.0001,
        "use_day_of_week": False, "use_holiday": False, "use_school_holiday": False,
    },
    {
        "material": "8", "rolling_windows": [66], "alpha": 0.01,
        "use_day_of_week": False, "use_holiday": False, "use_school_holiday": True,
    },
    {
        "material": "9", "rolling_windows": [22], "alpha": 0.0001,
        "use_day_of_week": False, "use_holiday": False, "use_school_holiday": True,
    },
    {
        "material": "10", "rolling_windows": [], "alpha": 0.01,
        "use_day_of_week": False, "use_holiday": True, "use_school_holiday": False,
    },
    {
        "material": "11", "rolling_windows": [66], "alpha": 0.0001,
        "use_day_of_week": False, "use_holiday": True, "use_school_holiday": False,
    },
    {
        "material": "12", "rolling_windows": [132], "alpha": 0.0001,
        "use_day_of_week": False, "use_holiday": True, "use_school_holiday": False,
    },
    {
        "material": "13",
        "lags": [1, 5, 10, 22, 66, 132],
        "rolling_windows": [],
        "alpha": 0.01,
        "use_day_of_week": False, "use_holiday": False, "use_school_holiday": True,
    },
]

H132_INPUT_VARIANTS = [
    {
        "name": "input_lag132",
        "params": {
            "input_lags": [132],
            "input_rolling": [],
        },
    },
    {
        "name": "input_mean5_lag132",
        "params": {
            "input_lags": [],
            "input_rolling": [(132, 5)],
        },
    },
    {
        "name": "input_mean132_lag132",
        "params": {
            "input_lags": [],
            "input_rolling": [(132, 132)],
        },
    },
]


def _build_h132_input_candidates():
    candidates = []

    for profile in H132_RIDGE_FROZEN_PROFILES:
        material = profile["material"]
        rolling_windows = profile["rolling_windows"]
        alpha = profile["alpha"]
        lags = profile.get("lags", [1, 2, 3, 4, 5, 10, 22, 66, 132])

        structure_name = (
            f"roll{rolling_windows[0]}"
            if rolling_windows
            else "no_roll"
        )

        calendar_parts = []
        if profile["use_day_of_week"]:
            calendar_parts.append("dow")
        if profile["use_holiday"]:
            calendar_parts.append("holiday")
        if profile["use_school_holiday"]:
            calendar_parts.append("schoolholiday")
        calendar_name = "_".join(calendar_parts)

        for variant in H132_INPUT_VARIANTS:
            name_parts = [
                f"ridge_h132_m{material}",
                structure_name,
                f"alpha_{alpha}",
            ]
            if calendar_name:
                name_parts.append(calendar_name)
            name_parts.append(variant["name"])

            candidates.append(
                {
                    "name": "_".join(name_parts),
                    "materials": [material],
                    "params": {
                        "lags": lags,
                        "rolling_windows": rolling_windows,
                        "alpha": alpha,
                        "use_day_of_week": profile["use_day_of_week"],
                        "use_holiday": profile["use_holiday"],
                        "use_school_holiday": profile["use_school_holiday"],
                        **variant["params"],
                    },
                }
            )

    return candidates


H132_RIDGE_INPUT_CANDIDATES = _build_h132_input_candidates()

TRAINING_CONFIG = {
    "run_name": "h132_ridge_input_ablation_v04",
    "initial_ratio": INITIAL_RATIO,
    "horizons": HORIZONS,
    "materials": MATERIALS,
    "models": {
        "ridge": H132_RIDGE_INPUT_CANDIDATES,
    },
}
