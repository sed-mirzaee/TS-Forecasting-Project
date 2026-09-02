from src.config import HORIZONS, MATERIALS

FORECASTING_CONFIG = {
    "forecast_start": "2025-01-01",
    "forecast_end": "2025-12-31",
    "horizons": HORIZONS,
    "materials": MATERIALS,
    "selected_models": {

        # ========================================================
        # Horizon 10 - frozen
        # ========================================================
        10: {
            # M1: best CV model = SARIMA(1,1,2)(1,0,0)[5]
            "1": {
                "model": "arima/sarima",
                "params": {
                    "order": (1, 1, 2),
                    "seasonal_order": (1, 0, 0, 5),
                },
            },

            # M2: Ridge + DOW + Input mean(5) ending at lag 10
            "2": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10],
                    "rolling_windows": [10],
                    "use_day_of_week": True,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [(10, 5)],
                    "alpha": 0.0001,
                },
            },

            # M3: Ridge + school-holiday indicator; no Input
            "3": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10],
                    "rolling_windows": [10],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.01,
                },
            },

            # M4: recursive moving-average baseline, window defaults to horizon=10
            "4": {
                "model": "moving_average",
                "params": {},
            },

            # M5: Ridge + Input lag 10
            "5": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10],
                    "rolling_windows": [10],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [10],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M6: Ridge + Input mean(5) ending at lag 10
            "6": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10],
                    "rolling_windows": [10],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [(10, 5)],
                    "alpha": 0.0001,
                },
            },

            # M7: best CV model = ARIMA(0,1,1)
            "7": {
                "model": "arima/sarima",
                "params": {
                    "order": (0, 1, 1),
                    "seasonal_order": (0, 0, 0, 0),
                },
            },

            # M8: best CV model = ARIMA(0,1,1)
            "8": {
                "model": "arima/sarima",
                "params": {
                    "order": (0, 1, 1),
                    "seasonal_order": (0, 0, 0, 0),
                },
            },

            # M9: Ridge + school-holiday indicator + Input lag 10
            "9": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10],
                    "rolling_windows": [10],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [10],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M10: Ridge + DOW + Input lag 10
            "10": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10],
                    "rolling_windows": [10],
                    "use_day_of_week": True,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [10],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M11-M13: Naive was clearly best in CV
            "11": {
                "model": "naive",
                "params": {},
            },
            "12": {
                "model": "naive",
                "params": {},
            },
            "13": {
                "model": "naive",
                "params": {},
            },
        },

        # ========================================================
        # Horizon 22 - frozen
        # ========================================================
        22: {
            # M1: moving average baseline; window defaults to horizon=22
            "1": {
                "model": "moving_average",
                "params": {},
            },

            # M2: Ridge, roll22, no calendar, no Input
            "2": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22],
                    "rolling_windows": [22],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M3: Ridge + school holiday
            "3": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22],
                    "rolling_windows": [22],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M4: Ridge + school holiday
            "4": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22],
                    "rolling_windows": [22],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M5: Ridge + DOW
            "5": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22],
                    "rolling_windows": [22],
                    "use_day_of_week": True,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.001,
                },
            },

            # M6: Ridge + school holiday + Input mean(22) ending at lag 22
            "6": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22],
                    "rolling_windows": [22],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [(22, 22)],
                    "alpha": 0.0001,
                },
            },

            # M7: Ridge + public holiday + Input lag 22
            "7": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22],
                    "rolling_windows": [22],
                    "use_day_of_week": False,
                    "use_holiday": True,
                    "use_school_holiday": False,
                    "input_lags": [22],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M8: Ridge + school holiday
            "8": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22],
                    "rolling_windows": [22],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M9: Ridge + roll10 + school holiday; no Input
            "9": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22],
                    "rolling_windows": [10],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M10: Ridge + roll10 + DOW + Input lag 22
            "10": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22],
                    "rolling_windows": [10],
                    "use_day_of_week": True,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [22],
                    "input_rolling": [],
                    "alpha": 0.01,
                },
            },

            # M11: SARIMA(1,1,2)(1,0,0)[5]
            "11": {
                "model": "arima/sarima",
                "params": {
                    "order": (1, 1, 2),
                    "seasonal_order": (1, 0, 0, 5),
                },
            },

            # M12-M13: Naive
            "12": {"model": "naive", "params": {}},
            "13": {"model": "naive", "params": {}},
        },

        # ========================================================
        # Horizon 66 - frozen
        # ========================================================
        66: {
            # M1: SARIMA(1,1,2)(1,0,0)[5]
            "1": {
                "model": "arima/sarima",
                "params": {
                    "order": (1, 1, 2),
                    "seasonal_order": (1, 0, 0, 5),
                },
            },

            # M2: Ridge + roll22; no calendar, no Input
            "2": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66],
                    "rolling_windows": [22],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M3: Ridge + roll22 + school holiday + Input mean(5) at lag66
            "3": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66],
                    "rolling_windows": [22],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [(66, 5)],
                    "alpha": 0.01,
                },
            },

            # M4: Ridge + roll66 + school holiday + Input mean(5) at lag66
            "4": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66],
                    "rolling_windows": [66],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [(66, 5)],
                    "alpha": 0.01,
                },
            },

            # M5: Ridge + roll22 + DOW + Input lag66
            "5": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66],
                    "rolling_windows": [22],
                    "use_day_of_week": True,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [66],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M6: Ridge + roll66 + school holiday + Input lag66
            "6": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66],
                    "rolling_windows": [66],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [66],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M7: Ridge + roll66 + holiday; no Input
            "7": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66],
                    "rolling_windows": [66],
                    "use_day_of_week": False,
                    "use_holiday": True,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M8: Ridge + roll66 + school holiday + Input mean(5) at lag66
            "8": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66],
                    "rolling_windows": [66],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [(66, 5)],
                    "alpha": 0.0001,
                },
            },

            # M9: Ridge + no rolling + school holiday; no Input
            "9": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66],
                    "rolling_windows": [],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M10: Ridge + roll10 + DOW; no Input
            "10": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66],
                    "rolling_windows": [10],
                    "use_day_of_week": True,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.01,
                },
            },

            # M11: Ridge + roll66 + holiday; no Input
            "11": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66],
                    "rolling_windows": [66],
                    "use_day_of_week": False,
                    "use_holiday": True,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.001,
                },
            },

            # M12-M13: Naive
            "12": {"model": "naive", "params": {}},
            "13": {"model": "naive", "params": {}},
        },

        # ========================================================
        # Horizon 132 - frozen
        # ========================================================
        132: {
            # M1: ARIMA(1,1,2)
            "1": {
                "model": "arima/sarima",
                "params": {
                    "order": (1, 1, 2),
                    "seasonal_order": (0, 0, 0, 0),
                },
            },

            # M2: Ridge + roll66; no calendar, no Input
            "2": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66, 132],
                    "rolling_windows": [66],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M3: Ridge + roll132 + DOW; no Input
            "3": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66, 132],
                    "rolling_windows": [132],
                    "use_day_of_week": True,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M4: Ridge + roll132 + school holiday; no Input
            "4": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66, 132],
                    "rolling_windows": [132],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M5: Ridge + roll22 + DOW; no Input
            "5": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66, 132],
                    "rolling_windows": [22],
                    "use_day_of_week": True,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M6: moving average baseline; window defaults to horizon=132
            "6": {
                "model": "moving_average",
                "params": {},
            },

            # M7: Ridge + roll132; no calendar, no Input
            "7": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66, 132],
                    "rolling_windows": [132],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M8: Ridge + roll66 + school holiday; no Input
            "8": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66, 132],
                    "rolling_windows": [66],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.01,
                },
            },

            # M9: Ridge + roll22 + school holiday; no Input
            "9": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66, 132],
                    "rolling_windows": [22],
                    "use_day_of_week": False,
                    "use_holiday": False,
                    "use_school_holiday": True,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.0001,
                },
            },

            # M10: Ridge, no rolling + public holiday; no Input
            "10": {
                "model": "ridge",
                "params": {
                    "lags": [1, 2, 3, 4, 5, 10, 22, 66, 132],
                    "rolling_windows": [],
                    "use_day_of_week": False,
                    "use_holiday": True,
                    "use_school_holiday": False,
                    "input_lags": [],
                    "input_rolling": [],
                    "alpha": 0.01,
                },
            },

            # M11: moving average baseline; window defaults to horizon=132
            "11": {
                "model": "moving_average",
                "params": {},
            },

            # M12-M13: Naive
            "12": {"model": "naive", "params": {}},
            "13": {"model": "naive", "params": {}},
        },
    },

    "postprocessing": {
        10: "raw",
        22: "raw",
        66: "raw",
        132: "raw",
    },
}
