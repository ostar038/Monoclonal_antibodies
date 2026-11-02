# ==========================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# ==========================================================================

from pathlib import Path

# Folder locations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
INITIAL_CONDITIONS_PATH = DATASETS_DIR / "initial_conditions.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

# Column names
COL_BATCH = "Batch ID"
COL_TIME = "Time [h]"

# Process variables forecast together by the model, in the order they appear in each batch file.
FEATURES = ["Glucose [mM]", "Glutamine [mM]", "Lactate [mM]", "Ammonia [mM]",
            "VCD [cells/mL]", "TCD [cells/mL]", "mAb [mg/L]"]
TARGET_MAB = "mAb [mg/L]"

# Initial-condition columns in initial_conditions.csv
IC_GLUCOSE = "Glucose [mM]"
IC_GLUTAMINE = "Glutamine [mM]"
IC_VCD = "VCD [cells/mL]"

# Forecast horizon.
# Every batch file covers 0-672 h, but mAb levels off within the first week and the most
# profitable harvest times in the data are all well below 336 h (14 days). Forecasting only
# this window reduces accumulated forecast error and computation time.
HORIZON_H = 336           # The most profitable harvest is always in the first week.

# Settings
RANDOM_STATE = 0
PREDICTION_INTERVAL = 0.95
N_SIMULATIONS = 1000      # Simulations per forecast for the final optimization.
N_SIMULATIONS_CV = 200    # Simulations per forecast during model comparison (keeps run time reasonable).
CV_SPLITS = 3             # GroupKFold splits used inside GridSearchCV.

# Models compared with leave-one-batch-out cross-validation
MODEL_TYPES = ["var", "poly", "knn"]
MODEL_LABELS = {"var": "VAR", "poly": "NVAR (Poly)", "knn": "NVAR (kNN)"}
P_VALUES = [1, 2, 3]      # Number of lagged time points (model order).

# Hyperparameter grids (searched with GridSearchCV + GroupKFold on each training set).
# Degree 1 is not searched because it is identical to the VAR model.
PARAM_GRID_POLY = dict(polynomialfeatures__degree=[2, 3])
# The kNN range starts at 2 and is extended to 40 because smaller ranges chose k at the upper edge.
PARAM_GRID_KNN = dict(kneighborsregressor__n_neighbors=[i for i in range(2, 41, 2)],
                      kneighborsregressor__weights=["uniform"])

# Economics
COST_GLUCOSE = 10        # $/L per mM of initial glucose
COST_GLUTAMINE = 20      # $/L per mM of initial glutamine
COST_VCD = 0.0005        # $/L per cell/mL of initial VCD
VALUE_MAB = 1000         # $/L per mg/L of mAb at harvest
DOWNTIME_D = 1           # Days of downtime between batch runs

# Plotting
FIG_WIDTH = 14.39 / 2.54   # Single-panel figure size in inches (converted from cm).
FIG_HEIGHT = 12.09 / 2.54
DPI = 200

FONTSIZE_XY_LABEL = 14
FONTSIZE_TICKS = 12
FONTSIZE_LEGEND = 12
FONTSIZE_TEXT = 6

MODEL_COLORS = {"var": "tab:green", "poly": "tab:blue", "knn": "tab:orange"}