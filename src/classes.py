# ==========================================================================
# CLASSES
# Define custom classes in this file.
# ==========================================================================
#
#   * StandardScalerVAR: scales each process variable using one mean and
#     standard deviation shared by all of its lagged copies.
#
#   * NVARModel: one time series model (VAR, polynomial NVAR, or kNN NVAR).
#     It stores everything needed to forecast a batch from its initial
#     conditions: the fitted pipeline, the training residuals used as white
#     noise, and the training range used to clip predictions.
#
# ==========================================================================

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.constants import CV_SPLITS, FEATURES, MODEL_LABELS, PARAM_GRID_KNN, PARAM_GRID_POLY


class StandardScalerVAR:
    """
    Standard scaler for vector autoregressive (VAR / NVAR) feature matrices.

    The lagged feature matrix contains p repeated copies of the same process
    variables, e.g. with two variables and p = 2:

        [GLC(t-1), VCD(t-1), GLC(t-2), VCD(t-2)]

    sklearn's StandardScaler would scale each column separately. Here, every
    lagged copy of a variable shares the same mean and standard deviation, so
    GLC(t-1) and GLC(t-2) are scaled identically while GLC and VCD (which differ
    by ~5 orders of magnitude) are still scaled independently.
    """

    def __init__(self, p):
        self.p = p
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        self.n_samples_seen_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, x, y=None):
        # Number of process variables in each lag block.
        n_var_features = x.shape[1] // self.p
        assert n_var_features * self.p == x.shape[1]

        # Stack the lag blocks vertically so each column is one process variable.
        x_sf = np.concatenate([x[:, i * n_var_features:(i + 1) * n_var_features] for i in range(self.p)], axis=0)

        # One set of statistics per process variable, tiled across all lags.
        self.mean_ = np.tile(np.mean(x_sf, axis=0), (1, self.p))
        self.scale_ = np.tile(np.std(x_sf, axis=0), (1, self.p))
        self.var_ = np.tile(np.var(x_sf, axis=0), (1, self.p))

        self.n_samples_seen_ = x.shape[0]
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return self.scale_ * x + self.mean_

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def get_params(self, deep=True):
        return {"p": self.p}

    def set_params(self, **params):
        if "p" in params:
            self.p = params["p"]
        return self


class NVARModel:
    """A VAR or NVAR model of order p that forecasts all process variables together."""

    def __init__(self, model_type, p):
        self.model_type = model_type   # "var", "poly", or "knn"
        self.p = p                     # Number of lagged time points used as inputs.

        # Filled in by fit().
        self.model = None
        self.residuals = None          # 1-step-ahead training residuals, one column per variable.
        self.clip_min = None           # Smallest value of each variable in the training data.
        self.clip_max = None           # Largest value of each variable in the training data.
        self.best_params = {}

    def _make_pipeline(self):
        """Build the untrained pipeline and the hyperparameter grid for this model type."""
        if self.model_type == "var":
            return make_pipeline(StandardScalerVAR(p=self.p), LinearRegression()), None
        if self.model_type == "poly":
            pipeline = make_pipeline(StandardScalerVAR(p=self.p), PolynomialFeatures(include_bias=False),
                                     LinearRegression())
            return pipeline, PARAM_GRID_POLY
        if self.model_type == "knn":
            return make_pipeline(StandardScalerVAR(p=self.p), KNeighborsRegressor()), PARAM_GRID_KNN
        raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, x_lag, x_cur, b_cur):
        """Train on lagged values; tune hyperparameters with GridSearchCV that never splits a batch."""
        pipeline, param_grid = self._make_pipeline()

        if param_grid is None:
            # VAR has no hyperparameters to tune.
            self.model = pipeline.fit(x_lag, x_cur)
        else:
            # GroupKFold keeps every batch entirely in one fold, preventing leakage between folds.
            self.model = GridSearchCV(pipeline, param_grid=param_grid, cv=GroupKFold(n_splits=CV_SPLITS),
                                      scoring="neg_root_mean_squared_error")
            self.model.fit(x_lag, x_cur, groups=b_cur)
            self.best_params = self.model.best_params_

        # Residuals of the 1-step-ahead predictions on the training data.
        # These are sampled as white noise during the forecast simulations.
        self.residuals = self.model.predict(x_lag) - x_cur

        # Training range of each process variable (per variable, not one global range,
        # since VCD is ~10^6 cells/mL while mAb is ~1 mg/L).
        self.clip_min = np.min(x_cur, axis=0)
        self.clip_max = np.max(x_cur, axis=0)
        return self

    def simulate(self, x_initial, n_steps, n_simulations, rng):
        """
        Forecast batches recursively from their initial states, with white noise.

        x_initial : array of shape (n_conditions, n_features), one initial state per row.
        Returns an array of shape (n_conditions * n_simulations, n_steps, n_features).

        All simulations are run side by side: at each time step, the model predicts the next
        state for every simulation in a single call, instead of looping over simulations one
        at a time. The result is the same; it is just much faster.
        """
        # One row per simulation: each initial condition is repeated n_simulations times.
        state = np.repeat(x_initial, n_simulations, axis=0)
        history = [state]

        while len(history) < n_steps:
            # Lagged states x(t-1), x(t-2), ..., x(t-p). At the start of the forecast there may be
            # fewer than p past states, so the earliest available state is repeated.
            lags = [history[-i] if len(history) >= i else history[0] for i in range(1, self.p + 1)]
            x_lag = np.concatenate(lags, axis=1)

            # Baseline prediction, kept within the range the model was trained on.
            x_cur = np.clip(self.model.predict(x_lag), a_min=self.clip_min, a_max=self.clip_max)

            # White noise: each variable's noise is sampled from that variable's own residuals.
            white_noise = np.stack([rng.choice(self.residuals[:, j], size=x_cur.shape[0])
                                    for j in range(self.residuals.shape[1])], axis=1)

            # Concentrations and cell densities cannot be negative.
            history.append(np.clip(x_cur + white_noise, a_min=0, a_max=None))

        return np.stack(history, axis=1)

    @property
    def name(self):
        """Readable model name including the order and the tuned hyperparameter, e.g. 'NVAR (Poly) p=2, degree=2'."""
        label = f"{MODEL_LABELS[self.model_type]} p={self.p}"
        settings = [f"{key.split('__')[-1]}={value}" for key, value in self.best_params.items()
                    if not key.endswith("weights")]
        return label + (", " + ", ".join(settings) if settings else "")

    @staticmethod
    def feature_index(feature):
        return FEATURES.index(feature)