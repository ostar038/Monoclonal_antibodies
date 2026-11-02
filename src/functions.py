# ==========================================================================
# FUNCTIONS
# ==========================================================================
#
#   1. Helpers: progress bar and lagged-value dataset.
#   2. Data: load the batch files and the initial conditions.
#   3. Economics: cost, value, profit, space-time profit (STP), space-time yield (STY).
#   4. Model comparison: leave-one-batch-out cross-validation of forecasts.
#   5. Optimization: forecast every candidate initial condition and find the best harvest time.
#   6. Figures.
#
# ==========================================================================

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy import stats
from sklearn.metrics import root_mean_squared_error

from src.classes import NVARModel
from src.constants import (
    COL_BATCH, COL_TIME, COST_GLUCOSE, COST_GLUTAMINE, COST_VCD, DATASETS_DIR, DOWNTIME_D, DPI, FEATURES,
    FIG_HEIGHT, FIG_WIDTH, FIGURES_DIR, FONTSIZE_LEGEND, FONTSIZE_TEXT, FONTSIZE_TICKS, FONTSIZE_XY_LABEL,
    HORIZON_H, IC_GLUCOSE, IC_GLUTAMINE, IC_VCD, INITIAL_CONDITIONS_PATH, MODEL_COLORS, MODEL_LABELS,
    PREDICTION_INTERVAL, TARGET_MAB, VALUE_MAB,
)

ALPHA = 1 - PREDICTION_INTERVAL
IDX_MAB = FEATURES.index(TARGET_MAB)


# ==========================================================================
#   1. Helpers
# ==========================================================================

def progress_bar(c_iter, n_iters, n_updates=1000, length_bar=20):
    """Print a progress bar that updates on the same line."""
    update_every = max(1, round(n_iters / n_updates))
    if c_iter % update_every == 0:
        progress = c_iter / n_iters
        length_fill = round(length_bar * progress)
        length_void = length_bar - length_fill
        sys.stdout.write(f"\r{progress:6.0%} [" + length_fill * "=" + length_void * " " + "]")


def generate_x_lag_cur_var(p, batch_ids, df, features, col_batch=COL_BATCH, col_time=COL_TIME):
    """
    Build the lagged-value dataset for a VAR / NVAR model of order p.

    Each row of x_lag holds the p previous states of all process variables,
    [x(t-1), x(t-2), ..., x(t-p)], and the same row of x_cur holds the current state x(t).
    Each batch is processed separately so lagged values never cross from one batch to another.

    Returns x_lag, x_cur, t_cur (time of each x_cur row) and b_cur (batch of each x_cur row).
    """
    x_lag = []
    x_cur = []
    t_cur = []
    b_cur = []

    for batch_id in batch_ids:
        mask = df.loc[:, col_batch] == batch_id
        x_batch = df.loc[mask, features].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()
        b_batch = df.loc[mask, col_batch].to_numpy()

        # Current observations x(t). The first p points have no complete history, so they are skipped.
        current_values = x_batch[p:]
        current_times = t_batch[p:]
        current_batches = b_batch[p:]

        # Lagged states x(t-1), x(t-2), ..., x(t-p), placed side by side.
        lagged_values = []
        for i in range(1, p + 1):
            lagged_values.append(x_batch[p - i:-i])
        lagged_values = np.concatenate(lagged_values, axis=1)

        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)
        b_cur.append(current_batches)

    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)
    b_cur = np.concatenate(b_cur, axis=0)

    return x_lag, x_cur, t_cur, b_cur


# ==========================================================================
#   2. Data
# ==========================================================================

def load_data():
    """
    Load every batch_####.csv file in the datasets folder into one table.

    A "Batch ID" column is added (taken from the file name) and only the time points up to the
    forecast horizon are kept. The initial conditions are returned for those same batches only.
    """
    df_ic_all = pd.read_csv(INITIAL_CONDITIONS_PATH)

    frames = []
    for file_name in sorted(os.listdir(DATASETS_DIR)):
        if file_name.startswith("batch_") and file_name.endswith(".csv"):
            df_batch = pd.read_csv(os.path.join(DATASETS_DIR, file_name))
            batch_id = int(file_name.removeprefix("batch_").removesuffix(".csv"))
            df_batch.insert(0, COL_BATCH, batch_id)
            frames.append(df_batch)

    df = pd.concat(frames, ignore_index=True)
    df = df.loc[df.loc[:, COL_TIME] <= HORIZON_H].reset_index(drop=True)

    batch_ids = np.unique(df.loc[:, COL_BATCH])
    df_ic = df_ic_all.loc[df_ic_all.loc[:, COL_BATCH].isin(batch_ids)].reset_index(drop=True)
    return df, df_ic, batch_ids


def initial_state(glucose, glutamine, vcd):
    """
    Initial state of a batch, in the order of FEATURES.
    Lactate, ammonia and mAb start at 0, and every cell is alive at the start, so TCD = VCD.
    Accepts single values or arrays (one row per initial condition).
    """
    glucose, glutamine, vcd = np.atleast_1d(glucose, glutamine, vcd)
    zeros = np.zeros_like(glucose, dtype=float)
    return np.stack([glucose, glutamine, zeros, zeros, vcd, vcd, zeros], axis=1).astype(float)


def initial_state_of_batch(df_ic, batch_id):
    """Look up a batch's initial glucose, glutamine and VCD, and return its initial state."""
    row = df_ic.loc[df_ic.loc[:, COL_BATCH] == batch_id].iloc[0]
    return initial_state(row[IC_GLUCOSE], row[IC_GLUTAMINE], row[IC_VCD]), row


def forecast_times(n_steps, dt_h):
    """Time points of a forecast: 0, dt, 2*dt, ... [h]."""
    return np.arange(n_steps) * dt_h


def quantiles(x, axis=0):
    """Lower PI bound, median and upper PI bound of simulations along one axis."""
    return (np.quantile(x, q=ALPHA / 2, axis=axis),
            np.quantile(x, q=0.5, axis=axis),
            np.quantile(x, q=1 - ALPHA / 2, axis=axis))


# ==========================================================================
#   3. Economics
# ==========================================================================

def cost(glucose, glutamine, vcd):
    """Cost of the initial medium and inoculum [$/L]."""
    return COST_GLUCOSE * glucose + COST_GLUTAMINE * glutamine + COST_VCD * vcd


def economics(mab, glucose, glutamine, vcd, t_h):
    """
    Value, profit, space-time profit (STP) and space-time yield (STY) at harvest time t_h [h].

    STP [$/(L*d)] = (value - cost) / (harvest time [d] + downtime [d])
    STY [mg/(L*d)] = mAb / (harvest time [d] + downtime [d])
    Works element-wise on arrays, so it can be applied to every simulation and time at once.
    """
    cycle_time_d = t_h / 24 + DOWNTIME_D
    value = VALUE_MAB * mab
    profit = value - cost(glucose, glutamine, vcd)
    stp = profit / cycle_time_d
    sty = mab / cycle_time_d
    return value, profit, stp, sty


def measured_economics(df, df_ic):
    """Best measured STP of each batch, and the harvest time at which it occurs."""
    records = []
    for batch_id in np.unique(df.loc[:, COL_BATCH]):
        mask = df.loc[:, COL_BATCH] == batch_id
        t = df.loc[mask, COL_TIME].to_numpy()
        mab = df.loc[mask, TARGET_MAB].to_numpy()
        _, row = initial_state_of_batch(df_ic, batch_id)
        g, q, vcd = row[IC_GLUCOSE], row[IC_GLUTAMINE], row[IC_VCD]
        value, profit, stp, sty = economics(mab, g, q, vcd, t)
        stp[t == 0] = -np.inf   # Harvesting at time 0 is not a real option.
        i = np.argmax(stp)
        records.append({COL_BATCH: batch_id, IC_GLUCOSE: g, IC_GLUTAMINE: q, IC_VCD: vcd,
                        "Harvest Time [h]": t[i], "mAb [mg/L]": mab[i], "Cost [$/L]": cost(g, q, vcd),
                        "Value [$/L]": value[i], "STP [$/(L*d)]": stp[i], "STY [mg/(L*d)]": sty[i]})
    return pd.DataFrame.from_records(records).sort_values("STP [$/(L*d)]", ascending=False)


# ==========================================================================
#   4. Model comparison (leave-one-batch-out cross-validation)
# ==========================================================================

def leave_one_batch_out(model_type, p, df, df_ic, batch_ids, n_simulations, rng):
    """
    Leave-one-batch-out cross-validation of the full forecast.

    For each batch: train on all other batches, forecast the left-out batch from its initial
    conditions only (as would be done for a new, untested batch), and compare the median mAb
    forecast with the measured mAb.

    Returns one record per left-out batch and the mAb forecasts (lower PI, median, upper PI).
    """
    records = []
    forecasts = {}

    for i, batch_id_test in enumerate(batch_ids):
        batch_ids_train = np.array([b for b in batch_ids if b != batch_id_test])
        x_lag, x_cur, _, b_cur = generate_x_lag_cur_var(p=p, batch_ids=batch_ids_train, df=df, features=FEATURES)
        model = NVARModel(model_type, p).fit(x_lag, x_cur, b_cur)

        # Measured data of the test batch.
        mask = df.loc[:, COL_BATCH] == batch_id_test
        t_test = df.loc[mask, COL_TIME].to_numpy()
        x_test = df.loc[mask, FEATURES].to_numpy()

        # Forecast from the initial conditions only.
        x0, _ = initial_state_of_batch(df_ic, batch_id_test)
        x_sim = model.simulate(x0, n_steps=t_test.shape[0], n_simulations=n_simulations, rng=rng)
        mab_lower, mab_median, mab_upper = quantiles(x_sim[:, :, IDX_MAB], axis=0)

        mab_test = x_test[:, IDX_MAB]
        inside = (mab_test >= mab_lower) & (mab_test <= mab_upper)

        records.append({"Model": MODEL_LABELS[model_type], "p": p, COL_BATCH: batch_id_test,
                        "Hyperparameter": ", ".join(f"{k.split('__')[-1]}={v}" for k, v in model.best_params.items()
                                                    if not k.endswith("weights")),
                        "RMSE mAb [mg/L]": root_mean_squared_error(mab_test, mab_median),
                        "PI Coverage": np.mean(inside)})
        forecasts[batch_id_test] = (t_test, mab_test, mab_lower, mab_median, mab_upper)
        progress_bar(i + 1, len(batch_ids))

    sys.stdout.write("\n")
    return records, forecasts


def summarize_cv(df_cv):
    """Mean and standard deviation of the per-batch results for each model and p."""
    df_summary = (df_cv.groupby(["Model", "p"], sort=False)
                  .agg(**{"RMSE Mean [mg/L]": ("RMSE mAb [mg/L]", "mean"),
                          "RMSE SD [mg/L]": ("RMSE mAb [mg/L]", "std"),
                          "PI Coverage Mean": ("PI Coverage", "mean"),
                          "Most Common Hyperparameter": ("Hyperparameter", lambda s: s.mode().iloc[0])})
                  .reset_index())
    return df_summary


# ==========================================================================
#   5. Optimization
# ==========================================================================

def optimize_initial_conditions(model, df_ic, dt_h, n_steps, n_simulations, rng):
    """
    Forecast every combination of the tested glucose, glutamine and VCD levels.

    Only combinations of levels that appear in the training data are tested, so the search
    stays inside the range the model has seen (the model is not used to extrapolate).
    For each combination, the median STP is calculated at every possible harvest time; the
    harvest time with the highest median STP is kept, together with the 95% PIs at that time.

    Returns a table with one row per combination, and the median STP curve of every combination.
    """
    levels_g = np.unique(df_ic.loc[:, IC_GLUCOSE])
    levels_q = np.unique(df_ic.loc[:, IC_GLUTAMINE])
    levels_v = np.unique(df_ic.loc[:, IC_VCD])
    t = forecast_times(n_steps, dt_h)

    records = []
    stp_curves = []
    n_total = len(levels_g)

    # One glucose level at a time (49 combinations x n_simulations rows), to limit memory use.
    for i, g in enumerate(levels_g):
        q_grid, v_grid = np.meshgrid(levels_q, levels_v, indexing="ij")
        q_grid, v_grid = q_grid.ravel(), v_grid.ravel()
        g_grid = np.full_like(q_grid, g, dtype=float)

        x0 = initial_state(g_grid, q_grid, v_grid)
        x_sim = model.simulate(x0, n_steps=n_steps, n_simulations=n_simulations, rng=rng)

        # Shape: (combination, simulation, time)
        mab = x_sim[:, :, IDX_MAB].reshape(len(g_grid), n_simulations, n_steps)
        shape = (-1, 1, 1)
        value, _, stp, sty = economics(mab, g_grid.reshape(shape), q_grid.reshape(shape),
                                       v_grid.reshape(shape), t.reshape(1, 1, -1))

        stp_median = np.quantile(stp, q=0.5, axis=1)
        stp_median[:, 0] = -np.inf  # Harvesting at time 0 is not a real option.

        for k in range(len(g_grid)):
            j = np.argmax(stp_median[k])
            value_pi = quantiles(value[k, :, j])
            stp_pi = quantiles(stp[k, :, j])
            sty_pi = quantiles(sty[k, :, j])
            records.append({IC_GLUCOSE: g, IC_GLUTAMINE: q_grid[k], IC_VCD: v_grid[k],
                            "Harvest Time [h]": t[j], "Harvest Time [d]": t[j] / 24,
                            "Cost [$/L]": cost(g, q_grid[k], v_grid[k]),
                            "Value Median [$/L]": value_pi[1], "Value Lower [$/L]": value_pi[0],
                            "Value Upper [$/L]": value_pi[2],
                            "STP Median [$/(L*d)]": stp_pi[1], "STP Lower [$/(L*d)]": stp_pi[0],
                            "STP Upper [$/(L*d)]": stp_pi[2],
                            "STY Median [mg/(L*d)]": sty_pi[1], "STY Lower [mg/(L*d)]": sty_pi[0],
                            "STY Upper [mg/(L*d)]": sty_pi[2]})
            stp_curves.append(stp_median[k])

        progress_bar(i + 1, n_total)

    sys.stdout.write("\n")
    df_opt = pd.DataFrame.from_records(records)
    return df_opt, np.array(stp_curves)


# ==========================================================================
#   6. Figures
# ==========================================================================

def _style_axis(ax, x_label, y_label, legend=True):
    ax.set_xlabel(x_label, fontsize=FONTSIZE_XY_LABEL)
    ax.set_ylabel(y_label, fontsize=FONTSIZE_XY_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
    if legend:
        ax.legend(fontsize=FONTSIZE_LEGEND)


def plot_batches(df, file_name):
    """Measured profiles of every process variable for all batches (one subplot per variable)."""
    fig, axes = plt.subplots(2, 4, figsize=(29.21 / 2.54, 12.09 / 2.54), dpi=DPI, layout="constrained")
    axes = axes.ravel()
    for batch_id in np.unique(df.loc[:, COL_BATCH]):
        mask = df.loc[:, COL_BATCH] == batch_id
        for ax, feature in zip(axes, FEATURES):
            ax.plot(df.loc[mask, COL_TIME], df.loc[mask, feature], linewidth=0.8, alpha=0.7)
    for ax, feature in zip(axes, FEATURES):
        ax.set_xlabel(COL_TIME)
        ax.set_ylabel(feature)
        ax.xaxis.set_major_locator(MultipleLocator(96))
    axes[-1].axis("off")
    fig.savefig(os.path.join(FIGURES_DIR, file_name))
    plt.close(fig)


def plot_cv_comparison(df_summary, file_name):
    """Forecast RMSE of mAb versus model order p, one line per model type (error bars = 1 SD)."""
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, layout="constrained")
    for model_type, label in MODEL_LABELS.items():
        df_m = df_summary.loc[df_summary.loc[:, "Model"] == label]
        if df_m.empty:
            continue
        ax.errorbar(df_m.loc[:, "p"], df_m.loc[:, "RMSE Mean [mg/L]"], yerr=df_m.loc[:, "RMSE SD [mg/L]"],
                    label=label, color=MODEL_COLORS[model_type], marker="o", capsize=4)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    _style_axis(ax, "Model Order p", f"Forecast RMSE, {TARGET_MAB}")
    fig.savefig(os.path.join(FIGURES_DIR, file_name))
    plt.close(fig)


def plot_cv_forecasts(forecasts, df_ic, model_label, file_name):
    """Leave-one-batch-out mAb forecasts (median and 95% PI) against the measured test batch."""
    batch_ids = list(forecasts.keys())
    ncols = 6
    nrows = int(np.ceil(len(batch_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(29.21 / 2.54, 1.5 * 12.09 / 2.54), dpi=DPI,
                             layout="constrained", sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, batch_id in zip(axes, batch_ids):
        t_test, mab_test, lower, median, upper = forecasts[batch_id]
        _, row = initial_state_of_batch(df_ic, batch_id)
        ax.scatter(t_test, mab_test, label="Test", s=6, color="tab:orange", edgecolors="black", linewidth=0.3,
                   zorder=3)
        ax.plot(t_test, median, label=model_label, color="black", linewidth=1, zorder=4)
        ax.fill_between(t_test, lower, upper, label=f"{PREDICTION_INTERVAL:.0%} PI", color="tab:blue",
                        edgecolors="none", alpha=0.3, zorder=2)
        message = (f"Batch {batch_id}\nRMSE: {root_mean_squared_error(mab_test, median):.2f} mg/L\n"
                   f"GLC: {row[IC_GLUCOSE]:.0f} mM\nGLN: {row[IC_GLUTAMINE]:.0f} mM\n"
                   f"VCD: {row[IC_VCD] / 1e6:.1f}e6/mL")
        ax.text(x=0.02, y=0.96, s=message, va="top", ha="left", transform=ax.transAxes, fontname="monospace",
                fontsize=FONTSIZE_TEXT)
        ax.xaxis.set_major_locator(MultipleLocator(96))

    for i, ax in enumerate(axes):
        if i >= len(batch_ids):
            ax.axis("off")
        if i % ncols == 0:
            ax.set_ylabel(TARGET_MAB, fontsize=8)
        if i >= len(axes) - ncols:
            ax.set_xlabel(COL_TIME, fontsize=8)
    axes[ncols - 1].legend(loc="lower right", fontsize=6)
    fig.savefig(os.path.join(FIGURES_DIR, file_name))
    plt.close(fig)


def plot_white_noise(model, df, batch_ids, feature, file_name):
    """
    Residuals of the 1-step-ahead predictions versus time, with the mean and a 95% interval
    (Student's t) at each time point. White noise should be centred on zero at every time.
    """
    x_lag, x_cur, t_cur, _ = generate_x_lag_cur_var(p=model.p, batch_ids=batch_ids, df=df, features=FEATURES)
    idx = FEATURES.index(feature)
    res = model.model.predict(x_lag)[:, idx] - x_cur[:, idx]

    t_unique = np.unique(t_cur)
    res_mean, res_lower, res_upper = [], [], []
    for t in t_unique:
        res_t = res[t_cur == t]
        n_mean = np.mean(res_t)
        n_sd = np.std(res_t)
        pi_sd = stats.t.isf(q=ALPHA / 2, df=res_t.shape[0] - 1)
        res_mean.append(n_mean)
        res_lower.append(n_mean - pi_sd * n_sd)
        res_upper.append(n_mean + pi_sd * n_sd)

    y_max = max(np.max(np.abs(res)), np.max(np.abs(res_lower)), np.max(np.abs(res_upper)))

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, layout="constrained")
    ax.scatter(t_cur, res, label="Noise", color="tab:orange", s=12, edgecolors="black", linewidth=0.5,
               alpha=0.7, zorder=3)
    ax.plot(t_unique, res_mean, label="Mean", color="black", zorder=4)
    ax.fill_between(t_unique, res_lower, res_upper, label=f"{PREDICTION_INTERVAL:.0%} PI", color="tab:blue",
                    edgecolors="none", alpha=0.3, zorder=2)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5, zorder=1)
    ax.xaxis.set_major_locator(MultipleLocator(48))
    ax.set_ylim(bottom=-1.05 * y_max, top=1.05 * y_max)
    _style_axis(ax, COL_TIME, f"{feature} (Residuals)", legend=False)
    ax.legend(loc="upper left", fontsize=FONTSIZE_LEGEND)
    fig.savefig(os.path.join(FIGURES_DIR, file_name))
    plt.close(fig)


def plot_recommended_forecast(x_sim, t, file_name, df=None, batch_ids_measured=()):
    """Forecast of every process variable (median and 95% PI) for the recommended initial conditions."""
    fig, axes = plt.subplots(2, 4, figsize=(29.21 / 2.54, 12.09 / 2.54), dpi=DPI, layout="constrained")
    axes = axes.ravel()
    lower, median, upper = quantiles(x_sim, axis=0)

    for j, (ax, feature) in enumerate(zip(axes, FEATURES)):
        # Measured batches run at the same initial conditions, if any.
        for k, batch_id in enumerate(batch_ids_measured):
            mask = df.loc[:, COL_BATCH] == batch_id
            ax.scatter(df.loc[mask, COL_TIME], df.loc[mask, feature], s=6, color="tab:orange", edgecolors="black",
                       linewidth=0.3, zorder=3, label="Measured" if k == 0 else None)
        ax.plot(t, median[:, j], color="black", label="Median", zorder=4)
        ax.fill_between(t, lower[:, j], upper[:, j], color="tab:blue", edgecolors="none", alpha=0.3,
                        label=f"{PREDICTION_INTERVAL:.0%} PI", zorder=2)
        ax.set_xlabel(COL_TIME)
        ax.set_ylabel(feature)
        ax.xaxis.set_major_locator(MultipleLocator(96))
    axes[0].legend(fontsize=7)
    axes[-1].axis("off")
    fig.savefig(os.path.join(FIGURES_DIR, file_name))
    plt.close(fig)


def plot_harvest_time(t, values, best_t, y_label, file_name, measured=None):
    """
    STP or STY versus harvest time for the recommended conditions: median and 95% PI of the
    simulations, with the recommended harvest time marked.
    """
    lower, median, upper = quantiles(values, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, layout="constrained")
    if measured is not None:
        for k, (t_m, y_m) in enumerate(measured):
            ax.scatter(t_m, y_m, s=12, color="tab:orange", edgecolors="black", linewidth=0.5, zorder=3,
                       label="Measured" if k == 0 else None)
    ax.plot(t[1:], median[1:], color="black", label="Median", zorder=4)
    ax.fill_between(t[1:], lower[1:], upper[1:], color="tab:blue", edgecolors="none", alpha=0.3,
                    label=f"{PREDICTION_INTERVAL:.0%} PI", zorder=2)
    ax.axvline(best_t, color="black", linestyle="--", linewidth=1, label=f"Harvest: {best_t:.0f} h")
    ax.xaxis.set_major_locator(MultipleLocator(48))
    _style_axis(ax, "Harvest Time [h]", y_label.replace("$", r"\$"))
    fig.savefig(os.path.join(FIGURES_DIR, file_name))
    plt.close(fig)


def plot_effect_of_initial_conditions(df_opt, best, column, label, file_name):
    """
    Effect of each initial condition on the best achievable median STP (left) and STY (right).

    One factor is varied over its tested levels while the other two are held at their
    recommended values. The 95% PI at each level is shown as a band.
    """
    others = [c for c in [IC_GLUCOSE, IC_GLUTAMINE, IC_VCD] if c != column]
    mask = (df_opt.loc[:, others[0]] == best[others[0]]) & (df_opt.loc[:, others[1]] == best[others[1]])
    df_line = df_opt.loc[mask].sort_values(column)
    x = df_line.loc[:, column].to_numpy()
    if column == IC_VCD:
        x = x / 1e6
        label = "VCD [10$^6$ cells/mL]"

    fig, axes = plt.subplots(1, 2, figsize=(29.21 / 2.54, 12.09 / 2.54), dpi=DPI, layout="constrained")
    for ax, name, units in zip(axes, ["STP", "STY"], ["$/(L*d)", "mg/(L*d)"]):
        ax.plot(x, df_line.loc[:, f"{name} Median [{units}]"], color="black", marker="o", label="Median")
        ax.fill_between(x, df_line.loc[:, f"{name} Lower [{units}]"], df_line.loc[:, f"{name} Upper [{units}]"],
                        color="tab:blue", edgecolors="none", alpha=0.3, label=f"{PREDICTION_INTERVAL:.0%} PI")
        # r"\$" shows a dollar sign (a plain "$" starts a math expression in matplotlib).
        _style_axis(ax, label, f"{name} [{units}]".replace("$", r"\$"))
    fig.savefig(os.path.join(FIGURES_DIR, file_name))
    plt.close(fig)