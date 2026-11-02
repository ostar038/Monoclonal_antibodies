# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================

# ============================================================================
#   Optimizing Monoclonal Antibody Production
# ============================================================================
#
#   * Time series models (VAR and nonlinear VAR) are trained on 30 batch runs
#     to forecast glucose, glutamine, lactate, ammonia, VCD, TCD and mAb from
#     the initial conditions alone.
#
#   * The models are compared with leave-one-batch-out cross-validation.
#
#   * The selected model forecasts every candidate initial condition with
#     white-noise simulations, giving the initial glucose, glutamine, VCD and
#     harvest time that maximize the space-time profit (STP), with 95%
#     prediction intervals.
#
#
# ============================================================================

import os

import numpy as np
import pandas as pd

from src.classes import NVARModel
from src.constants import (
    COL_BATCH, COL_TIME, FEATURES, FIGURES_DIR, IC_GLUCOSE, IC_GLUTAMINE, IC_VCD, MODEL_LABELS, MODEL_TYPES,
    N_SIMULATIONS, N_SIMULATIONS_CV, P_VALUES, RANDOM_STATE, TABLES_DIR, TARGET_MAB,
)
from src.functions import (
    cost, economics, forecast_times, generate_x_lag_cur_var, initial_state, leave_one_batch_out, load_data,
    measured_economics, optimize_initial_conditions, plot_batches, plot_cv_comparison, plot_cv_forecasts,
    plot_effect_of_initial_conditions, plot_harvest_time, plot_recommended_forecast, plot_white_noise,
    quantiles, summarize_cv,
)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    # ==========================
    #   1. Load and explore
    # ==========================
    df, df_ic, batch_ids = load_data()
    dt_h = np.diff(np.unique(df.loc[:, COL_TIME]))[0]       # Sampling interval (8 h)
    n_steps = np.unique(df.loc[:, COL_TIME]).shape[0]       # Time points per forecast
    print(f"\nLoaded {len(batch_ids)} batches, {n_steps} time points each (0-{df[COL_TIME].max():.0f} h).")

    df_ic.to_csv(os.path.join(TABLES_DIR, "table_01_initial_conditions.csv"), index=False)
    plot_batches(df, "figure_01_measured_batches.png")

    # Best STP measured in each batch (a reference point for the model results).
    df_measured = measured_economics(df, df_ic)
    df_measured.to_csv(os.path.join(TABLES_DIR, "table_02_measured_stp.csv"), index=False)
    print("\nBest measured STP (top 5 batches):\n")
    print(df_measured.head(5).round(2).to_string(index=False))

    # ==========================
    #   2. Compare models with leave-one-batch-out cross-validation
    # ==========================
    records = []
    all_forecasts = {}
    for model_type in MODEL_TYPES:
        for p in P_VALUES:
            print(f"\nLeave-one-batch-out: {MODEL_LABELS[model_type]} p={p}")
            records_m, forecasts = leave_one_batch_out(model_type, p, df, df_ic, batch_ids, N_SIMULATIONS_CV, rng)
            records += records_m
            all_forecasts[(model_type, p)] = forecasts

    df_cv = pd.DataFrame.from_records(records)
    df_cv.to_csv(os.path.join(TABLES_DIR, "table_03_cv_per_batch.csv"), index=False)
    df_summary = summarize_cv(df_cv)
    df_summary.to_csv(os.path.join(TABLES_DIR, "table_04_cv_summary.csv"), index=False)
    print("\nModel comparison (mAb forecast from initial conditions only):\n")
    print(df_summary.round(3).to_string(index=False))
    plot_cv_comparison(df_summary, "figure_02_cv_rmse_vs_p.png")

    # Select the model with the lowest mean forecast RMSE.
    best_row = df_summary.loc[df_summary.loc[:, "RMSE Mean [mg/L]"].idxmin()]
    best_type = {label: key for key, label in MODEL_LABELS.items()}[best_row["Model"]]
    best_p = int(best_row["p"])
    print(f"\nSelected model: {best_row['Model']} p={best_p} "
          f"(RMSE {best_row['RMSE Mean [mg/L]']:.3f} +/- {best_row['RMSE SD [mg/L]']:.3f} mg/L)")

    for model_type in MODEL_TYPES:
        # Forecast figure for the selected model and for the best p of each alternative.
        df_m = df_summary.loc[df_summary.loc[:, "Model"] == MODEL_LABELS[model_type]]
        p = int(df_m.loc[df_m.loc[:, "RMSE Mean [mg/L]"].idxmin(), "p"])
        plot_cv_forecasts(all_forecasts[(model_type, p)], df_ic, f"{MODEL_LABELS[model_type]} p={p}",
                          f"figure_03_cv_forecasts_{model_type}_p{p:02d}.png")

    # ==========================
    #   3. Train the selected model on all 30 batches and check the white noise
    # ==========================
    x_lag, x_cur, _, b_cur = generate_x_lag_cur_var(p=best_p, batch_ids=batch_ids, df=df, features=FEATURES)
    model = NVARModel(best_type, best_p).fit(x_lag, x_cur, b_cur)
    print(f"\nFinal model trained on all batches: {model.name}")

    plot_white_noise(model, df, batch_ids, TARGET_MAB, "figure_04_white_noise_mab.png")

    # ==========================
    #   4. Optimize the initial conditions and harvest time
    # ==========================
    print(f"\nForecasting all candidate initial conditions ({N_SIMULATIONS} simulations each)...")
    df_opt, _ = optimize_initial_conditions(model, df_ic, dt_h, n_steps, N_SIMULATIONS, rng)
    df_opt = df_opt.sort_values("STP Median [$/(L*d)]", ascending=False).reset_index(drop=True)
    df_opt.to_csv(os.path.join(TABLES_DIR, "table_05_optimization_all_conditions.csv"), index=False)

    best = df_opt.iloc[0]
    print("\nTop 5 initial conditions by median STP:\n")
    print(df_opt.head(5).loc[:, [IC_GLUCOSE, IC_GLUTAMINE, IC_VCD, "Harvest Time [h]",
                                 "STP Median [$/(L*d)]", "STY Median [mg/(L*d)]"]].round(2).to_string(index=False))

    # ==========================
    #   5. Recommended conditions: final forecast and prediction intervals
    # ==========================
    # A fresh set of simulations for the recommended conditions, so its PIs are not the same
    # simulations that were used to pick it.
    g, q, vcd = best[IC_GLUCOSE], best[IC_GLUTAMINE], best[IC_VCD]
    t = forecast_times(n_steps, dt_h)
    x_sim = model.simulate(initial_state(g, q, vcd), n_steps=n_steps, n_simulations=N_SIMULATIONS, rng=rng)
    value, _, stp, sty = economics(x_sim[:, :, FEATURES.index(TARGET_MAB)], g, q, vcd, t)

    # Harvest time chosen by the optimization. (The STP curve is flat near its peak, so re-picking it
    # from the new simulations could shift it by one sampling interval for no real reason.)
    j = int(np.argmin(np.abs(t - best["Harvest Time [h]"])))
    value_pi, stp_pi, sty_pi = quantiles(value[:, j]), quantiles(stp[:, j]), quantiles(sty[:, j])
    mab_pi = quantiles(x_sim[:, j, FEATURES.index(TARGET_MAB)])

    recommendation = {
        IC_GLUCOSE: g, IC_GLUTAMINE: q, IC_VCD: vcd,
        "Harvest Time [h]": t[j], "Harvest Time [d]": t[j] / 24,
        "Cost [$/L]": cost(g, q, vcd),
        "mAb Median [mg/L]": mab_pi[1], "mAb Lower [mg/L]": mab_pi[0], "mAb Upper [mg/L]": mab_pi[2],
        "Value Median [$/L]": value_pi[1], "Value Lower [$/L]": value_pi[0], "Value Upper [$/L]": value_pi[2],
        "STP Median [$/(L*d)]": stp_pi[1], "STP Lower [$/(L*d)]": stp_pi[0], "STP Upper [$/(L*d)]": stp_pi[2],
        "STY Median [mg/(L*d)]": sty_pi[1], "STY Lower [mg/(L*d)]": sty_pi[0], "STY Upper [mg/(L*d)]": sty_pi[2],
    }
    pd.DataFrame([recommendation]).to_csv(os.path.join(TABLES_DIR, "table_06_recommendation.csv"), index=False)

    print("\nRecommended operating conditions:")
    print(f"  Glucose:      {g:.0f} mM")
    print(f"  Glutamine:    {q:.0f} mM")
    print(f"  VCD:          {vcd:.2e} cells/mL")
    print(f"  Harvest time: {t[j]:.0f} h ({t[j] / 24:.2f} d)")
    print(f"  Cost:         {recommendation['Cost [$/L]']:.2f} $/L")
    print(f"  Value:        {value_pi[1]:.1f} $/L  (95% PI {value_pi[0]:.1f} to {value_pi[2]:.1f})")
    print(f"  STP:          {stp_pi[1]:.1f} $/(L*d)  (95% PI {stp_pi[0]:.1f} to {stp_pi[2]:.1f})")
    print(f"  STY:          {sty_pi[1]:.3f} mg/(L*d)  (95% PI {sty_pi[0]:.3f} to {sty_pi[2]:.3f})")

    # Measured batches that were run at exactly the recommended conditions (if any), for comparison.
    mask = (df_ic[IC_GLUCOSE] == g) & (df_ic[IC_GLUTAMINE] == q) & (df_ic[IC_VCD] == vcd)
    batches_same = df_ic.loc[mask, COL_BATCH].to_list()
    measured_stp = []
    for batch_id in batches_same:
        m = (df[COL_BATCH] == batch_id) & (df[COL_TIME] > 0)
        t_m = df.loc[m, COL_TIME].to_numpy()
        _, _, stp_m, _ = economics(df.loc[m, TARGET_MAB].to_numpy(), g, q, vcd, t_m)
        measured_stp.append((t_m, stp_m))

    plot_recommended_forecast(x_sim, t, "figure_05_recommended_forecast.png", df, batches_same)
    plot_harvest_time(t, stp, t[j], "STP [$/(L*d)]", "figure_06_stp_vs_harvest_time.png", measured_stp)
    plot_harvest_time(t, sty, t[j], "STY [mg/(L*d)]", "figure_07_sty_vs_harvest_time.png")

    # ==========================
    #   6. Influence of each initial condition on STP and STY
    # ==========================
    for k, (column, label) in enumerate([(IC_GLUCOSE, IC_GLUCOSE), (IC_GLUTAMINE, IC_GLUTAMINE),
                                         (IC_VCD, IC_VCD)], start=8):
        name = column.split(" ")[0].lower()
        plot_effect_of_initial_conditions(df_opt, best, column, label, f"figure_{k:02d}_effect_{name}.png")

    print(f"\nFigures saved to '{FIGURES_DIR}' and tables saved to '{TABLES_DIR}'.\n")


if __name__ == "__main__":
    main()