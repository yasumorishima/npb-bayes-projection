"""Bayesian wOBA/ERA projection for NPB foreign players using PyMC.

Step 2: Hierarchical Normal model with conversion-factor priors.

Model (hitters):
    cf_mu ~ Normal(1.2, 0.3)            # population conversion factor
    cf_sigma ~ HalfNormal(0.3)           # player-level variation
    cf_i ~ Normal(cf_mu, cf_sigma)       # per-player conversion factor
    mu_npb_i = prev_wOBA_i * cf_i        # expected NPB wOBA
    sigma_obs ~ HalfNormal(0.1)          # residual noise
    wOBA_obs_i ~ Normal(mu_npb_i, sigma_obs)

Model (pitchers):
    Same structure with ERA and inverted interpretation.

Usage:
    python src/bayesian_model.py [--test-only]

Designed to run in GitHub Actions (ubuntu-latest, ~2GB RAM sufficient).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FOREIGN_DIR = DATA_DIR / "foreign"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = DATA_DIR / "model"

# Train/test split year
SPLIT_YEAR = 2020  # train: 2015-2019, test: 2020-2025


def load_hitter_pairs() -> list[dict]:
    """Load hitters with both prev_wOBA and NPB first-year wOBA."""
    path = FOREIGN_DIR / "player_conversion_details.csv"
    pairs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("prev_wOBA") and row.get("npb_first_wOBA") and row.get("wOBA_ratio"):
                try:
                    pairs.append({
                        "name": row["english_name"],
                        "npb_name": row["npb_name"],
                        "origin_league": row["origin_league"],
                        "year": int(row["npb_first_year"]),
                        "prev_wOBA": float(row["prev_wOBA"]),
                        "npb_wOBA": float(row["npb_first_wOBA"]),
                        "ratio": float(row["wOBA_ratio"]),
                    })
                except (ValueError, KeyError):
                    continue
    return pairs


def load_pitcher_pairs() -> list[dict]:
    """Load pitchers with both prev_ERA and NPB first-year ERA."""
    path = FOREIGN_DIR / "player_conversion_details.csv"
    pairs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("prev_ERA") and row.get("npb_first_ERA") and row.get("ERA_ratio"):
                try:
                    pairs.append({
                        "name": row["english_name"],
                        "npb_name": row["npb_name"],
                        "origin_league": row["origin_league"],
                        "year": int(row["npb_first_year"]),
                        "prev_ERA": float(row["prev_ERA"]),
                        "npb_ERA": float(row["npb_first_ERA"]),
                        "ratio": float(row["ERA_ratio"]),
                    })
                except (ValueError, KeyError):
                    continue
    return pairs


def load_npb_league_avg_woba() -> dict[int, float]:
    """Load NPB league-average wOBA per year from sabermetrics data."""
    path = RAW_DIR / "npb_sabermetrics_2015_2025.csv"
    yearly: dict[int, list[float]] = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                year = int(row["year"])
                woba = float(row["wOBA"])
                pa = int(row["PA"])
                if pa >= 100:
                    yearly[year].append(woba)
            except (ValueError, KeyError):
                continue
    return {y: float(np.mean(vals)) for y, vals in yearly.items()}


def fit_hitter_model(
    train_data: list[dict],
    draws: int = 2000,
    tune: int = 1000,
) -> tuple[pm.Model, az.InferenceData]:
    """Fit hierarchical conversion factor model for hitters.

    Returns (model, trace).
    """
    prev_woba = np.array([d["prev_wOBA"] for d in train_data])
    npb_woba = np.array([d["npb_wOBA"] for d in train_data])
    n = len(train_data)

    print(f"\nFitting hitter model (n={n}, draws={draws}, tune={tune})...")

    with pm.Model() as model:
        # Population-level conversion factor
        cf_mu = pm.Normal("cf_mu", mu=1.2, sigma=0.3)
        cf_sigma = pm.HalfNormal("cf_sigma", sigma=0.3)

        # Player-level conversion factors
        cf_i = pm.Normal("cf_i", mu=cf_mu, sigma=cf_sigma, shape=n)

        # Expected NPB wOBA
        mu_npb = prev_woba * cf_i

        # Residual observation noise
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.1)

        # Likelihood
        pm.Normal("wOBA_obs", mu=mu_npb, sigma=sigma_obs, observed=npb_woba)

        # Sample
        trace = pm.sample(
            draws=draws,
            tune=tune,
            cores=2,
            random_seed=42,
            return_inferencedata=True,
        )

    return model, trace


def fit_pitcher_model(
    train_data: list[dict],
    draws: int = 2000,
    tune: int = 1000,
) -> tuple[pm.Model, az.InferenceData]:
    """Fit hierarchical conversion factor model for pitchers."""
    prev_era = np.array([d["prev_ERA"] for d in train_data])
    npb_era = np.array([d["npb_ERA"] for d in train_data])
    n = len(train_data)

    print(f"\nFitting pitcher model (n={n}, draws={draws}, tune={tune})...")

    with pm.Model() as model:
        # Population-level conversion factor for ERA
        cf_mu = pm.Normal("cf_mu", mu=0.6, sigma=0.3)
        cf_sigma = pm.HalfNormal("cf_sigma", sigma=0.3)

        # Player-level conversion factors
        cf_i = pm.Normal("cf_i", mu=cf_mu, sigma=cf_sigma, shape=n)

        # Expected NPB ERA
        mu_npb = prev_era * cf_i

        # Residual noise
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

        # Likelihood
        pm.Normal("ERA_obs", mu=mu_npb, sigma=sigma_obs, observed=npb_era)

        # Sample
        trace = pm.sample(
            draws=draws,
            tune=tune,
            cores=2,
            random_seed=42,
            return_inferencedata=True,
        )

    return model, trace


def predict_new_player(
    trace: az.InferenceData,
    prev_stat: float,
    n_samples: int = 4000,
) -> dict[str, float]:
    """Generate posterior predictive for a new player.

    Draws cf_new ~ Normal(cf_mu_post, cf_sigma_post) from the trace,
    then predicted = prev_stat * cf_new + noise.

    Returns dict with mean, median, hdi_80, hdi_95.
    """
    cf_mu_samples = trace.posterior["cf_mu"].values.flatten()
    cf_sigma_samples = trace.posterior["cf_sigma"].values.flatten()
    sigma_obs_samples = trace.posterior["sigma_obs"].values.flatten()

    rng = np.random.default_rng(42)
    idx = rng.choice(len(cf_mu_samples), size=n_samples, replace=True)

    cf_new = rng.normal(cf_mu_samples[idx], cf_sigma_samples[idx])
    noise = rng.normal(0, sigma_obs_samples[idx])
    predicted = prev_stat * cf_new + noise

    return {
        "mean": float(np.mean(predicted)),
        "median": float(np.median(predicted)),
        "std": float(np.std(predicted)),
        "hdi_80_low": float(np.percentile(predicted, 10)),
        "hdi_80_high": float(np.percentile(predicted, 90)),
        "hdi_95_low": float(np.percentile(predicted, 2.5)),
        "hdi_95_high": float(np.percentile(predicted, 97.5)),
    }


def backtest(
    trace: az.InferenceData,
    test_data: list[dict],
    stat_key: str,
    prev_key: str,
    league_avg: dict[int, float] | None = None,
) -> dict[str, float]:
    """Backtest model predictions against actual outcomes.

    Compares:
    - Baseline: league average (or global mean of test set)
    - Raw conversion factor: prev_stat * median(cf_mu)
    - Bayesian: full posterior predictive

    Returns dict of MAE values.
    """
    cf_mu_median = float(np.median(trace.posterior["cf_mu"].values))

    baseline_errors = []
    raw_cf_errors = []
    bayes_errors = []
    predictions = []

    for d in test_data:
        actual = d[stat_key]
        prev = d[prev_key]
        year = d["year"]

        # Baseline: league average or global mean
        if league_avg and year in league_avg:
            baseline_pred = league_avg[year]
        else:
            baseline_pred = float(np.mean([t[stat_key] for t in test_data]))

        # Raw conversion factor
        raw_pred = prev * cf_mu_median

        # Bayesian posterior predictive
        bayes = predict_new_player(trace, prev)
        bayes_pred = bayes["mean"]

        baseline_errors.append(abs(actual - baseline_pred))
        raw_cf_errors.append(abs(actual - raw_pred))
        bayes_errors.append(abs(actual - bayes_pred))

        predictions.append({
            "name": d["name"],
            "npb_name": d["npb_name"],
            "year": year,
            "origin_league": d["origin_league"],
            f"prev_{stat_key}": f"{prev:.4f}",
            f"actual_{stat_key}": f"{actual:.4f}",
            "baseline_pred": f"{baseline_pred:.4f}",
            "raw_cf_pred": f"{raw_pred:.4f}",
            "bayes_pred": f"{bayes_pred:.4f}",
            "bayes_hdi_80": f"[{bayes['hdi_80_low']:.4f}, {bayes['hdi_80_high']:.4f}]",
            "bayes_hdi_95": f"[{bayes['hdi_95_low']:.4f}, {bayes['hdi_95_high']:.4f}]",
        })

    results = {
        "baseline_mae": float(np.mean(baseline_errors)),
        "raw_cf_mae": float(np.mean(raw_cf_errors)),
        "bayes_mae": float(np.mean(bayes_errors)),
        "n_test": len(test_data),
        "predictions": predictions,
    }

    # Coverage: how often does actual fall within 80% HDI?
    in_80 = sum(
        1 for d, p in zip(test_data, predictions)
        if float(p["bayes_hdi_80"].strip("[]").split(",")[0]) <= d[stat_key]
        <= float(p["bayes_hdi_80"].strip("[]").split(",")[1])
    )
    in_95 = sum(
        1 for d, p in zip(test_data, predictions)
        if float(p["bayes_hdi_95"].strip("[]").split(",")[0]) <= d[stat_key]
        <= float(p["bayes_hdi_95"].strip("[]").split(",")[1])
    )
    results["coverage_80"] = in_80 / len(test_data)
    results["coverage_95"] = in_95 / len(test_data)

    return results


def write_outputs(
    hitter_trace: az.InferenceData | None,
    pitcher_trace: az.InferenceData | None,
    hitter_backtest: dict | None,
    pitcher_backtest: dict | None,
) -> None:
    """Write model outputs to CSV and JSON."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Trace summary
    if hitter_trace is not None:
        summary = az.summary(hitter_trace, var_names=["cf_mu", "cf_sigma", "sigma_obs"])
        summary.to_csv(MODEL_DIR / "hitter_trace_summary.csv")
        print(f"\nHitter trace summary:\n{summary}")

    if pitcher_trace is not None:
        summary = az.summary(pitcher_trace, var_names=["cf_mu", "cf_sigma", "sigma_obs"])
        summary.to_csv(MODEL_DIR / "pitcher_trace_summary.csv")
        print(f"\nPitcher trace summary:\n{summary}")

    # Backtest results
    if hitter_backtest is not None:
        # Summary JSON
        summary = {k: v for k, v in hitter_backtest.items() if k != "predictions"}
        with open(MODEL_DIR / "hitter_backtest_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Predictions CSV
        preds = hitter_backtest["predictions"]
        if preds:
            keys = list(preds[0].keys())
            with open(MODEL_DIR / "hitter_backtest_predictions.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(preds)

    if pitcher_backtest is not None:
        summary = {k: v for k, v in pitcher_backtest.items() if k != "predictions"}
        with open(MODEL_DIR / "pitcher_backtest_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        preds = pitcher_backtest["predictions"]
        if preds:
            keys = list(preds[0].keys())
            with open(MODEL_DIR / "pitcher_backtest_predictions.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="Skip training, run quick test")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    args = parser.parse_args()

    # Load data
    hitter_pairs = load_hitter_pairs()
    pitcher_pairs = load_pitcher_pairs()
    league_avg_woba = load_npb_league_avg_woba()

    print(f"Total hitter pairs: {len(hitter_pairs)}")
    print(f"Total pitcher pairs: {len(pitcher_pairs)}")

    # Split
    h_train = [d for d in hitter_pairs if d["year"] < SPLIT_YEAR]
    h_test = [d for d in hitter_pairs if d["year"] >= SPLIT_YEAR]
    p_train = [d for d in pitcher_pairs if d["year"] < SPLIT_YEAR]
    p_test = [d for d in pitcher_pairs if d["year"] >= SPLIT_YEAR]

    print(f"\nHitters — train: {len(h_train)}, test: {len(h_test)}")
    print(f"Pitchers — train: {len(p_train)}, test: {len(p_test)}")

    if args.test_only:
        print("\n[test-only mode: skipping model fitting]")
        return

    # === Hitter Model ===
    hitter_trace = None
    hitter_bt = None
    if h_train:
        _, hitter_trace = fit_hitter_model(h_train, draws=args.draws, tune=args.tune)

        # Backtest
        if h_test:
            print("\n=== Hitter Backtest ===")
            hitter_bt = backtest(
                hitter_trace, h_test,
                stat_key="npb_wOBA", prev_key="prev_wOBA",
                league_avg=league_avg_woba,
            )
            print(f"  Baseline MAE (league avg): {hitter_bt['baseline_mae']:.4f}")
            print(f"  Raw CF MAE:                {hitter_bt['raw_cf_mae']:.4f}")
            print(f"  Bayesian MAE:              {hitter_bt['bayes_mae']:.4f}")
            print(f"  80% coverage: {hitter_bt['coverage_80']:.1%}")
            print(f"  95% coverage: {hitter_bt['coverage_95']:.1%}")

            improvement = (hitter_bt["baseline_mae"] - hitter_bt["bayes_mae"]) / hitter_bt["baseline_mae"] * 100
            print(f"  Improvement vs baseline: {improvement:+.1f}%")

    # === Pitcher Model ===
    pitcher_trace = None
    pitcher_bt = None
    if p_train:
        _, pitcher_trace = fit_pitcher_model(p_train, draws=args.draws, tune=args.tune)

        # Backtest
        if p_test:
            print("\n=== Pitcher Backtest ===")
            pitcher_bt = backtest(
                pitcher_trace, p_test,
                stat_key="npb_ERA", prev_key="prev_ERA",
            )
            print(f"  Baseline MAE (global avg): {pitcher_bt['baseline_mae']:.2f}")
            print(f"  Raw CF MAE:                {pitcher_bt['raw_cf_mae']:.2f}")
            print(f"  Bayesian MAE:              {pitcher_bt['bayes_mae']:.2f}")
            print(f"  80% coverage: {pitcher_bt['coverage_80']:.1%}")
            print(f"  95% coverage: {pitcher_bt['coverage_95']:.1%}")

            improvement = (pitcher_bt["baseline_mae"] - pitcher_bt["bayes_mae"]) / pitcher_bt["baseline_mae"] * 100
            print(f"  Improvement vs baseline: {improvement:+.1f}%")

    # Write outputs
    write_outputs(hitter_trace, pitcher_trace, hitter_bt, pitcher_bt)
    print(f"\nOutputs written to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
