"""
Step 7b: Team-level Marcel vs Stan comparison (2022-2025).

Uses actual player PA/IP weights and Stan predictions for ALL players
(Japanese + foreign first-year) to compare team RS/RA, and then
converts to team win predictions via Pythagorean expectation.

Method:
  Marcel team RS  = K_WOBA × Σ_player(Marcel_wOBA × actual_PA)
  Stan   team RS  = K_WOBA × Σ_player(Stan_wOBA   × actual_PA)

  Marcel team RA  = Σ_player(Marcel_ERA × actual_IP / 9)
  Stan   team RA  = Σ_player(Stan_ERA   × actual_IP / 9)

  Pythagorean wins → compare MAE vs actual W

Output:
  data/projections/team_compare_results.json
  data/projections/team_compare_results.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
RAW_DIR   = DATA_DIR / "raw"
OUT_DIR   = DATA_DIR / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NPB_PYTH_EXP = 1.83
NPB_HIST_RS  = 535.0

# K_WOBA: RS ≈ K_WOBA × Σ(wOBA × PA)
# Calibrated: NPB_HIST_RS / (avg_wOBA × NPB_TARGET_PA) ≈ 535 / (0.310 × 5300) = 0.3256
K_WOBA   = 0.3256

COMPARE_YEARS = list(range(2022, 2026))  # overlap with Stan backtest period

# ── Data source (npb-prediction GitHub raw) ───────────────────────────────────
NPBP_BASE = (
    "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main/data/projections"
)


def ip_to_decimal(ip: float) -> float:
    whole  = int(ip)
    thirds = round((ip - whole) * 10)
    return whole + thirds / 3.0


def load_player_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Stan predictions for all players:
    - Japanese: from stan_jpn_model output (data/model/jpn_*.csv)
    - Foreign first-year: from stan_model output (data/model/stan_*_v1_predictions.csv)
    Returns (hitters, pitchers) each with columns:
      year, player, team, marcel_woba/era, stan_woba/era
    """
    # Japanese predictions
    jpn_h = pd.read_csv(MODEL_DIR / "jpn_hitter_predictions.csv",  encoding="utf-8-sig")
    jpn_p = pd.read_csv(MODEL_DIR / "jpn_pitcher_predictions.csv", encoding="utf-8-sig")

    # Foreign first-year predictions (add team via raw NPB stats)
    fgn_h = pd.read_csv(MODEL_DIR / "stan_hitter_v1_predictions.csv")
    fgn_p = pd.read_csv(MODEL_DIR / "stan_pitcher_v1_predictions.csv")

    # Load raw data to get team/PA/IP for foreign players
    raw_h = pd.read_csv(RAW_DIR / "npb_hitters_2015_2025.csv",  encoding="utf-8-sig")
    raw_p = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig")

    # For foreign hitters: baseline_pred → marcel_woba, stan_pred → stan_woba
    # Note: foreign model predicts wOBA (same metric as jpn model)
    fgn_h = fgn_h.rename(columns={"baseline_pred": "marcel_woba", "stan_pred": "stan_woba",
                                    "npb_name": "player"})
    fgn_h = fgn_h.merge(
        raw_h[["player", "year", "team", "PA"]],
        on=["player", "year"], how="inner"
    )
    fgn_h = fgn_h[["year", "player", "team", "marcel_woba", "stan_woba", "PA"]]
    fgn_h["actual_PA"] = fgn_h["PA"]
    fgn_h = fgn_h.drop(columns=["PA"])

    # For foreign pitchers: baseline_pred → marcel_era, stan_pred → stan_era
    fgn_p = fgn_p.rename(columns={"baseline_pred": "marcel_era", "stan_pred": "stan_era",
                                    "npb_name": "player"})
    raw_p_dec = raw_p.copy()
    raw_p_dec["actual_IP"] = raw_p_dec["IP"].apply(ip_to_decimal)
    fgn_p = fgn_p.merge(
        raw_p_dec[["player", "year", "team", "actual_IP"]],
        on=["player", "year"], how="inner"
    )
    fgn_p = fgn_p[["year", "player", "team", "marcel_era", "stan_era", "actual_IP"]]

    # Filter to comparison years
    jpn_h = jpn_h[jpn_h["year"].isin(COMPARE_YEARS)]
    jpn_p = jpn_p[jpn_p["year"].isin(COMPARE_YEARS)]
    fgn_h = fgn_h[fgn_h["year"].isin(COMPARE_YEARS)]
    fgn_p = fgn_p[fgn_p["year"].isin(COMPARE_YEARS)]

    # Rename for merge
    jpn_h = jpn_h.rename(columns={"actual_PA": "actual_PA"})

    # Remove foreign first-year players from Japanese set (avoid double-counting)
    fgn_h_keys = set(zip(fgn_h["player"], fgn_h["year"]))
    fgn_p_keys = set(zip(fgn_p["player"], fgn_p["year"]))
    jpn_h = jpn_h[~jpn_h.apply(lambda r: (r["player"], r["year"]) in fgn_h_keys, axis=1)]
    jpn_p = jpn_p[~jpn_p.apply(lambda r: (r["player"], r["year"]) in fgn_p_keys, axis=1)]

    # Combine Japanese + Foreign
    all_h = pd.concat([
        jpn_h[["year", "player", "team", "marcel_woba", "stan_woba", "actual_PA"]],
        fgn_h[["year", "player", "team", "marcel_woba", "stan_woba", "actual_PA"]],
    ], ignore_index=True)

    all_p = pd.concat([
        jpn_p[["year", "player", "team", "marcel_era", "stan_era", "actual_IP"]],
        fgn_p[["year", "player", "team", "marcel_era", "stan_era", "actual_IP"]],
    ], ignore_index=True)

    print(f"  All hitters: {len(all_h)} player-years  "
          f"({len(fgn_h)} foreign first-year)")
    print(f"  All pitchers: {len(all_p)} player-years  "
          f"({len(fgn_p)} foreign first-year)")

    return all_h, all_p


def compute_team_rs_ra(all_h: pd.DataFrame, all_p: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player predictions to team-year RS/RA for both Marcel and Stan."""
    # Hitters → RS
    h = all_h.copy()
    h["rs_marcel"] = K_WOBA * h["marcel_woba"] * h["actual_PA"]
    h["rs_stan"]   = K_WOBA * h["stan_woba"]   * h["actual_PA"]
    rs = h.groupby(["year", "team"])[["rs_marcel", "rs_stan"]].sum().reset_index()

    # Pitchers → RA
    p = all_p.copy()
    p["ra_marcel"] = p["marcel_era"] * p["actual_IP"] / 9.0
    p["ra_stan"]   = p["stan_era"]   * p["actual_IP"] / 9.0
    ra = p.groupby(["year", "team"])[["ra_marcel", "ra_stan"]].sum().reset_index()

    # Merge
    team = rs.merge(ra, on=["year", "team"], how="inner")
    return team


def pythagorean_wins(rs, ra, g):
    rs = np.clip(rs, 1.0, None)
    ra = np.clip(ra, 1.0, None)
    wpct = rs ** NPB_PYTH_EXP / (rs ** NPB_PYTH_EXP + ra ** NPB_PYTH_EXP)
    return wpct * g


def run_comparison() -> None:
    print("Loading player predictions...")
    all_h, all_p = load_player_predictions()

    print("Aggregating to team-year RS/RA...")
    team = compute_team_rs_ra(all_h, all_p)

    print("Loading actual results...")
    actual = pd.read_csv(
        "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main"
        "/data/projections/pythagorean_2015_2025.csv",
        encoding="utf-8-sig",
    )
    actual = actual[actual["year"].isin(COMPARE_YEARS)][["year", "team", "G", "W", "RS", "RA"]]

    merged = team.merge(actual, on=["year", "team"], how="inner")
    print(f"  Matched: {len(merged)} team-years")

    # Post-hoc scale RS/RA to league average (same as team_sim.py)
    # This removes systematic Marcel bias, testing only the correction from K%/BB%
    for yr in COMPARE_YEARS:
        mask = merged["year"] == yr
        if not mask.any():
            continue
        for col_rs, col_ra in [("rs_marcel", "ra_marcel"), ("rs_stan", "ra_stan")]:
            avg_rs = merged.loc[mask, col_rs].mean()
            avg_ra = merged.loc[mask, col_ra].mean()
            if avg_rs > 0:
                merged.loc[mask, col_rs] *= NPB_HIST_RS / avg_rs
            if avg_ra > 0:
                merged.loc[mask, col_ra] *= NPB_HIST_RS / avg_ra

    # Pythagorean wins
    merged["W_marcel"] = pythagorean_wins(
        merged["rs_marcel"].values, merged["ra_marcel"].values, merged["G"].values)
    merged["W_stan"] = pythagorean_wins(
        merged["rs_stan"].values, merged["ra_stan"].values, merged["G"].values)

    merged["err_marcel"] = merged["W_marcel"] - merged["W"]
    merged["err_stan"]   = merged["W_stan"]   - merged["W"]

    mae_marcel = float(merged["err_marcel"].abs().mean())
    mae_stan   = float(merged["err_stan"].abs().mean())
    bias_m = float(merged["err_marcel"].mean())
    bias_s = float(merged["err_stan"].mean())

    print(f"\n── Team Win MAE: Marcel vs Stan (2022-2025, n={len(merged)}) ──────────")
    print(f"{'':20s}  {'Marcel':>8s}  {'Stan':>8s}  {'Δ (Stan-Marcel)':>15s}")
    print(f"{'MAE (wins)':20s}  {mae_marcel:8.3f}  {mae_stan:8.3f}  {mae_stan - mae_marcel:+15.3f}")
    print(f"{'Bias (wins)':20s}  {bias_m:+8.3f}  {bias_s:+8.3f}  {bias_s - bias_m:+15.3f}")

    # Year-by-year breakdown
    print(f"\n{'Year':>6}  {'MAE Marcel':>10}  {'MAE Stan':>8}  {'Δ MAE':>8}  N")
    for yr, grp in merged.groupby("year"):
        m = grp["err_marcel"].abs().mean()
        s = grp["err_stan"].abs().mean()
        print(f"  {yr}  {m:10.3f}  {s:8.3f}  {s - m:+8.3f}  {len(grp)}")

    # Save
    out_rows = []
    for _, row in merged.iterrows():
        out_rows.append({
            "year":      int(row["year"]),
            "team":      row["team"],
            "actual_W":  float(row["W"]),
            "W_marcel":  round(float(row["W_marcel"]), 1),
            "W_stan":    round(float(row["W_stan"]), 1),
            "err_marcel": round(float(row["err_marcel"]), 1),
            "err_stan":   round(float(row["err_stan"]), 1),
        })

    summary = {
        "mae_marcel":  round(mae_marcel, 3),
        "mae_stan":    round(mae_stan, 3),
        "delta_mae":   round(mae_stan - mae_marcel, 3),
        "bias_marcel": round(bias_m, 3),
        "bias_stan":   round(bias_s, 3),
        "n":           len(merged),
        "years":       COMPARE_YEARS,
        "detail":      out_rows,
    }

    json_path = OUT_DIR / "team_compare_results.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {json_path}")

    (
        merged[["year", "team", "W", "W_marcel", "W_stan", "err_marcel", "err_stan"]]
        .rename(columns={"W": "actual_W"})
        .sort_values(["year", "team"])
        .to_csv(OUT_DIR / "team_compare_results.csv", index=False, encoding="utf-8-sig")
    )
    print(f"Saved -> {OUT_DIR / 'team_compare_results.csv'}")


if __name__ == "__main__":
    run_comparison()
