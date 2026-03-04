"""Step 12 diagnostic: Big-miss team-years breakdown.

For team-years with |err| > 10W, show:
  1. Foreign player RS/RA contribution (in-model vs missing)
  2. Marcel vs Stan player-level breakdown (where Stan helped / hurt)
  3. Coverage gap analysis

Output:
  data/projections/big_miss_diagnosis.json
  stdout: formatted tables
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stan_jpn_model import MIN_IP, MIN_PA, build_dataset, compute_fip_column, ip_to_decimal, load_birthday_df, standardize_features
from statistical_validation import (
    ALPHA_JPN_H, ALPHA_JPN_P, JPN_YEARS, K_WOBA, NPB_HIST_RS, NPB_PYTH_EXP,
    _compute_league_averages, _load_raw_data, _reassign_teams,
    ridge_fit_predict,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FOREIGN_DIR = DATA_DIR / "foreign"
OUT_DIR = DATA_DIR / "projections"

ERR_THRESHOLD = 10.0  # wins


def load_foreign_names():
    """Load set of foreign player names from master CSV."""
    master = pd.read_csv(FOREIGN_DIR / "foreign_players_master.csv", encoding="utf-8")
    return set(master["npb_name"].values)


def run_loocv_with_detail():
    """Run 8-year LOO-CV and return player-level h_df, p_df with reassigned teams."""
    saber = pd.read_csv(RAW_DIR / "npb_sabermetrics_2015_2025.csv", encoding="utf-8-sig")
    pitchers = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig")
    saber = saber.dropna(subset=["wOBA"])
    pitchers = compute_fip_column(pitchers)
    bday_df = load_birthday_df()

    feat_h = ["K_pct", "BB_pct", "BABIP", "age_from_peak"]
    feat_p = ["K_pct", "BB_pct", "age_from_peak"]

    all_h, all_p = [], []

    for hold_yr in JPN_YEARS:
        train_years = [y for y in JPN_YEARS if y != hold_yr]
        train_h, train_p, _ = build_dataset(saber, pitchers, train_years, bday_df)
        test_h, test_p, _ = build_dataset(saber, pitchers, [hold_yr], bday_df)

        if len(test_h) == 0 or len(test_p) == 0:
            continue

        # Hitters
        train_z_h, test_z_h, _, _ = standardize_features(train_h, test_h, feat_h)
        y_h = (train_h["actual_woba"] - train_h["marcel_woba"]).values
        delta_h, _ = ridge_fit_predict(train_z_h, y_h, test_z_h, ALPHA_JPN_H)
        stan_woba = test_h["marcel_woba"].values + delta_h

        for i, (_, row) in enumerate(test_h.iterrows()):
            all_h.append({
                "year": hold_yr, "player": row["player"], "team": row["team"],
                "actual": row["actual_woba"], "marcel": row["marcel_woba"],
                "stan": stan_woba[i], "actual_PA": row["actual_PA"],
            })

        # Pitchers
        train_z_p, test_z_p, _, _ = standardize_features(train_p, test_p, feat_p)
        y_p = (train_p["actual_era"] - train_p["marcel_era"]).values
        delta_p, _ = ridge_fit_predict(train_z_p, y_p, test_z_p, ALPHA_JPN_P)
        stan_era = test_p["marcel_era"].values + delta_p

        for i, (_, row) in enumerate(test_p.iterrows()):
            all_p.append({
                "year": hold_yr, "player": row["player"], "team": row["team"],
                "actual": row["actual_era"], "marcel": row["marcel_era"],
                "stan": stan_era[i], "actual_IP": row["actual_IP"],
            })

    h_df = pd.DataFrame(all_h)
    p_df = pd.DataFrame(all_p)

    # Reassign teams
    saber_raw, pitchers_raw = _load_raw_data()
    h_df, p_df = _reassign_teams(h_df, p_df, saber_raw, pitchers_raw)

    return h_df, p_df, saber_raw, pitchers_raw


def main():
    print("=" * 70)
    print("Step 12 Diagnostic: Big-Miss Team-Years Breakdown")
    print("=" * 70)

    foreign_names = load_foreign_names()
    h_df, p_df, saber, pitchers_raw = run_loocv_with_detail()
    lg_woba, lg_era = _compute_league_averages(saber, pitchers_raw)

    # Load actual wins
    actual = pd.read_csv(
        "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main"
        "/data/projections/pythagorean_2015_2025.csv",
        encoding="utf-8-sig",
    )

    # --- Compute team predictions (same logic as team_level_mae) ---
    from statistical_validation import _impute_missing_players

    impute = _impute_missing_players(h_df, p_df, saber, pitchers_raw, lg_woba, lg_era)

    h = h_df.copy()
    h["rs_marcel"] = K_WOBA * h["marcel"] * h["actual_PA"]
    h["rs_stan"] = K_WOBA * h["stan"] * h["actual_PA"]
    rs = h.groupby(["year", "team"])[["rs_marcel", "rs_stan"]].sum().reset_index()

    p = p_df.copy()
    p["ra_marcel"] = p["marcel"] * p["actual_IP"] / 9.0
    p["ra_stan"] = p["stan"] * p["actual_IP"] / 9.0
    ra = p.groupby(["year", "team"])[["ra_marcel", "ra_stan"]].sum().reset_index()

    team = rs.merge(ra, on=["year", "team"], how="inner")
    merged = team.merge(actual[["year", "team", "G", "W"]], on=["year", "team"], how="inner")

    # Add imputation
    merged["imp_rs"] = merged.apply(
        lambda r: impute.get((r["year"], r["team"]), {}).get("imputed_rs", 0), axis=1)
    merged["imp_ra"] = merged.apply(
        lambda r: impute.get((r["year"], r["team"]), {}).get("imputed_ra", 0), axis=1)
    merged["rs_marcel"] += merged["imp_rs"]
    merged["rs_stan"] += merged["imp_rs"]
    merged["ra_marcel"] += merged["imp_ra"]
    merged["ra_stan"] += merged["imp_ra"]

    # Scaling
    for yr in merged["year"].unique():
        mask = merged["year"] == yr
        for col_m, col_s in [("rs_marcel", "rs_stan"), ("ra_marcel", "ra_stan")]:
            avg = merged.loc[mask, col_m].mean()
            if avg > 0:
                f = NPB_HIST_RS / avg
                merged.loc[mask, col_m] *= f
                merged.loc[mask, col_s] *= f

    # Pythagorean wins
    for model in ["marcel", "stan"]:
        rs_v = np.clip(merged[f"rs_{model}"].values, 1.0, None)
        ra_v = np.clip(merged[f"ra_{model}"].values, 1.0, None)
        wpct = rs_v ** NPB_PYTH_EXP / (rs_v ** NPB_PYTH_EXP + ra_v ** NPB_PYTH_EXP)
        merged[f"W_{model}"] = wpct * merged["G"].values

    merged["err_M"] = merged["W_marcel"] - merged["W"]
    merged["err_S"] = merged["W_stan"] - merged["W"]

    # --- Identify big misses ---
    big = merged[(merged["err_M"].abs() > ERR_THRESHOLD) | (merged["err_S"].abs() > ERR_THRESHOLD)]
    big = big.sort_values("err_M", key=abs, ascending=False)

    print(f"\nBig-miss team-years (|err| > {ERR_THRESHOLD}W): {len(big)}")
    print("=" * 70)

    results = []

    for _, row in big.iterrows():
        yr = int(row["year"])
        tm = row["team"]
        print(f"\n{'='*60}")
        print(f"  {yr} {tm}  actual={int(row['W'])}W  "
              f"Marcel={row['W_marcel']:.1f}  Stan={row['W_stan']:.1f}  "
              f"err_M={row['err_M']:+.1f}  err_S={row['err_S']:+.1f}")
        print(f"{'='*60}")

        # --- Hitter breakdown ---
        yr_h = h_df[(h_df["year"] == yr) & (h_df["team"] == tm)].copy()
        yr_h["is_foreign"] = yr_h["player"].isin(foreign_names)
        yr_h["rs_actual"] = K_WOBA * yr_h["actual"] * yr_h["actual_PA"]
        yr_h["rs_marcel"] = K_WOBA * yr_h["marcel"] * yr_h["actual_PA"]
        yr_h["rs_stan"] = K_WOBA * yr_h["stan"] * yr_h["actual_PA"]
        yr_h["err_m"] = yr_h["rs_marcel"] - yr_h["rs_actual"]
        yr_h["err_s"] = yr_h["rs_stan"] - yr_h["rs_actual"]
        yr_h["stan_better"] = yr_h["err_s"].abs() < yr_h["err_m"].abs()

        # Missing foreign hitters (in raw data but not in model)
        all_h_raw = saber[(saber["year"] == yr) & (saber["team"] == tm)]
        all_h_raw = all_h_raw[all_h_raw["PA"] >= MIN_PA]
        model_players_h = set(yr_h["player"])
        missing_h = all_h_raw[~all_h_raw["player"].isin(model_players_h)].copy()
        missing_h["is_foreign"] = missing_h["player"].isin(foreign_names)

        print(f"\n  [Hitters in model: {len(yr_h)}  Missing: {len(missing_h)}]")

        # In-model foreign hitters
        fgn_h = yr_h[yr_h["is_foreign"]]
        jpn_h = yr_h[~yr_h["is_foreign"]]

        if len(fgn_h) > 0:
            print(f"  Foreign hitters IN MODEL ({len(fgn_h)}):")
            for _, ph in fgn_h.sort_values("actual_PA", ascending=False).iterrows():
                print(f"    {ph['player']:10s}  PA={int(ph['actual_PA']):>3d}  "
                      f"wOBA: actual={ph['actual']:.3f} M={ph['marcel']:.3f} S={ph['stan']:.3f}  "
                      f"RS_err: M={ph['err_m']:+.1f} S={ph['err_s']:+.1f}  "
                      f"{'✓Stan' if ph['stan_better'] else '✗Marcel'}")

        # Missing foreign hitters
        missing_fgn_h = missing_h[missing_h["is_foreign"]]
        missing_jpn_h = missing_h[~missing_h["is_foreign"]]
        if len(missing_fgn_h) > 0:
            total_fgn_pa = int(missing_fgn_h["PA"].sum())
            print(f"  Foreign hitters MISSING ({len(missing_fgn_h)}, {total_fgn_pa} PA):")
            for _, mh in missing_fgn_h.sort_values("PA", ascending=False).iterrows():
                woba = mh["wOBA"] if pd.notna(mh["wOBA"]) else 0
                print(f"    {mh['player']:10s}  PA={int(mh['PA']):>3d}  wOBA={woba:.3f}")

        if len(missing_jpn_h) > 0:
            total_jpn_pa = int(missing_jpn_h["PA"].sum())
            print(f"  Japanese hitters MISSING ({len(missing_jpn_h)}, {total_jpn_pa} PA):")
            for _, mh in missing_jpn_h.sort_values("PA", ascending=False).head(5).iterrows():
                woba = mh["wOBA"] if pd.notna(mh["wOBA"]) else 0
                print(f"    {mh['player']:10s}  PA={int(mh['PA']):>3d}  wOBA={woba:.3f}")

        # --- Pitcher breakdown ---
        yr_p = p_df[(p_df["year"] == yr) & (p_df["team"] == tm)].copy()
        yr_p["is_foreign"] = yr_p["player"].isin(foreign_names)
        yr_p["ra_actual"] = yr_p["actual"] * yr_p["actual_IP"] / 9.0
        yr_p["ra_marcel"] = yr_p["marcel"] * yr_p["actual_IP"] / 9.0
        yr_p["ra_stan"] = yr_p["stan"] * yr_p["actual_IP"] / 9.0
        yr_p["err_m"] = yr_p["ra_marcel"] - yr_p["ra_actual"]
        yr_p["err_s"] = yr_p["ra_stan"] - yr_p["ra_actual"]
        yr_p["stan_better"] = yr_p["err_s"].abs() < yr_p["err_m"].abs()

        # Missing foreign pitchers
        all_p_raw = pitchers_raw[(pitchers_raw["year"] == yr) & (pitchers_raw["team"] == tm)]
        all_p_raw = all_p_raw[all_p_raw["IP_dec"] >= MIN_IP]
        model_players_p = set(yr_p["player"])
        missing_p = all_p_raw[~all_p_raw["player"].isin(model_players_p)].copy()
        missing_p["is_foreign"] = missing_p["player"].isin(foreign_names)

        print(f"\n  [Pitchers in model: {len(yr_p)}  Missing: {len(missing_p)}]")

        fgn_p = yr_p[yr_p["is_foreign"]]
        jpn_p = yr_p[~yr_p["is_foreign"]]

        if len(fgn_p) > 0:
            print(f"  Foreign pitchers IN MODEL ({len(fgn_p)}):")
            for _, pp in fgn_p.sort_values("actual_IP", ascending=False).iterrows():
                print(f"    {pp['player']:10s}  IP={pp['actual_IP']:>5.1f}  "
                      f"ERA: actual={pp['actual']:.2f} M={pp['marcel']:.2f} S={pp['stan']:.2f}  "
                      f"RA_err: M={pp['err_m']:+.1f} S={pp['err_s']:+.1f}  "
                      f"{'✓Stan' if pp['stan_better'] else '✗Marcel'}")

        missing_fgn_p = missing_p[missing_p["is_foreign"]]
        missing_jpn_p = missing_p[~missing_p["is_foreign"]]
        if len(missing_fgn_p) > 0:
            total_fgn_ip = round(missing_fgn_p["IP_dec"].sum(), 1)
            print(f"  Foreign pitchers MISSING ({len(missing_fgn_p)}, {total_fgn_ip} IP):")
            for _, mp in missing_fgn_p.sort_values("IP_dec", ascending=False).iterrows():
                era = mp["ERA_num"] if pd.notna(mp["ERA_num"]) else 0
                print(f"    {mp['player']:10s}  IP={mp['IP_dec']:>5.1f}  ERA={era:.2f}")

        if len(missing_jpn_p) > 0:
            total_jpn_ip = round(missing_jpn_p["IP_dec"].sum(), 1)
            print(f"  Japanese pitchers MISSING ({len(missing_jpn_p)}, {total_jpn_ip} IP):")
            for _, mp in missing_jpn_p.sort_values("IP_dec", ascending=False).head(5).iterrows():
                era = mp["ERA_num"] if pd.notna(mp["ERA_num"]) else 0
                print(f"    {mp['player']:10s}  IP={mp['IP_dec']:>5.1f}  ERA={era:.2f}")

        # --- Stan vs Marcel summary for this team-year ---
        n_h_stan = int(yr_h["stan_better"].sum())
        n_p_stan = int(yr_p["stan_better"].sum()) if len(yr_p) > 0 else 0
        rs_err_m = float(yr_h["err_m"].sum())
        rs_err_s = float(yr_h["err_s"].sum())
        ra_err_m = float(yr_p["err_m"].sum()) if len(yr_p) > 0 else 0
        ra_err_s = float(yr_p["err_s"].sum()) if len(yr_p) > 0 else 0

        print(f"\n  Summary:")
        print(f"    Hitters:  Stan better {n_h_stan}/{len(yr_h)}  "
              f"RS_err Marcel={rs_err_m:+.1f}  Stan={rs_err_s:+.1f}")
        print(f"    Pitchers: Stan better {n_p_stan}/{len(yr_p)}  "
              f"RA_err Marcel={ra_err_m:+.1f}  Stan={ra_err_s:+.1f}")

        # Foreign contribution to missing RS/RA
        fgn_miss_rs = float(K_WOBA * missing_fgn_h["wOBA"].fillna(0).values @ missing_fgn_h["PA"].values) if len(missing_fgn_h) > 0 else 0
        jpn_miss_rs = float(K_WOBA * missing_jpn_h["wOBA"].fillna(0).values @ missing_jpn_h["PA"].values) if len(missing_jpn_h) > 0 else 0
        fgn_miss_ra = float((missing_fgn_p["ERA_num"].fillna(0) * missing_fgn_p["IP_dec"] / 9.0).sum()) if len(missing_fgn_p) > 0 else 0
        jpn_miss_ra = float((missing_jpn_p["ERA_num"].fillna(0) * missing_jpn_p["IP_dec"] / 9.0).sum()) if len(missing_jpn_p) > 0 else 0

        print(f"    Missing RS: foreign={fgn_miss_rs:.1f}  japanese={jpn_miss_rs:.1f}")
        print(f"    Missing RA: foreign={fgn_miss_ra:.1f}  japanese={jpn_miss_ra:.1f}")

        results.append({
            "year": yr, "team": tm,
            "actual_W": int(row["W"]),
            "W_marcel": round(float(row["W_marcel"]), 1),
            "W_stan": round(float(row["W_stan"]), 1),
            "err_M": round(float(row["err_M"]), 1),
            "err_S": round(float(row["err_S"]), 1),
            "n_hitters": len(yr_h), "n_pitchers": len(yr_p),
            "n_missing_h": len(missing_h), "n_missing_p": len(missing_p),
            "n_missing_foreign_h": len(missing_fgn_h),
            "n_missing_foreign_p": len(missing_fgn_p),
            "missing_foreign_PA": int(missing_fgn_h["PA"].sum()) if len(missing_fgn_h) > 0 else 0,
            "missing_foreign_IP": round(float(missing_fgn_p["IP_dec"].sum()), 1) if len(missing_fgn_p) > 0 else 0,
            "missing_foreign_RS": round(fgn_miss_rs, 1),
            "missing_foreign_RA": round(fgn_miss_ra, 1),
            "stan_better_h": f"{n_h_stan}/{len(yr_h)}",
            "stan_better_p": f"{n_p_stan}/{len(yr_p)}",
        })

    # --- Overall Stan effectiveness analysis ---
    print("\n" + "=" * 70)
    print("Stan Effectiveness: Where does Stan improve over Marcel?")
    print("=" * 70)

    h_all = h_df.copy()
    h_all["is_foreign"] = h_all["player"].isin(foreign_names)
    h_all["abs_err_m"] = (h_all["actual"] - h_all["marcel"]).abs()
    h_all["abs_err_s"] = (h_all["actual"] - h_all["stan"]).abs()
    h_all["stan_better"] = h_all["abs_err_s"] < h_all["abs_err_m"]

    p_all = p_df.copy()
    p_all["is_foreign"] = p_all["player"].isin(foreign_names)
    p_all["abs_err_m"] = (p_all["actual"] - p_all["marcel"]).abs()
    p_all["abs_err_s"] = (p_all["actual"] - p_all["stan"]).abs()
    p_all["stan_better"] = p_all["abs_err_s"] < p_all["abs_err_m"]

    for label, df, metric in [("Hitters (wOBA)", h_all, "wOBA"),
                                ("Pitchers (ERA)", p_all, "ERA")]:
        print(f"\n  {label}:")
        for group, sub in [("Japanese", df[~df["is_foreign"]]),
                           ("Foreign", df[df["is_foreign"]])]:
            n = len(sub)
            if n == 0:
                continue
            n_stan = int(sub["stan_better"].sum())
            mae_m = float(sub["abs_err_m"].mean())
            mae_s = float(sub["abs_err_s"].mean())
            print(f"    {group:>10s}: n={n:>4d}  Stan wins {n_stan}/{n} ({100*n_stan/n:.1f}%)  "
                  f"MAE Marcel={mae_m:.4f}  Stan={mae_s:.4f}  Δ={mae_s-mae_m:+.4f}")

        # By Marcel error magnitude (quintiles)
        df["marcel_err_abs"] = (df["actual"] - df["marcel"]).abs()
        df["quintile"] = pd.qcut(df["marcel_err_abs"], 5, labels=["Q1(small)", "Q2", "Q3", "Q4", "Q5(large)"])
        print(f"\n    By Marcel error magnitude ({metric}):")
        for q in ["Q1(small)", "Q2", "Q3", "Q4", "Q5(large)"]:
            sub = df[df["quintile"] == q]
            n = len(sub)
            n_stan = int(sub["stan_better"].sum())
            mae_m = float(sub["abs_err_m"].mean())
            mae_s = float(sub["abs_err_s"].mean())
            print(f"      {q:>10s}: n={n:>3d}  Stan wins {n_stan}/{n} ({100*n_stan/n:.1f}%)  "
                  f"MAE M={mae_m:.4f}  S={mae_s:.4f}  Δ={mae_s-mae_m:+.4f}")

    # Save JSON
    out = {"threshold": ERR_THRESHOLD, "big_misses": results}
    out_path = OUT_DIR / "big_miss_diagnosis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
