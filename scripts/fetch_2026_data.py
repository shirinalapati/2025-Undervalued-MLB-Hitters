"""
Daily data pipeline for the 2026 MLB live season.

FanGraphs blocks pybaseball with HTTP 403 for 2026, so we assemble all needed
columns from three Baseball Savant (Statcast) endpoints + Baseball Reference:

  Source 1 — statcast_batter_expected_stats(2026)
             → wOBA, xwOBA, xBA, xSLG, PA, BA + luck differentials

  Source 2 — statcast_batter_exitvelo_barrels(2026)
             → Barrel%, HardHit% (ev95%), Exit Velo, Sweet Spot%

  Source 3 — statcast_batter_percentile_ranks(2026)
             → BB%, K%, Chase%, whiff%, hard_hit%, exit_velocity (backup)

  Source 4 — batting_stats_bref(2026)  [Baseball Reference]
             → AB, H, 2B, 3B, HR, R, RBI, BB, SO, OBP, SLG (→ ISO, OPS)

All Statcast tables share player_id → clean merge.
BRef matched by normalised player name.

Applies the same UVS formula as the 2025 project, then adds
reliability weighting on top (w = PA / (PA + 120)).

Output:
  data/processed/comprehensive_stats_2026.csv
  data/processed/last_updated_2026.txt
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV     = DATA_PROCESSED / "comprehensive_stats_2026.csv"
TIMESTAMP_FILE = DATA_PROCESSED / "last_updated_2026.txt"

SEASON = 2026
MIN_PA = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ── Fetch helpers ──────────────────────────────────────────────────────────

def _pyb():
    try:
        import pybaseball as pyb
        pyb.cache.enable()
        return pyb
    except ImportError:
        log.error("pybaseball not installed. Run: pip install pybaseball")
        raise


def fetch_expected_stats(pyb, year: int) -> pd.DataFrame:
    """Statcast expected stats: xwOBA, xBA, xSLG, wOBA, PA, BA, luck diffs."""
    log.info("  Fetching Statcast expected stats…")
    try:
        df = pyb.statcast_batter_expected_stats(year, minPA=1)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "est_woba": "xwoba",
            "est_ba":   "xba",
            "est_slg":  "xslg",
            "est_woba_minus_woba_diff": "xwoba_minus_woba",
            "est_ba_minus_ba_diff":     "xba_minus_ba",
            "est_slg_minus_slg_diff":   "xslg_minus_slg",
        })
        log.info(f"    → {len(df)} rows")
        return df
    except Exception as exc:
        log.warning(f"    Expected stats failed: {exc}")
        return pd.DataFrame()


def fetch_exit_velo(pyb, year: int) -> pd.DataFrame:
    """Statcast exit-velo / barrels: Barrel%, HardHit%, Exit Velo, Sweet Spot%."""
    log.info("  Fetching Statcast exit-velo / barrels…")
    try:
        df = pyb.statcast_batter_exitvelo_barrels(year, minBBE=1)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "brl_percent":          "barrel_batted_rate",
            "ev95percent":          "hard_hit_percent",
            "avg_hit_speed":        "avg_exit_velocity",
            "anglesweetspotpercent":"sweet_spot_percent",
            "brl_pa":               "barrel_pa_rate",
        })
        # Keep only the columns we need + player_id
        keep = ["player_id", "barrel_batted_rate", "hard_hit_percent",
                "avg_exit_velocity", "sweet_spot_percent", "barrel_pa_rate"]
        df = df[[c for c in keep if c in df.columns]]
        log.info(f"    → {len(df)} rows")
        return df
    except Exception as exc:
        log.warning(f"    Exit-velo fetch failed: {exc}")
        return pd.DataFrame()


def fetch_percentile_ranks(pyb, year: int) -> pd.DataFrame:
    """Statcast percentile ranks: BB%, K%, Chase%, whiff%, sprint speed."""
    log.info("  Fetching Statcast percentile ranks…")
    try:
        df = pyb.statcast_batter_percentile_ranks(year)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "chase_percent": "o_swing_percent",
            "k_percent":     "k_percent",
            "bb_percent":    "bb_percent",
            "whiff_percent": "whiff_percent",
            "sprint_speed":  "sprint_speed",
            "oaa":           "oaa",
        })
        keep = ["player_id", "k_percent", "bb_percent", "o_swing_percent",
                "whiff_percent", "sprint_speed", "oaa"]
        df = df[[c for c in keep if c in df.columns]]
        log.info(f"    → {len(df)} rows")
        return df
    except Exception as exc:
        log.warning(f"    Percentile ranks failed: {exc}")
        return pd.DataFrame()


def fetch_bref_batting(pyb, year: int) -> pd.DataFrame:
    """Baseball Reference: counting stats + OBP/SLG for ISO, OPS, wRC+ proxy."""
    log.info("  Fetching Baseball Reference batting stats…")
    try:
        df = pyb.batting_stats_bref(year)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "SO": "k_bref", "BB": "bb_bref",
            "HR": "HR", "R": "R", "RBI": "RBI",
            "AB": "AB", "H": "H", "2B": "2B", "3B": "3B",
            "OBP": "OBP", "BA": "BA_bref", "SB": "SB",
        })
        # Derived stats
        if "OBP" in df.columns and "SLG" in df.columns:
            df["OPS"]  = df["OBP"].fillna(0) + df["SLG"].fillna(0)
            df["ISO"]  = (df["SLG"].fillna(0) - df["BA_bref"].fillna(0)).round(3)
        # Normalise name for merge
        if "Name" in df.columns:
            df["_bref_name"] = df["Name"].str.lower().str.strip()
        keep = ["_bref_name", "HR", "R", "RBI", "AB", "H", "2B", "3B",
                "OBP", "SLG", "ISO", "OPS", "BA_bref", "SB",
                "k_bref", "bb_bref"]
        df = df[[c for c in keep if c in df.columns]]
        log.info(f"    → {len(df)} rows")
        return df
    except Exception as exc:
        log.warning(f"    BRef fetch failed: {exc}")
        return pd.DataFrame()


# ── Merge ──────────────────────────────────────────────────────────────────

def _norm_name(s: pd.Series) -> pd.Series:
    """'Alvarez, Yordan' → 'yordan alvarez'."""
    def _flip(x):
        x = str(x).strip()
        if ", " in x:
            parts = x.split(", ", 1)
            return f"{parts[1]} {parts[0]}".lower()
        return x.lower()
    return s.apply(_flip)


def build_dataset(expected, ev, percentile, bref) -> pd.DataFrame:
    """Merge all sources on player_id (Statcast) then name-match BRef."""
    if expected.empty:
        log.error("No expected stats — cannot build dataset.")
        return pd.DataFrame()

    df = expected.copy()

    # Merge exit-velo on player_id
    if not ev.empty and "player_id" in ev.columns:
        df = df.merge(ev, on="player_id", how="left")
        log.info(f"  After EV merge: {len(df)} rows")

    # Merge percentile ranks on player_id
    if not percentile.empty and "player_id" in percentile.columns:
        df = df.merge(percentile, on="player_id", how="left")
        log.info(f"  After percentile merge: {len(df)} rows")

    # Merge BRef by normalised name
    if not bref.empty and "_bref_name" in bref.columns:
        if "last_name, first_name" in df.columns:
            df["_norm_name"] = _norm_name(df["last_name, first_name"])
        elif "player_name" in df.columns:
            df["_norm_name"] = df["player_name"].str.lower().str.strip()
        else:
            df["_norm_name"] = ""
        df = df.merge(bref, left_on="_norm_name", right_on="_bref_name", how="left")
        df.drop(columns=["_norm_name", "_bref_name"], errors="ignore", inplace=True)
        log.info(f"  After BRef merge: {len(df)} rows")

    return df


# ── Derived columns ────────────────────────────────────────────────────────

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # xISO = xSLG − xBA
    if "xslg" in df.columns and "xba" in df.columns:
        df["xiso"] = (df["xslg"] - df["xba"]).round(3)

    # BA from BRef where wOBA source doesn't have it
    if "BA_bref" in df.columns and "ba" not in df.columns:
        df["ba"] = df["BA_bref"]
    elif "ba" not in df.columns and "ba" in df.columns:
        pass
    # use woba-source ba if available
    if "ba" not in df.columns:
        df["ba"] = np.nan

    # OBP from BRef or estimate from wOBA
    if "OBP" in df.columns:
        df["obp"] = df["OBP"]

    # SLG
    if "SLG" in df.columns:
        df["slg"] = df["SLG"]

    # Unified HR/R/RBI/AB/H
    for upper, lower in [("HR","hr"),("R","r"),("RBI","rbi"),("AB","ab"),("H","h")]:
        if lower not in df.columns and upper in df.columns:
            df[lower] = df[upper]

    # K / BB counting (from BRef)
    if "k" not in df.columns and "k_bref" in df.columns:
        df["k"] = df["k_bref"]
    if "bb" not in df.columns and "bb_bref" in df.columns:
        df["bb"] = df["bb_bref"]

    # BB% / K% from counts if percentile source was empty
    pa = df.get("pa", pd.Series(np.nan, index=df.index))
    if "bb_percent" not in df.columns or df["bb_percent"].isna().all():
        if "bb" in df.columns:
            df["bb_percent"] = (df["bb"] / pa.replace(0, np.nan) * 100).round(1)
    if "k_percent" not in df.columns or df["k_percent"].isna().all():
        if "k" in df.columns:
            df["k_percent"] = (df["k"] / pa.replace(0, np.nan) * 100).round(1)

    # Approximate wRC+ from wOBA (league avg wOBA ≈ 0.316 for 2026)
    LG_WOBA = 0.316
    if "woba" in df.columns and ("wrc_plus" not in df.columns or df["wrc_plus"].isna().all()):
        df["wrc_plus"] = ((df["woba"] / LG_WOBA) * 100).round(1)

    # Player name (unified)
    for col in ["last_name, first_name", "player_name", "Name"]:
        if col in df.columns and df[col].notna().any():
            df["name"] = df[col]
            break
    if "name" not in df.columns:
        df["name"] = "Unknown"
    df["name"] = df["name"].fillna("Unknown")

    df["position_type"] = "Hitter"
    df["season"] = SEASON

    return df


# ── Main ───────────────────────────────────────────────────────────────────

def write_timestamp() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    TIMESTAMP_FILE.write_text(ts)
    log.info(f"Timestamp written: {ts}")


def main() -> None:
    log.info("=" * 60)
    log.info("2026 MLB Live Season Data Pipeline")
    log.info("Sources: Statcast (3 endpoints) + Baseball Reference")
    log.info("=" * 60)

    pyb = _pyb()

    # ── Step 1: Fetch all sources ───────────────────────────────────────
    log.info("Step 1: Fetching data…")
    expected   = fetch_expected_stats(pyb, SEASON)
    ev         = fetch_exit_velo(pyb, SEASON)
    percentile = fetch_percentile_ranks(pyb, SEASON)
    bref       = fetch_bref_batting(pyb, SEASON)

    if expected.empty:
        log.error("Cannot continue without expected stats. Aborting.")
        sys.exit(1)

    # ── Step 2: Merge ───────────────────────────────────────────────────
    log.info("Step 2: Merging sources…")
    df = build_dataset(expected, ev, percentile, bref)

    # PA filter
    if "pa" in df.columns:
        df = df[df["pa"] >= MIN_PA].copy()
    log.info(f"  → {len(df)} hitters with >= {MIN_PA} PA")

    df = add_derived_columns(df)

    # ── Step 3: UVS (same formula as 2025) ─────────────────────────────
    log.info("Step 3: Computing UVS (same formula as 2025)…")
    try:
        from src.utils.uvs_metrics import calculate_all_uvs_metrics
        df = calculate_all_uvs_metrics(df)
        log.info("  UVS computed.")
    except Exception as exc:
        log.warning(f"  UVS skipped: {exc}")

    try:
        from src.utils.tova_metrics import calculate_all_composite_metrics
        df = calculate_all_composite_metrics(df)
        log.info("  TOVA+/BOV computed.")
    except Exception as exc:
        log.warning(f"  TOVA+ skipped: {exc}")

    # ── Step 4: Reliability weighting ──────────────────────────────────
    log.info("Step 4: Applying reliability weighting…")
    try:
        from src.utils.reliability import apply_reliability_weighting
        df = apply_reliability_weighting(df)
        log.info("  Done.")
    except Exception as exc:
        log.warning(f"  Reliability skipped: {exc}")

    # ── Step 5: Sort & rank ─────────────────────────────────────────────
    sort_col = "adjusted_uvs" if "adjusted_uvs" in df.columns else "uvs"
    if sort_col in df.columns and df[sort_col].notna().any():
        df = df.sort_values(sort_col, ascending=False, na_position="last")
    df["undervalued_rank"] = range(1, len(df) + 1)

    # ── Step 6: Save ────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Saved {len(df)} hitters → {OUTPUT_CSV}")
    write_timestamp()

    # Summary
    log.info("=" * 60)
    filled = {c: df[c].notna().sum() for c in
              ["barrel_batted_rate","hard_hit_percent","bb_percent","k_percent",
               "o_swing_percent","wrc_plus","hr","r","rbi"] if c in df.columns}
    for col, cnt in filled.items():
        log.info(f"  {col}: {cnt}/{len(df)} filled")

    top5 = df.head(5)
    name_col = next((c for c in ["name","last_name, first_name"] if c in top5.columns), None)
    if name_col:
        log.info(f"\n  Top 5 ({sort_col}):")
        for _, row in top5.iterrows():
            pa  = int(row["pa"]) if "pa" in row else "?"
            adj = round(row.get(sort_col, 0), 2)
            rel = round(row.get("reliability_pct", 0), 1)
            log.info(f"    {row[name_col]}  PA={pa}  {sort_col}={adj}  rel={rel}%")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
