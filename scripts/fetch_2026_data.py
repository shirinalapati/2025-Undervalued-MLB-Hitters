"""
Daily data pipeline for the 2026 MLB live season.

Fetches current-season Statcast + FanGraphs data, calculates UVS and reliability
weighting, then saves the results to:
    data/processed/comprehensive_stats_2026.csv
    data/processed/last_updated_2026.txt

Designed to be run as a daily scheduled task (GitHub Actions cron or
PythonAnywhere scheduled tasks).

Usage:
    python scripts/fetch_2026_data.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV  = DATA_PROCESSED / "comprehensive_stats_2026.csv"
TIMESTAMP_FILE = DATA_PROCESSED / "last_updated_2026.txt"

# ── Config ─────────────────────────────────────────────────────────────────
SEASON = 2026
MIN_PA_FOR_DISPLAY = 10       # Show players with as few as 10 PA (reliability handles noise)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _import_pybaseball():
    try:
        import pybaseball as pyb
        pyb.cache.enable()
        return pyb
    except ImportError:
        log.error("pybaseball is not installed. Run: pip install pybaseball")
        raise


def fetch_statcast_expected(year: int, pyb) -> pd.DataFrame:
    """Pull Statcast expected-stats leaderboard (xwOBA, xBA, xSLG, Barrel%, …)."""
    log.info(f"Fetching Statcast expected stats for {year}…")
    try:
        # minPA=1 so we capture every batter; reliability weighting handles small samples
        df = pyb.statcast_batter_expected_stats(year, minPA=MIN_PA_FOR_DISPLAY)
        if df is None or df.empty:
            log.warning("Statcast expected stats returned no data.")
            return pd.DataFrame()
        log.info(f"  → {len(df)} hitters from Statcast")
        return df
    except Exception as exc:
        log.error(f"Statcast fetch failed: {exc}")
        return pd.DataFrame()


def fetch_fangraphs_batting(year: int, pyb) -> pd.DataFrame:
    """Pull FanGraphs batting leaderboard (WAR, wRC+, BB%, K%, discipline…)."""
    log.info(f"Fetching FanGraphs batting stats for {year}…")
    try:
        # qual=0 → no PA minimum filter from FanGraphs side
        df = pyb.batting_stats(year, qual=0)
        if df is None or df.empty:
            log.warning("FanGraphs batting stats returned no data.")
            return pd.DataFrame()
        log.info(f"  → {len(df)} hitters from FanGraphs")
        return df
    except Exception as exc:
        log.error(f"FanGraphs fetch failed: {exc}")
        return pd.DataFrame()


def merge_datasets(statcast_df: pd.DataFrame, fg_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Statcast and FanGraphs frames on player name."""
    if statcast_df.empty:
        return fg_df if not fg_df.empty else pd.DataFrame()
    if fg_df.empty:
        return statcast_df

    # Normalise names for fuzzy join
    def _norm(s):
        return s.str.lower().str.strip() if hasattr(s, "str") else ""

    if "last_name, first_name" in statcast_df.columns:
        statcast_df["_merge_name"] = statcast_df["last_name, first_name"].apply(
            lambda x: " ".join(reversed(str(x).split(", "))) if pd.notna(x) and ", " in str(x) else str(x)
        )
        statcast_df["_merge_name"] = _norm(statcast_df["_merge_name"])
    elif "Name" in statcast_df.columns:
        statcast_df["_merge_name"] = _norm(statcast_df["Name"])
    else:
        return statcast_df

    if "Name" in fg_df.columns:
        fg_df["_merge_name"] = _norm(fg_df["Name"])
    else:
        return statcast_df

    merged = pd.merge(statcast_df, fg_df, on="_merge_name", how="left", suffixes=("", "_fg"))
    merged.drop(columns=["_merge_name"], errors="ignore", inplace=True)
    log.info(f"  → merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


def compute_derived_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Add xISO, BA, salary column, and mark position type."""
    df = df.copy()

    # xISO = xSLG − xBA
    xslg = df.get("est_slg", df.get("xslg", pd.Series(np.nan, index=df.index)))
    xba  = df.get("est_ba",  df.get("xba",  pd.Series(np.nan, index=df.index)))
    if "xiso" not in df.columns:
        df["xiso"] = xslg - xba

    # BA if missing
    if "ba" not in df.columns and "BA" not in df.columns:
        h  = df.get("H",  pd.Series(np.nan, index=df.index))
        ab = df.get("AB", pd.Series(np.nan, index=df.index))
        df["ba"] = (h / ab.replace(0, np.nan)).round(3)

    # Salary column name
    salary_col = f"salary_{year}"
    if salary_col not in df.columns:
        for alt in [f"salary_{year}_x", "salary", "Salary ($M)"]:
            if alt in df.columns:
                df[salary_col] = df[alt]
                break
        else:
            df[salary_col] = np.nan

    # WAR / $1M
    war_col = "WAR" if "WAR" in df.columns else "war"
    if war_col in df.columns and salary_col in df.columns:
        valid = df[salary_col].notna() & (df[salary_col] > 0)
        df["war_per_salary"] = np.where(
            valid,
            df[war_col] / (df[salary_col] + 0.1),
            np.nan
        )

    # Mark as Hitter (this pipeline is hitters-only)
    df["position_type"] = "Hitter"
    df["season"] = year

    return df


def filter_minimum_pa(df: pd.DataFrame, min_pa: int = MIN_PA_FOR_DISPLAY) -> pd.DataFrame:
    """Keep batters with at least `min_pa` plate appearances."""
    pa_col = next((c for c in ["pa", "PA"] if c in df.columns), None)
    if pa_col is None:
        return df
    before = len(df)
    df = df[df[pa_col] >= min_pa].copy()
    log.info(f"  PA filter ({min_pa}+): {before} → {len(df)} hitters")
    return df


def compute_uvs_and_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate UVS composite score and reliability-adjusted score."""
    try:
        from src.utils.uvs_metrics import calculate_all_uvs_metrics
        df = calculate_all_uvs_metrics(df)
        log.info("  UVS metrics calculated.")
    except Exception as exc:
        log.warning(f"  UVS calculation skipped: {exc}")

    try:
        from src.utils.tova_metrics import calculate_all_composite_metrics
        df = calculate_all_composite_metrics(df)
        log.info("  TOVA+/BOV metrics calculated.")
    except Exception as exc:
        log.warning(f"  Composite metrics skipped: {exc}")

    try:
        from src.utils.reliability import apply_reliability_weighting
        df = apply_reliability_weighting(df)
        log.info("  Reliability weighting applied.")
    except Exception as exc:
        log.warning(f"  Reliability weighting skipped: {exc}")

    # Rank by adjusted_uvs for live mode; fall back to uvs
    sort_col = "adjusted_uvs" if "adjusted_uvs" in df.columns else "uvs"
    if sort_col in df.columns and df[sort_col].notna().any():
        df = df.sort_values(sort_col, ascending=False, na_position="last")
        df["undervalued_rank"] = range(1, len(df) + 1)

    return df


def write_timestamp() -> None:
    """Write current UTC time to the last-updated file."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    TIMESTAMP_FILE.write_text(ts)
    log.info(f"  Timestamp written: {ts}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info(f"2026 MLB Live Season Data Pipeline")
    log.info("=" * 60)

    pyb = _import_pybaseball()

    statcast_df = fetch_statcast_expected(SEASON, pyb)
    fg_df       = fetch_fangraphs_batting(SEASON, pyb)
    df          = merge_datasets(statcast_df, fg_df)

    if df.empty:
        log.error("No data collected. Aborting.")
        sys.exit(1)

    df = filter_minimum_pa(df, MIN_PA_FOR_DISPLAY)
    df = compute_derived_columns(df, SEASON)
    df = compute_uvs_and_reliability(df)

    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Saved {len(df)} hitters → {OUTPUT_CSV}")

    write_timestamp()

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
