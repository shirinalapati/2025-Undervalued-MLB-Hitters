"""
Daily data pipeline for the 2026 MLB live season.

Uses the EXACT same data sources and UVS formula as the 2025 project:
  - Statcast (Baseball Savant): xwOBA, xBA, xSLG, Barrel%, HardHit%, Exit Velo, Sweet Spot%
  - FanGraphs: WAR, wRC+, BB%, K%, Chase%, Z-Contact%, GB/FB/LD%, OPS, ISO, R, RBI
  - Salary data (CSV / Spotrac)

The only differences from 2025:
  - year = 2026
  - min_pa = 10  (no hard cutoff; reliability weighting handles small-sample noise)
  - Saves to data/processed/comprehensive_stats_2026.csv
  - Writes data/processed/last_updated_2026.txt timestamp

Run daily via GitHub Actions (.github/workflows/update_2026_data.yml).

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

OUTPUT_CSV     = DATA_PROCESSED / "comprehensive_stats_2026.csv"
TIMESTAMP_FILE = DATA_PROCESSED / "last_updated_2026.txt"

SEASON = 2026
MIN_PA = 10   # Show all active batters; reliability weighting handles noise

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def write_timestamp() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    TIMESTAMP_FILE.write_text(ts)
    log.info(f"Timestamp written: {ts}")


def main() -> None:
    log.info("=" * 60)
    log.info(f"2026 MLB Live Season Data Pipeline")
    log.info(f"Using identical formula and data sources as 2025 project")
    log.info("=" * 60)

    # ── Step 1: Fetch data using the same pipeline as 2025 ─────────────
    log.info("Step 1: Fetching Statcast + FanGraphs data...")
    from src.data_pipeline.fetch_advanced_metrics import combine_all_data
    df = combine_all_data(year=SEASON, min_pa=MIN_PA)

    if df is None or df.empty:
        log.error("No data returned. Check network connection and pybaseball installation.")
        sys.exit(1)

    log.info(f"  → {len(df)} total records fetched")

    # Keep hitters only (pipeline also fetches pitchers; we only need hitters here)
    if "position_type" in df.columns:
        df = df[df["position_type"] == "Hitter"].copy()
        log.info(f"  → {len(df)} hitters after position filter")

    # ── Step 2: Calculate advanced metrics (same as 2025) ──────────────
    log.info("Step 2: Calculating advanced metrics (UVS, TOVA+, etc.)...")
    try:
        from src.utils.metrics import calculate_all_advanced_metrics
        df = calculate_all_advanced_metrics(df)
        log.info("  Advanced metrics calculated.")
    except Exception as exc:
        log.warning(f"  Advanced metrics skipped: {exc}")

    try:
        from src.utils.uvs_metrics import calculate_all_uvs_metrics
        df = calculate_all_uvs_metrics(df)
        log.info("  UVS calculated.")
    except Exception as exc:
        log.warning(f"  UVS skipped: {exc}")

    try:
        from src.utils.tova_metrics import calculate_all_composite_metrics
        df = calculate_all_composite_metrics(df)
        log.info("  TOVA+/BOV calculated.")
    except Exception as exc:
        log.warning(f"  TOVA+ skipped: {exc}")

    # ── Step 3: Apply reliability weighting (2026-specific) ────────────
    log.info("Step 3: Applying reliability weighting...")
    try:
        from src.utils.reliability import apply_reliability_weighting
        df = apply_reliability_weighting(df)
        log.info("  Reliability weighting applied.")
        if "reliability_pct" in df.columns:
            log.info(f"  Reliability stats: "
                     f"min={df['reliability_pct'].min():.1f}%  "
                     f"mean={df['reliability_pct'].mean():.1f}%  "
                     f"max={df['reliability_pct'].max():.1f}%")
    except Exception as exc:
        log.warning(f"  Reliability weighting skipped: {exc}")

    # ── Step 4: Sort and rank ───────────────────────────────────────────
    sort_col = "adjusted_uvs" if "adjusted_uvs" in df.columns else "uvs"
    if sort_col in df.columns and df[sort_col].notna().any():
        df = df.sort_values(sort_col, ascending=False, na_position="last")
    df["undervalued_rank"] = range(1, len(df) + 1)
    df["season"] = SEASON

    # ── Step 5: Save ────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Saved {len(df)} hitters → {OUTPUT_CSV}")
    write_timestamp()

    # ── Summary ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Pipeline complete.")
    if "pa" in df.columns:
        log.info(f"  PA range: {int(df['pa'].min())} – {int(df['pa'].max())}")
    if sort_col in df.columns:
        top5 = df.head(5)
        name_col = next((c for c in ["name", "Name", "last_name, first_name"] if c in top5.columns), None)
        if name_col:
            log.info("  Top 5 by adjusted UVS:")
            for _, row in top5.iterrows():
                pa_val = int(row["pa"]) if "pa" in row else "?"
                score  = round(row[sort_col], 2) if sort_col in row else "?"
                log.info(f"    {row[name_col]}  (PA={pa_val}, {sort_col}={score})")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
