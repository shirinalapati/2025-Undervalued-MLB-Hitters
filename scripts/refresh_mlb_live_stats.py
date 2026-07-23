#!/usr/bin/env python3
"""
Near-live refresh: MLB Stats API → merge into existing 2026 CSV.

Updates PA, HR, R, RBI, BA, OBP, SLG, OPS, BABIP, BB%, K%, and recalculates
UVS using preserved Statcast advanced metrics (xwOBA, Barrel%, HardHit%, etc.).

Run frequently (every 30–60 min). Full Statcast scrape stays on fetch_2026_data.py.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.mlb_stats_api import fetch_live_hitting, merge_live_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

SEASON = 2026
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "comprehensive_stats_2026.csv"
TIMESTAMP_FILE = PROJECT_ROOT / "data" / "processed" / "last_updated_mlb_live_2026.txt"


def write_timestamp() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    TIMESTAMP_FILE.write_text(ts)
    log.info("MLB live timestamp: %s", ts)


def main() -> int:
    if not OUTPUT_CSV.exists():
        log.error("No base CSV at %s — run scripts/fetch_2026_data.py first.", OUTPUT_CSV)
        return 1

    import pandas as pd

    log.info("Loading existing dataset…")
    df = pd.read_csv(OUTPUT_CSV, low_memory=False)
    if df.empty:
        log.error("Existing CSV is empty.")
        return 1

    log.info("Fetching MLB Stats API (season %s)…", SEASON)
    player_ids = pd.to_numeric(df.get("player_id"), errors="coerce").dropna().astype(int).tolist()
    mlb = fetch_live_hitting(SEASON, player_ids=player_ids)
    if mlb.empty:
        log.error("MLB API returned no rows.")
        return 1

    df = merge_live_stats(df, mlb)

    # Reuse derived-column logic from full Statcast pipeline
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from fetch_2026_data import add_derived_columns  # noqa: WPS433

    df = add_derived_columns(df)

    log.info("Recomputing UVS (Statcast advanced stats preserved)…")
    try:
        from src.utils.uvs_metrics import calculate_all_uvs_metrics

        df = calculate_all_uvs_metrics(df)
    except Exception as exc:
        log.warning("UVS recompute skipped: %s", exc)

    try:
        from src.utils.tova_metrics import calculate_all_composite_metrics

        df = calculate_all_composite_metrics(df)
    except Exception as exc:
        log.warning("TOVA+ skipped: %s", exc)

    if "uvs" in df.columns and df["uvs"].notna().any():
        df = df.sort_values("uvs", ascending=False, na_position="last")
    df["undervalued_rank"] = range(1, len(df) + 1)

    df.to_csv(OUTPUT_CSV, index=False)
    write_timestamp()
    log.info("Saved %d hitters → %s", len(df), OUTPUT_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
