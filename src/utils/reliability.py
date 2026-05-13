"""
Sample-size-aware reliability weighting for live-season mode.

For 2026 live-season analysis, raw scores computed from small PA samples are
regressed toward the league average using a reliability weight:

    w = PA / (PA + k)          (James-Stein shrinkage, k = 120 by default)
    adjusted_uvs = w * uvs_normalized + (1 - w) * LEAGUE_AVG

This prevents early-season noise from inflating or deflating rankings while
still allowing breakout players to surface as their samples grow.
"""

import pandas as pd
import numpy as np

RELIABILITY_K = 120          # PA denominator constant (regress faster at smaller samples)
LEAGUE_AVG_SCORE = 50.0      # Center point on 0-100 normalized scale


def compute_reliability(pa: pd.Series, k: float = RELIABILITY_K) -> pd.Series:
    """
    Compute reliability weight w = PA / (PA + k).

    Returns values in [0, 1]:
      - 0 PA  → w = 0.00 (fully regressed to league average)
      - 120 PA → w = 0.50 (halfway between raw and average)
      - 350 PA → w = 0.74 (mostly trusting the raw score)
      - 600 PA → w = 0.83
    """
    pa = pa.fillna(0).clip(lower=0)
    return pa / (pa + k)


def normalize_uvs_to_100(uvs: pd.Series) -> pd.Series:
    """Min-max normalize a UVS series (z-score composite) to 0–100 scale."""
    uvs = uvs.fillna(uvs.mean() if uvs.notna().any() else 0)
    lo, hi = uvs.min(), uvs.max()
    if hi > lo:
        return (uvs - lo) / (hi - lo) * 100.0
    return pd.Series(LEAGUE_AVG_SCORE, index=uvs.index)


def apply_reliability_weighting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add reliability columns to a 2026-mode hitter DataFrame.

    Adds three columns:
      - reliability      : w in [0, 1]
      - reliability_pct  : w * 100, displayed as a percentage
      - uvs_normalized   : raw UVS mapped to 0–100
      - adjusted_uvs     : reliability-weighted score (sort key for 2026 leaderboard)

    Args:
        df: DataFrame with 'uvs' and at least one PA column ('pa' or 'PA').

    Returns:
        DataFrame with new columns added (copy).
    """
    df = df.copy()

    # Locate PA column
    pa_col = next((c for c in ['pa', 'PA'] if c in df.columns), None)
    pa = df[pa_col].fillna(0) if pa_col else pd.Series(0, index=df.index)

    df['reliability'] = compute_reliability(pa)
    df['reliability_pct'] = (df['reliability'] * 100).round(1)

    if 'uvs' in df.columns and df['uvs'].notna().any():
        df['uvs_normalized'] = normalize_uvs_to_100(df['uvs'])
    else:
        df['uvs_normalized'] = LEAGUE_AVG_SCORE

    df['adjusted_uvs'] = (
        df['reliability'] * df['uvs_normalized'] +
        (1.0 - df['reliability']) * LEAGUE_AVG_SCORE
    ).round(2)

    return df
