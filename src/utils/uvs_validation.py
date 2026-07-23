"""
2025 → 2026 follow-up validation for the UVS model.

Compares frozen 2025 full-season scores against 2026 live outcomes for
hitters who appear in both datasets. Used by the dashboard Model Validation tab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BENCHMARK_SEASON = 2025
FOLLOWUP_SEASON = 2026
MIN_PA_2025 = 200
MIN_PA_2026 = 50  # minimum 2026 PA for follow-up stats (reduces noise)

TIER_LABELS_5 = ["Q1 (lowest UVS)", "Q2", "Q3", "Q4", "Q5 (highest UVS)"]
TIER_LABELS_10 = [f"D{i}" for i in range(1, 11)]


def _data_dir(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return root / "data" / "processed"


def load_validation_cohort(
    project_root: Path | None = None,
    min_pa_2026: int = MIN_PA_2026,
) -> pd.DataFrame:
    """
    Merge 2025 qualified hitters with 2026 follow-up stats on player_id.
    Returns one row per player with 2025 scores and 2026 outcomes.
    """
    data_dir = _data_dir(project_root)
    path_2025 = data_dir / f"comprehensive_stats_{BENCHMARK_SEASON}.csv"
    path_2026 = data_dir / f"comprehensive_stats_{FOLLOWUP_SEASON}.csv"

    if not path_2025.exists() or not path_2026.exists():
        return pd.DataFrame()

    df25 = pd.read_csv(path_2025, low_memory=False)
    df26 = pd.read_csv(path_2026)

    if "position_type" in df25.columns:
        df25 = df25[df25["position_type"] == "Hitter"].copy()
    df25 = df25[df25["pa"] >= MIN_PA_2025].copy()
    if "player_id" in df25.columns:
        df25 = df25.sort_values("uvs", ascending=False, na_position="last")
        df25 = df25.drop_duplicates(subset=["player_id"], keep="first")
    df25 = df25.sort_values("uvs", ascending=False, na_position="last").reset_index(drop=True)
    df25["undervalued_rank"] = range(1, len(df25) + 1)

    name_col = next((c for c in ["name", "Name", "last_name, first_name"] if c in df25.columns), None)
    if name_col:
        df25["name"] = df25[name_col]

    keep_25 = ["player_id", "name", "uvs", "undervalued_rank", "woba", "wRC+", "war", "pa"]
    keep_25 = [c for c in keep_25 if c in df25.columns]
    base = df25[keep_25].copy()

    rename_25 = {
        "woba": "woba_2025",
        "wRC+": "wrc_plus_2025",
        "war": "war_2025",
        "pa": "pa_2025",
        "uvs": "uvs_2025",
    }
    base = base.rename(columns={k: v for k, v in rename_25.items() if k in base.columns})

    follow = df26[["player_id", "woba", "wrc_plus", "war", "pa", "uvs"]].rename(
        columns={
            "woba": "woba_2026",
            "wrc_plus": "wrc_plus_2026",
            "war": "war_2026",
            "pa": "pa_2026",
            "uvs": "uvs_2026",
        }
    )
    follow = follow.drop_duplicates(subset=["player_id"], keep="first")

    merged = base.merge(follow, on="player_id", how="inner")
    if min_pa_2026 > 0 and "pa_2026" in merged.columns:
        merged = merged[merged["pa_2026"] >= min_pa_2026].copy()

    merged["delta_woba"] = merged["woba_2026"] - merged["woba_2025"]
    if "wrc_plus_2025" in merged.columns and "wrc_plus_2026" in merged.columns:
        merged["delta_wrc_plus"] = merged["wrc_plus_2026"] - merged["wrc_plus_2025"]
    if "war_2025" in merged.columns and "war_2026" in merged.columns:
        merged["delta_war"] = merged["war_2026"] - merged["war_2025"]

    return merged.reset_index(drop=True)


def assign_uvs_tiers(df: pd.DataFrame, n_tiers: int = 5) -> pd.DataFrame:
    """Add uvs_tier column via equal-count bins on 2025 UVS."""
    out = df.copy()
    if out.empty or "uvs_2025" not in out.columns:
        out["uvs_tier"] = pd.NA
        return out

    labels = TIER_LABELS_5 if n_tiers == 5 else [f"T{i}" for i in range(1, n_tiers + 1)]
    out["uvs_tier"] = pd.qcut(out["uvs_2025"], n_tiers, labels=labels, duplicates="drop")
    return out


def build_tier_calibration(df: pd.DataFrame, n_tiers: int = 5) -> pd.DataFrame:
    """Mean follow-up outcomes by 2025 UVS tier."""
    tiered = assign_uvs_tiers(df, n_tiers)
    if tiered.empty:
        return pd.DataFrame()

    agg: dict[str, Any] = {
        "n": ("player_id", "count"),
        "mean_uvs_2025": ("uvs_2025", "mean"),
        "mean_woba_2025": ("woba_2025", "mean"),
        "mean_woba_2026": ("woba_2026", "mean"),
        "mean_delta_woba": ("delta_woba", "mean"),
    }
    if "uvs_2026" in tiered.columns:
        agg["mean_uvs_2026"] = ("uvs_2026", "mean")
    if "wrc_plus_2025" in tiered.columns:
        agg["mean_wrc_plus_2025"] = ("wrc_plus_2025", "mean")
    if "wrc_plus_2026" in tiered.columns:
        agg["mean_wrc_plus_2026"] = ("wrc_plus_2026", "mean")
    if "delta_wrc_plus" in tiered.columns:
        agg["mean_delta_wrc_plus"] = ("delta_wrc_plus", "mean")

    cal = tiered.groupby("uvs_tier", observed=True).agg(**agg).reset_index()
    for col in cal.columns:
        if col != "uvs_tier" and col != "n":
            cal[col] = cal[col].round(3)
    return cal


def build_baseline_comparison(df: pd.DataFrame, top_n: int = 35) -> pd.DataFrame:
    """
    Compare top-N 2025 UVS picks vs rest of cohort vs bottom quintile.
    """
    if df.empty:
        return pd.DataFrame()

    tiered = assign_uvs_tiers(df, 5)
    bottom_tier = tiered["uvs_tier"].astype(str).str.contains("Q1", na=False)

    groups = {
        f"Top {top_n} by 2025 UVS rank": tiered["undervalued_rank"] <= top_n,
        "All other matched hitters": tiered["undervalued_rank"] > top_n,
        "Bottom UVS quintile (Q1)": bottom_tier,
        "Full matched cohort": pd.Series(True, index=tiered.index),
    }

    rows = []
    for label, mask in groups.items():
        sub = tiered[mask]
        if sub.empty:
            continue
        row: dict[str, Any] = {
            "group": label,
            "n": len(sub),
            "mean_uvs_2025": round(sub["uvs_2025"].mean(), 3),
            "mean_woba_2025": round(sub["woba_2025"].mean(), 3),
            "mean_woba_2026": round(sub["woba_2026"].mean(), 3),
            "mean_delta_woba": round(sub["delta_woba"].mean(), 3),
        }
        if "wrc_plus_2025" in sub.columns:
            row["mean_wrc_plus_2025"] = round(sub["wrc_plus_2025"].mean(), 1)
        if "wrc_plus_2026" in sub.columns:
            row["mean_wrc_plus_2026"] = round(sub["wrc_plus_2026"].mean(), 1)
        if "delta_wrc_plus" in sub.columns:
            row["mean_delta_wrc_plus"] = round(sub["delta_wrc_plus"].mean(), 1)
        rows.append(row)

    return pd.DataFrame(rows)


def build_top_picks_table(df: pd.DataFrame, top_n: int = 35) -> pd.DataFrame:
    """Player-level follow-up for the highest 2025 UVS ranks."""
    if df.empty:
        return pd.DataFrame()

    cols = [
        "undervalued_rank",
        "name",
        "uvs_2025",
        "uvs_2026",
        "woba_2025",
        "wrc_plus_2025",
        "woba_2026",
        "wrc_plus_2026",
        "delta_woba",
        "delta_wrc_plus",
        "pa_2026",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df.nsmallest(top_n, "undervalued_rank")[cols].copy()
    for c in out.columns:
        if c not in ("name",):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.round(3)


def validation_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Headline stats for the validation intro."""
    if df.empty:
        return {"n_matched": 0}

    top35 = df[df["undervalued_rank"] <= 35]
    rest = df[df["undervalued_rank"] > 35]

    summary: dict[str, Any] = {
        "n_matched": len(df),
        "n_top35": len(top35),
        "corr_uvs_delta_woba": round(df["uvs_2025"].corr(df["delta_woba"]), 3),
        "mean_delta_woba_all": round(df["delta_woba"].mean(), 3),
        "mean_delta_woba_top35": round(top35["delta_woba"].mean(), 3) if len(top35) else None,
        "mean_delta_woba_rest": round(rest["delta_woba"].mean(), 3) if len(rest) else None,
    }
    if "delta_wrc_plus" in df.columns:
        summary["corr_uvs_delta_wrc"] = round(df["uvs_2025"].corr(df["delta_wrc_plus"]), 3)
        summary["mean_delta_wrc_top35"] = round(top35["delta_wrc_plus"].mean(), 1) if len(top35) else None
        summary["mean_delta_wrc_rest"] = round(rest["delta_wrc_plus"].mean(), 1) if len(rest) else None
    if "wrc_plus_2026" in df.columns:
        summary["corr_uvs_wrc_2026"] = round(df["uvs_2025"].corr(df["wrc_plus_2026"]), 3)
    if "uvs_2026" in df.columns:
        summary["corr_uvs_2025_2026"] = round(df["uvs_2025"].corr(df["uvs_2026"]), 3)
    return summary
