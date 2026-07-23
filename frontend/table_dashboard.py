"""
Undervalued MLB Hitters Dashboard — dual-mode (2025 full season / 2026 live season).

Tabs:
  0. About This Resource — app overview + metrics glossary
  1. About This Page   — season-specific methodology
  2. All Players Stats — full sortable/filterable table with CSV export
  3. Undervalued Players — UVS leaderboard
  4. Model Validation  — 2025 UVS vs 2026 follow-up outcomes

Season modes:
  2025 Full Season  → static benchmark; ≥200 PA; ranks by raw UVS (unchanged)
  2026 Live Season  → all hitters; same UVS formula & table format as 2025
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Paths & config ─────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))

PROJECT_ROOT  = Path(__file__).parent.parent
DATA_DIR      = PROJECT_ROOT / "data" / "processed"

BENCHMARK_SEASON  = 2025
CURRENT_SEASON    = 2026
MIN_PA_2025       = 200   # original 2025 filter; pipeline already pre-filters
LOW_SAMPLE_PA     = 50    # floor for 2026 low-sample flag (early season)
LOW_SAMPLE_PA_RATIO = 0.35  # flag PA below 35% of league median (updates each refresh)
MAX_PLAYERS_2025  = 350
MAX_PLAYERS_2026  = 520   # headroom above ~514 live hitters

SEASON_OPTIONS = [
    {"label": "2025 Full Season", "value": 2025},
    {"label": "2026 Live Season", "value": 2026},
]

# In-memory cache: season -> (csv_mtime, dataframe)
_SEASON_CACHE: dict[int, tuple[float, pd.DataFrame]] = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
from src.utils.live_refresh import (  # noqa: E402
    is_enabled as live_refresh_enabled,
    mlb_refresh_interval_label,
    poll_interval_ms,
    pull_latest_committed_data,
    refresh_interval_label,
    refresh_interval_seconds,
    refresh_status_label,
    start_background_scheduler,
)
from src.utils.uvs_validation import (  # noqa: E402
    build_baseline_comparison,
    build_tier_calibration,
    build_top_picks_table,
    load_validation_cohort,
    validation_summary,
)


def coerce_season(val) -> int:
    """dcc.Store / network payloads may stringify ints — always compare as int."""
    if val is None:
        return BENCHMARK_SEASON
    try:
        return int(val)
    except (TypeError, ValueError):
        return BENCHMARK_SEASON


# ── App init ───────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server          # WSGI entry-point for PythonAnywhere / Render
app.title = "Undervalued MLB Hitters"


# ── Data loading ────────────────────────────────────────────────────────────

def _csv_path(season: int) -> Path | None:
    candidates = [
        DATA_DIR / f"comprehensive_stats_{season}.csv",
        PROJECT_ROOT / "data" / "processed" / f"comprehensive_stats_{season}.csv",
        Path(os.getcwd()) / "data" / "processed" / f"comprehensive_stats_{season}.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def _last_updated(season: int) -> str:
    if season == BENCHMARK_SEASON:
        return "2025 Full Season (final)"
    full_ts = DATA_DIR / f"last_updated_{season}.txt"
    mlb_ts = DATA_DIR / f"last_updated_mlb_live_{season}.txt"
    for base in (DATA_DIR, Path(os.getcwd()) / "data" / "processed"):
        full_path = base / f"last_updated_{season}.txt"
        mlb_path = base / f"last_updated_mlb_live_{season}.txt"
        if full_path.exists() or mlb_path.exists():
            full_ts, mlb_ts = full_path, mlb_path
            break
    parts = []
    if mlb_ts.exists():
        parts.append(f"PA live: {mlb_ts.read_text().strip()}")
    if full_ts.exists():
        parts.append(f"Advanced (Statcast): {full_ts.read_text().strip()}")
    return "  ·  ".join(parts) if parts else "Unknown"


def load_season_data(season: int) -> pd.DataFrame:
    """
    Load processed CSV for the given season.
    2025: ≥200 PA, frozen from CSV (no metric recompute).
    2026: all hitters, same UVS formula, cached for fast search/tab switches.
    """
    season = coerce_season(season)
    path = _csv_path(season)
    if path is None:
        return pd.DataFrame()

    mtime = path.stat().st_mtime
    cached = _SEASON_CACHE.get(season)
    if cached and cached[0] == mtime:
        return cached[1].copy()

    df = pd.read_csv(path)

    if "position_type" in df.columns:
        df = df[df["position_type"] == "Hitter"].copy()

    pa_col = next((c for c in ["pa", "PA"] if c in df.columns), None)
    if season == BENCHMARK_SEASON and pa_col:
        df = df[df[pa_col] >= MIN_PA_2025].copy()

    needs_uvs = "uvs" not in df.columns or df["uvs"].isna().all()
    if needs_uvs:
        try:
            from src.utils.uvs_metrics import calculate_all_uvs_metrics
            df = calculate_all_uvs_metrics(df)
        except Exception:
            pass

    df = _fill_missing_columns(df, season)

    if "uvs" in df.columns and df["uvs"].notna().any():
        df = df.sort_values("uvs", ascending=False, na_position="last")
    df["undervalued_rank"] = range(1, len(df) + 1)

    _SEASON_CACHE[season] = (mtime, df.copy())
    return df.copy()


def _fill_missing_columns(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Fill / map derived columns needed by the table.
    Handles both FanGraphs-style names (Barrel%, HardHit%, Pull%, etc.)
    and lowercase/underscore names so 2025 and 2026 both display correctly.
    """
    df = df.copy()

    # ── Player name ────────────────────────────────────────────────────
    for col in ["last_name, first_name", "Name", "name"]:
        if col in df.columns and df[col].notna().any():
            df["name"] = df[col]
            break
    if "name" not in df.columns:
        df["name"] = "Unknown"
    df["name"] = df["name"].fillna("Unknown")

    # ── Season year (for live table label) ─────────────────────────────
    if "year" not in df.columns or df["year"].isna().all():
        for alt in ("Season", "season"):
            if alt in df.columns and df[alt].notna().any():
                df["year"] = pd.to_numeric(df[alt], errors="coerce")
                break
    if "year" not in df.columns:
        df["year"] = season

    # ── Expected stats ─────────────────────────────────────────────────
    for col in ["est_woba", "xwoba", "xwOBA"]:
        if col in df.columns and df[col].notna().any():
            df["xwoba"] = df[col]; break
    for src, dst in [("est_ba","xba"), ("est_slg","xslg")]:
        if src in df.columns and (dst not in df.columns or df[dst].isna().all()):
            df[dst] = df[src]
    if "xiso" not in df.columns or df["xiso"].isna().all():
        if "xslg" in df.columns and "xba" in df.columns:
            df["xiso"] = (df["xslg"] - df["xba"]).round(3)

    # ── Contact quality (FanGraphs names → standard names) ────────────
    # Barrel%
    if df.get("barrel_batted_rate", pd.Series()).isna().all():
        for alt in ["Barrel%", "barrel_percent"]:
            if alt in df.columns and df[alt].notna().any():
                df["barrel_batted_rate"] = df[alt]; break
    # HardHit%
    if df.get("hard_hit_percent", pd.Series()).isna().all():
        for alt in ["HardHit%", "Hard%"]:
            if alt in df.columns and df[alt].notna().any():
                df["hard_hit_percent"] = df[alt]; break
    # Exit Velo
    if "avg_exit_velocity" not in df.columns or df["avg_exit_velocity"].isna().all():
        for alt in ["exit_velocity", "launch_speed", "avg_hit_speed"]:
            if alt in df.columns and df[alt].notna().any():
                df["avg_exit_velocity"] = df[alt]; break

    # ── Plate discipline (FanGraphs → standard) ────────────────────────
    for fg, std in [("BB%","bb_percent"), ("K%","k_percent"),
                    ("O-Swing%","o_swing_percent"), ("Z-Contact%","z_contact_percent"),
                    ("Contact%","contact_percent")]:
        if std not in df.columns or df[std].isna().all():
            if fg in df.columns and df[fg].notna().any():
                df[std] = df[fg]

    # 2026: coalesce Savant + Statcast aggregate columns
    if season == CURRENT_SEASON:
        for target, sources in [
            ("z_contact_percent", ["z_contact_percent", "z_contact_percent_sc", "iz_contact_percent"]),
            ("pull_percent", ["pull_percent", "pull_percent_sc"]),
            ("o_swing_percent", ["o_swing_percent"]),
            ("contact_percent", ["contact_percent"]),
            ("gb_percent", ["gb_percent"]),
            ("fb_percent", ["fb_percent"]),
            ("ld_percent", ["ld_percent"]),
            ("oppo_percent", ["oppo_percent"]),
        ]:
            for src in sources:
                if src in df.columns and df[src].notna().any():
                    if target not in df.columns or df[target].isna().all():
                        df[target] = df[src]
                    else:
                        df[target] = df[target].fillna(df[src])
                    break
        pa_s = df.get("pa", df.get("PA"))
        if pa_s is not None:
            pa_num = pd.to_numeric(pa_s, errors="coerce")
            bb_cnt = df.get("bb", df.get("bb_bref", df.get("BB")))
            k_cnt = df.get("k", df.get("k_bref", df.get("SO")))
            if bb_cnt is not None:
                missing_bb = "bb_percent" not in df.columns or df["bb_percent"].isna().sum() > len(df) * 0.2
                if missing_bb:
                    df["bb_percent"] = (pd.to_numeric(bb_cnt, errors="coerce") / pa_num.replace(0, np.nan) * 100).round(1)
            if k_cnt is not None:
                missing_k = "k_percent" not in df.columns or df["k_percent"].isna().sum() > len(df) * 0.2
                if missing_k:
                    df["k_percent"] = (pd.to_numeric(k_cnt, errors="coerce") / pa_num.replace(0, np.nan) * 100).round(1)

    # ── Batted ball profile (FanGraphs → standard) ─────────────────────
    for fg, std in [("GB%","gb_percent"), ("FB%","fb_percent"), ("LD%","ld_percent"),
                    ("Pull%","pull_percent"), ("Oppo%","oppo_percent")]:
        if std not in df.columns or df[std].isna().all():
            if fg in df.columns and df[fg].notna().any():
                df[std] = df[fg]

    # ── Run production ─────────────────────────────────────────────────
    for fg, std in [("wRC+","wrc_plus"), ("WAR","war")]:
        if std not in df.columns or df[std].isna().all():
            if fg in df.columns and df[fg].notna().any():
                df[std] = df[fg]

    # ── Counting stats (uppercase → lowercase) ──────────────────────────
    for upper, lower in [("AB","ab"),("H","h"),("R","r"),("RBI","rbi"),
                          ("HR","hr"),("BB","bb"),("OBP","obp"),
                          ("SLG","slg"),("ISO","iso"),("OPS","ops"),
                          ("BABIP","babip"),("WAR","war")]:
        if lower not in df.columns or df[lower].isna().all():
            if upper in df.columns and df[upper].notna().any():
                df[lower] = df[upper]

    # BA (various sources)
    if "ba" not in df.columns or df["ba"].isna().all():
        for alt in ["BA", "BA_bref"]:
            if alt in df.columns and df[alt].notna().any():
                df["ba"] = df[alt]; break
        if ("ba" not in df.columns or df["ba"].isna().all()) and "H" in df.columns and "AB" in df.columns:
            df["ba"] = (df["H"] / df["AB"].replace(0, np.nan)).round(3)

    # K (strikeouts) — prefer SO over the sparsely-populated K column
    if "k" not in df.columns or df["k"].isna().sum() > len(df) * 0.5:
        for alt in ["SO", "K", "so", "k_bref"]:
            if alt in df.columns and df[alt].notna().sum() > df.get("k", pd.Series()).notna().sum():
                df["k"] = df[alt]; break

    # xHR — Statcast rarely ships est_hr; estimate from HR and expected vs actual SLG
    if "xhr" not in df.columns or df["xhr"].isna().all():
        for alt in ["est_hr", "xHR", "expected_hr"]:
            if alt in df.columns and df[alt].notna().any():
                df["xhr"] = df[alt]
                break
        else:
            hr = df.get("HR", df.get("hr"))
            xslg = df.get("est_slg", df.get("xslg"))
            slg = df.get("SLG", df.get("slg"))
            if hr is not None and xslg is not None and slg is not None:
                valid = hr.notna() & xslg.notna() & slg.notna() & (slg > 0)
                df["xhr"] = np.nan
                df.loc[valid, "xhr"] = (hr[valid] * (xslg[valid] / slg[valid])).round(2)

    # BABIP = (H − HR) / (AB − K − HR + SF)
    if "babip" not in df.columns or df["babip"].isna().all():
        h = df.get("H", df.get("h"))
        hr = df.get("HR", df.get("hr"))
        ab = df.get("AB", df.get("ab"))
        k = df.get("SO", df.get("k_bref", df.get("k")))
        sf = df.get("SF", pd.Series(0, index=df.index))
        if h is not None and hr is not None and ab is not None and k is not None:
            so = pd.to_numeric(k, errors="coerce").fillna(0)
            sf = pd.to_numeric(sf, errors="coerce").fillna(0)
            denom = pd.to_numeric(ab, errors="coerce") - so - pd.to_numeric(hr, errors="coerce") + sf
            df["babip"] = ((pd.to_numeric(h, errors="coerce") - pd.to_numeric(hr, errors="coerce"))
                           / denom.replace(0, np.nan)).round(3)

    # ISO / OPS — compute when uppercase copies are absent
    if "iso" not in df.columns or df["iso"].isna().any():
        if "ISO" in df.columns:
            df["iso"] = df["iso"].fillna(df["ISO"]) if "iso" in df.columns else df["ISO"]
        elif "slg" in df.columns and "ba" in df.columns:
            computed = (df["slg"] - df["ba"]).round(3)
            df["iso"] = df["iso"].fillna(computed) if "iso" in df.columns else computed
    if "ops" not in df.columns or df["ops"].isna().any():
        if "OPS" in df.columns:
            df["ops"] = df["ops"].fillna(df["OPS"]) if "ops" in df.columns else df["OPS"]
        elif "obp" in df.columns and "slg" in df.columns:
            computed = (df["obp"] + df["slg"]).round(3)
            df["ops"] = df["ops"].fillna(computed) if "ops" in df.columns else computed

    # ── Salary column ──────────────────────────────────────────────────
    salary_col = f"salary_{season}"
    if salary_col not in df.columns:
        df[salary_col] = np.nan
    for alt in [f"salary_{season}_x", "salary_2025", "salary_2025_x", "salary"]:
        if alt in df.columns:
            df[salary_col] = df[salary_col].fillna(df[alt])
    df["salary_display"] = df[salary_col]

    # ── WAR / $1M ──────────────────────────────────────────────────────
    war_s = df.get("war", df.get("WAR"))
    sal_s = df[salary_col]
    if war_s is not None and sal_s is not None:
        valid = sal_s.notna() & (sal_s > 0) & war_s.notna()
        if "war_per_salary" not in df.columns:
            df["war_per_salary"] = np.nan
        df.loc[valid, "war_per_salary"] = (
            war_s[valid] / (sal_s[valid] + 0.1)
        ).round(3)

    if season == CURRENT_SEASON:
        df = _fill_sparse_sample_defaults(df)

    return df


def _fill_sparse_sample_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing table stats for tiny samples / zero balls in play (2026 live)."""
    if "bip" in df.columns:
        bip = pd.to_numeric(df["bip"], errors="coerce").fillna(0)
    else:
        bip = pd.Series(0, index=df.index)
    ab = pd.to_numeric(df.get("ab", df.get("AB")), errors="coerce").fillna(0)
    pa = pd.to_numeric(df.get("pa", df.get("PA")), errors="coerce").fillna(0)
    h = pd.to_numeric(df.get("h", df.get("H")), errors="coerce").fillna(0)
    sparse = bip.le(0) | pa.lt(5)

    for col in [
        "barrel_batted_rate", "hard_hit_percent", "avg_exit_velocity",
        "sweet_spot_percent", "woba", "xwoba", "xba", "xslg", "xiso", "xhr",
        "z_contact_percent", "contact_percent", "o_swing_percent",
        "gb_percent", "fb_percent", "ld_percent", "pull_percent", "oppo_percent",
        "babip", "wrc_plus",
    ]:
        if col in df.columns:
            df.loc[sparse, col] = df.loc[sparse, col].fillna(0.0)

    if "ba" in df.columns:
        need_ba = df["ba"].isna() & ab.gt(0)
        df.loc[need_ba, "ba"] = (h[need_ba] / ab[need_ba]).round(3)
        df.loc[sparse, "ba"] = df.loc[sparse, "ba"].fillna(0.0)
        df.loc[ab.eq(0), "ba"] = df.loc[ab.eq(0), "ba"].fillna(0.0)

    if "slg" in df.columns:
        if "SLG" in df.columns:
            df["slg"] = df["slg"].fillna(df["SLG"])
        no_hit = sparse | (h.eq(0) & ab.gt(0))
        df.loc[no_hit, "slg"] = df.loc[no_hit, "slg"].fillna(0.0)
        df.loc[ab.eq(0), "slg"] = df.loc[ab.eq(0), "slg"].fillna(0.0)

    if "iso" in df.columns and "slg" in df.columns and "ba" in df.columns:
        need_iso = df["iso"].isna() & df["slg"].notna() & df["ba"].notna()
        df.loc[need_iso, "iso"] = (df.loc[need_iso, "slg"] - df.loc[need_iso, "ba"]).round(3)
        df.loc[sparse, "iso"] = df.loc[sparse, "iso"].fillna(0.0)

    if "ops" in df.columns and "obp" in df.columns and "slg" in df.columns:
        need_ops = df["ops"].isna() & df["obp"].notna() & df["slg"].notna()
        df.loc[need_ops, "ops"] = (df.loc[need_ops, "obp"] + df.loc[need_ops, "slg"]).round(3)
        df.loc[sparse, "ops"] = df.loc[sparse, "ops"].fillna(0.0)

    return df


def apply_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Filter rows by player name (case-insensitive, substring, regex-safe)."""
    if query is None or not str(query).strip():
        return df
    if "name" not in df.columns:
        return df
    parts = [p for p in str(query).strip().lower().split() if p]
    if not parts:
        return df
    col = df["name"].fillna("").astype(str)
    mask = pd.Series(True, index=df.index)
    for p in parts:
        mask &= col.str.lower().str.contains(p, regex=False, na=False)
    return df[mask]


def _data_availability_notice(season: int):
    """2026-only note for columns that require FanGraphs (blocked for live season)."""
    if coerce_season(season) != CURRENT_SEASON:
        return html.Div()
    return dbc.Alert([
        html.Strong("2026 data note — "),
        "Live 2026 stats use Statcast, Baseball Savant, and Baseball Reference. ",
        html.Strong("WAR"), " comes from BRef (bwar_bat). ",
        html.Strong("Salary"), " uses 2026 BRef when available, then 2025 contract data, "
        "prior-year BRef, or league minimum for pre-arb players. ",
        "Discipline and batted-ball rates come from Savant leaderboards and pitch-level Statcast.",
    ], color="info", className="mb-3")


def _low_sample_pa_threshold(df: pd.DataFrame, season: int) -> tuple[int, int]:
    """
    Season-adaptive low-sample PA cutoff for 2026.

    Uses max(50, 35% of league median PA) so the flag tightens as the season
    progresses — e.g. 75 PA may be fine in May but flagged in September.
    """
    if coerce_season(season) != CURRENT_SEASON:
        return MIN_PA_2025, MIN_PA_2025

    pa = pd.to_numeric(df.get("pa"), errors="coerce").dropna()
    if pa.empty:
        return LOW_SAMPLE_PA, 0

    regular = pa[pa >= 20]
    median_pa = int(round(regular.median())) if len(regular) else int(round(pa.median()))
    threshold = max(LOW_SAMPLE_PA, int(round(median_pa * LOW_SAMPLE_PA_RATIO)))
    return threshold, median_pa


def _sample_size_notice(season: int, df: pd.DataFrame | None = None):
    """2026-only caution about interpreting UVS for hitters with few PA/AB."""
    if coerce_season(season) != CURRENT_SEASON:
        return html.Div()

    threshold, median_pa = (LOW_SAMPLE_PA, 0)
    if df is not None and not df.empty:
        threshold, median_pa = _low_sample_pa_threshold(df, season)
    elif coerce_season(season) == CURRENT_SEASON:
        try:
            full = load_season_data(CURRENT_SEASON)
            if not full.empty:
                threshold, median_pa = _low_sample_pa_threshold(full, season)
        except Exception:
            pass
    return dbc.Alert([
        html.Strong("Sample size caution — "),
        "The 2026 view includes every hitter with Statcast data (no PA minimum). ",
        "A hitter with very little playing time can show an extreme UVS that is mostly noise, ",
        "not a reliable signal. Rows highlighted in ",
        html.Strong("yellow"), " are below the current low-sample cutoff — fewer than ",
        html.Strong(f"{threshold} PA"), " right now (about ",
        html.Strong("35% of the league median"), f", currently {median_pa} PA). ",
        "That threshold ", html.Strong("rises automatically"), " each time data refreshes, ",
        "so a player with 75 PA may look fine early in the year but be flagged by September ",
        "when regulars have 400+ PA. Always check the ",
        html.Strong("PA"), " column before drawing conclusions, and treat yellow rows with caution ",
        "until a hitter accumulates a fuller season of playing time.",
    ], color="warning", className="mb-3")


# ── Column definitions ──────────────────────────────────────────────────────

def _hitter_columns():
    """Shared column layout for both 2025 and 2026 tables."""
    return [
        {"name": "Rank",        "id": "undervalued_rank",    "type": "numeric"},
        {"name": "Player",      "id": "name",                "type": "text"},
        {"name": "UVS",         "id": "uvs",                 "type": "numeric"},
        {"name": "PA",          "id": "pa",                  "type": "numeric"},
        # Contact quality
        {"name": "Barrel%",     "id": "barrel_batted_rate",  "type": "numeric"},
        {"name": "HardHit%",    "id": "hard_hit_percent",    "type": "numeric"},
        {"name": "Exit Velo",   "id": "avg_exit_velocity",   "type": "numeric"},
        {"name": "Sweet Spot%", "id": "sweet_spot_percent",  "type": "numeric"},
        # Expected
        {"name": "wOBA",        "id": "woba",                "type": "numeric"},
        {"name": "xwOBA",       "id": "xwoba",               "type": "numeric"},
        {"name": "xBA",         "id": "xba",                 "type": "numeric"},
        {"name": "xSLG",        "id": "xslg",                "type": "numeric"},
        {"name": "xISO",        "id": "xiso",                "type": "numeric"},
        {"name": "xHR",         "id": "xhr",                 "type": "numeric"},
        # Discipline
        {"name": "BB%",         "id": "bb_percent",          "type": "numeric"},
        {"name": "K%",          "id": "k_percent",           "type": "numeric"},
        {"name": "Chase%",      "id": "o_swing_percent",     "type": "numeric"},
        {"name": "Z-Contact%",  "id": "z_contact_percent",   "type": "numeric"},
        {"name": "Contact%",    "id": "contact_percent",     "type": "numeric"},
        # Batted ball
        {"name": "GB%",         "id": "gb_percent",          "type": "numeric"},
        {"name": "FB%",         "id": "fb_percent",          "type": "numeric"},
        {"name": "LD%",         "id": "ld_percent",          "type": "numeric"},
        {"name": "Pull%",       "id": "pull_percent",        "type": "numeric"},
        {"name": "Oppo%",       "id": "oppo_percent",        "type": "numeric"},
        # Value
        {"name": "wRC+",        "id": "wrc_plus",            "type": "numeric"},
        {"name": "WAR",         "id": "war",                 "type": "numeric"},
        {"name": "WAR/$1M",     "id": "war_per_salary",      "type": "numeric"},
        {"name": "Salary ($M)", "id": "salary_display",      "type": "numeric"},
        # Traditional
        {"name": "AB",  "id": "ab",    "type": "numeric"},
        {"name": "H",   "id": "h",     "type": "numeric"},
        {"name": "R",   "id": "r",     "type": "numeric"},
        {"name": "RBI", "id": "rbi",   "type": "numeric"},
        {"name": "HR",  "id": "hr",    "type": "numeric"},
        {"name": "BB",  "id": "bb",    "type": "numeric"},
        {"name": "K",   "id": "k",     "type": "numeric"},
        {"name": "BA",  "id": "ba",    "type": "numeric"},
        {"name": "OBP", "id": "obp",   "type": "numeric"},
        {"name": "SLG", "id": "slg",   "type": "numeric"},
        {"name": "ISO", "id": "iso",   "type": "numeric"},
        {"name": "OPS", "id": "ops",   "type": "numeric"},
        {"name": "BABIP","id": "babip","type": "numeric"},
    ]


# ── Table column formatting ──────────────────────────────────────────────────

_PCT_IDS = {
    "barrel_batted_rate","hard_hit_percent","sweet_spot_percent",
    "bb_percent","k_percent","o_swing_percent","z_contact_percent",
    "contact_percent","gb_percent","fb_percent","ld_percent",
    "pull_percent","oppo_percent",
}
_THREE_DP  = {"uvs","woba","xwoba","xba","xslg","xiso","ba","obp","slg","iso","ops","babip"}
_TWO_DP    = {"war","war_per_salary","salary_display","xhr"}
_ONE_DP    = {"avg_exit_velocity","exit_velocity"}
_ZERO_DP   = {"undervalued_rank","year","pa","ab","h","r","rbi","hr","bb","k","wrc_plus"}


def _format_col(col_id: str, col_name: str) -> dict:
    cfg = {"name": col_name, "id": col_id}
    if col_id in _PCT_IDS:
        cfg.update({"type": "numeric", "format": {"specifier": ".1f"}})
    elif col_id in _THREE_DP:
        cfg.update({"type": "numeric", "format": {"specifier": ".3f"}})
    elif col_id in _TWO_DP:
        cfg.update({"type": "numeric", "format": {"specifier": ".2f"}})
    elif col_id in _ONE_DP:
        cfg.update({"type": "numeric", "format": {"specifier": ".1f"}})
    elif col_id in _ZERO_DP:
        cfg.update({"type": "numeric", "format": {"specifier": ".0f"}})
    else:
        cfg.update({"type": "numeric", "format": {"specifier": ".2f"}})
    return cfg


# ── Column aliases (FanGraphs/Statcast raw names → table IDs) ───────────────

_ALIASES = {
    "barrel_batted_rate": ["Barrel%", "barrel_batted_rate", "barrel_percent"],
    "hard_hit_percent":   ["HardHit%", "hard_hit_percent", "Hard%"],
    "avg_exit_velocity":  ["avg_exit_velocity", "exit_velocity", "launch_speed"],
    "sweet_spot_percent": ["sweet_spot_percent", "Sweet Spot%"],
    "bb_percent":         ["BB%", "bb_percent"],
    "k_percent":          ["K%", "k_percent"],
    "o_swing_percent":    ["O-Swing%", "o_swing_percent", "chase_percent"],
    "z_contact_percent":  ["Z-Contact%", "z_contact_percent", "iz_contact_percent", "z_contact_percent_sc"],
    "contact_percent":    ["Contact%", "contact_percent"],
    "pull_percent":       ["Pull%", "pull_percent", "pull_percent_sc"],
    "oppo_percent":       ["Oppo%", "oppo_percent"],
    "gb_percent":         ["GB%", "gb_percent"],
    "fb_percent":         ["FB%", "fb_percent"],
    "ld_percent":         ["LD%", "ld_percent"],
    "wrc_plus":           ["wRC+", "wrc_plus"],
    "xwoba":              ["est_woba", "xwoba", "xwOBA"],
    "xba":                ["est_ba",  "xba",  "xBA"],
    "xslg":               ["est_slg", "xslg", "xSLG"],
}

_PCT_DECIMAL_IDS = {
    "barrel_batted_rate","hard_hit_percent","sweet_spot_percent",
    "bb_percent","k_percent","o_swing_percent","z_contact_percent",
    "contact_percent","gb_percent","fb_percent","ld_percent",
    "pull_percent","oppo_percent",
}


def _build_table_records(df: pd.DataFrame, col_defs: list) -> list:
    """Vectorized record builder — avoids per-row Python loops."""
    if df.empty:
        return []

    col_ids = [c["id"] for c in col_defs]
    working = df.copy()

    for cid in col_ids:
        if cid in working.columns and working[cid].notna().any():
            continue
        for alt in _ALIASES.get(cid, []):
            if alt in working.columns and working[alt].notna().any():
                working[cid] = working[alt]
                break

    for cid in _PCT_DECIMAL_IDS:
        if cid not in working.columns:
            continue
        s = pd.to_numeric(working[cid], errors="coerce")
        mask = s.notna() & (s >= 0) & (s < 1.5)
        working.loc[mask, cid] = s[mask] * 100

    for cid in col_ids:
        if cid not in working.columns:
            working[cid] = np.nan

    subset = working[col_ids].copy()
    for cid in ("undervalued_rank", "year", "pa", "ab", "h", "r", "rbi", "hr", "bb", "k", "wrc_plus"):
        if cid in subset.columns:
            subset[cid] = pd.to_numeric(subset[cid], errors="coerce")

    records = subset.where(subset.notna(), None).to_dict("records")
    for rec in records:
        for k, v in list(rec.items()):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                rec[k] = ""
            elif k == "undervalued_rank" and v != "":
                rec[k] = int(v)
            elif hasattr(v, "item"):
                rec[k] = v.item()
    return records


# ── DataTable builder ───────────────────────────────────────────────────────

def build_datatable(df: pd.DataFrame, season: int, table_id: str = "stats-datatable") -> dash_table.DataTable:
    col_defs = _hitter_columns()
    formatted_cols = [_format_col(c["id"], c["name"]) for c in col_defs]
    records = _build_table_records(df, col_defs)

    style_data_conditional = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        {"if": {"column_id": "name"}, "fontWeight": "bold", "minWidth": "160px", "maxWidth": "210px"},
        {"if": {"column_id": "undervalued_rank"}, "textAlign": "center", "fontWeight": "bold"},
        {"if": {"column_id": "uvs"}, "color": "#1a7340", "fontWeight": "bold"},
    ]
    if coerce_season(season) == CURRENT_SEASON:
        low_pa, _ = _low_sample_pa_threshold(df, season)
        style_data_conditional.extend([
            {
                "if": {"filter_query": f"{{pa}} < {low_pa}"},
                "backgroundColor": "#fff8e6",
            },
            {
                "if": {"filter_query": f"{{pa}} < {low_pa}", "column_id": "pa"},
                "color": "#b85c00",
                "fontWeight": "bold",
            },
        ])

    return dash_table.DataTable(
        id=table_id,
        columns=formatted_cols,
        data=records,
        style_cell={
            "textAlign": "center",
            "padding": "8px 10px",
            "fontFamily": "Arial, sans-serif",
            "fontSize": "13px",
            "border": "1px solid #dee2e6",
            "minWidth": "70px",
            "maxWidth": "130px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_cell_conditional=[
            {"if": {"column_id": "name"}, "textAlign": "left"},
        ],
        style_header={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "center",
            "fontSize": "12px",
            "border": "1px solid #1a252f",
            "whiteSpace": "normal",
            "height": "auto",
            "padding": "10px 6px",
        },
        style_data={"backgroundColor": "white", "color": "black", "border": "1px solid #dee2e6"},
        style_data_conditional=style_data_conditional,
        style_table={"overflowX": "auto", "maxHeight": "780px", "minWidth": "100%"},
        sort_action="native",
        filter_action="native",
        page_action="native",
        page_current=0,
        page_size=50,
        fixed_rows={"headers": True},
        export_format="csv",
        export_headers="display",
        tooltip_duration=None,
    )


# ── Metrics glossary (shared) ────────────────────────────────────────────────

def _metrics_glossary_card():
    """Full metrics glossary — used on About This Resource and the toggle elsewhere."""
    p = {"fontSize": "14px", "lineHeight": "1.7", "marginBottom": "8px"}
    return dbc.Card([dbc.CardBody([
        html.H5("Metrics Glossary", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.H6("Contact Quality & Power", className="mt-2 mb-2"),
                html.P([html.Strong("Barrel %"),
                        " — % of batted balls hit with ideal exit velocity + launch angle "
                        "(Statcast \"barrels\")."], style=p),
                html.P([html.Strong("Hard Hit %"),
                        " — % of batted balls hit ≥ 95 mph Exit Velocity."], style=p),
                html.P([html.Strong("Exit Velocity"),
                        " — Average speed (mph) of all batted balls."], style=p),
                html.P([html.Strong("Sweet Spot %"),
                        " — % of batted balls with launch angle 8–32°."], style=p),
                html.P([html.Strong("xHR"),
                        " — Expected home runs based on launch angle + Exit Velocity "
                        "of each batted ball."], style=p),
                html.H6("Expected Performance (Statcast)", className="mt-4 mb-2"),
                html.P([html.Strong("wOBA"),
                        " — Weighted On-Base Avg = "
                        "(0.69×BB + 0.89×1B + 1.27×2B + 1.62×3B + 2.10×HR) / PA."], style=p),
                html.P([html.Strong("xwOBA"),
                        " — Expected wOBA using quality of contact and Ks/BBs, "
                        "not actual outcomes."], style=p),
                html.P([html.Strong("xBA"),
                        " — Expected batting average based on exit velocity + launch angle."], style=p),
                html.P([html.Strong("xSLG"),
                        " — Expected slugging % from contact quality."], style=p),
                html.P([html.Strong("xISO"),
                        " — xSLG − xBA; expected isolated power."], style=p),
                html.H6("Plate Discipline", className="mt-4 mb-2"),
                html.P([html.Strong("BB %"), " — Walks / Plate Appearances."], style=p),
                html.P([html.Strong("K %"), " — Strikeouts / Plate Appearances."], style=p),
                html.P([html.Strong("Chase % (O-Swing %)"),
                        " — % of swings at pitches outside strike zone."], style=p),
                html.P([html.Strong("Z-Contact %"),
                        " — % of swings on in-zone pitches that make contact."], style=p),
                html.P([html.Strong("Contact %"),
                        " — % of all swings that make contact."], style=p),
            ], md=6),
            dbc.Col([
                html.H6("Batted-Ball Profile", className="mt-2 mb-2"),
                html.P([html.Strong("GB %"),
                        " — % of batted balls that are grounders."], style=p),
                html.P([html.Strong("FB %"), " — % that are fly balls."], style=p),
                html.P([html.Strong("LD %"), " — % that are line drives."], style=p),
                html.P([html.Strong("Pull %"),
                        " — % of balls hit to pull side."], style=p),
                html.P([html.Strong("Oppo %"),
                        " — % hit to opposite field."], style=p),
                html.H6("Run Production / Value", className="mt-4 mb-2"),
                html.P([html.Strong("wRC+"),
                        " — Weighted Runs Created Plus; 100 = league avg "
                        "(offense adjusted for park/league)."], style=p),
                html.P([html.Strong("WAR"),
                        " — Wins Above Replacement; total value in wins over "
                        "replacement player."], style=p),
                html.P([html.Strong("WAR/$1M"),
                        " — WAR divided by salary (in millions); efficiency per "
                        "payroll dollar."], style=p),
                html.P([html.Strong("Salary ($M)"),
                        " — Player salary in millions."], style=p),
                html.H6("Traditional Counting Stats", className="mt-4 mb-2"),
                html.P([html.Strong("PA"), " — Plate appearances."], style=p),
                html.P([html.Strong("AB"),
                        " — At-bats (PA minus walks, HBP, sac flies/bunts, etc.)."], style=p),
                html.P([html.Strong("H"), " — Hits."], style=p),
                html.P([html.Strong("R"), " — Runs scored."], style=p),
                html.P([html.Strong("RBI"), " — Runs batted in."], style=p),
                html.P([html.Strong("HR"), " — Home runs."], style=p),
                html.P([html.Strong("BB"), " — Walks drawn."], style=p),
                html.P([html.Strong("K"), " — Strikeouts."], style=p),
                html.H6("Rate / Slash-Line Stats", className="mt-4 mb-2"),
                html.P([html.Strong("BA"), " — Batting Avg = H / AB."], style=p),
                html.P([html.Strong("OBP"),
                        " — On-Base % = (H + BB + HBP) / (AB + BB + HBP + SF)."], style=p),
                html.P([html.Strong("SLG"),
                        " — Slugging % = Total Bases / AB."], style=p),
                html.P([html.Strong("ISO"),
                        " — Isolated Power = SLG − BA."], style=p),
                html.P([html.Strong("OPS"),
                        " — On-Base + Slugging = OBP + SLG."], style=p),
                html.P([html.Strong("BABIP"),
                        " — Batting Avg on Balls in Play = "
                        "(H − HR) / (AB − K − HR + SF)."], style=p),
            ], md=6),
        ]),
    ])], className="mb-3")


# ── Layout ──────────────────────────────────────────────────────────────────

app.layout = dbc.Container([

    # ── Hidden stores ────────────────────────────────────────────────────
    dcc.Store(id="season-store", data=BENCHMARK_SEASON),
    dcc.Interval(
        id="data-poll-interval",
        interval=poll_interval_ms(),
        n_intervals=0,
    ),

    # ── Header ───────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.H1(id="app-title", children="Undervalued MLB Hitters Analysis",
                    className="text-center mb-1",
                    style={"color": "#1a1a1a", "fontWeight": "700"}),
            html.Div(id="app-subtitle", className="text-center text-muted mb-1",
                     style={"fontSize": "15px"}),
            html.Div(id="last-updated-display", className="text-center mb-3",
                     style={"fontSize": "12px", "color": "#888"}),
            html.Hr(),
        ])
    ]),

    dcc.Store(id="active-tab-store", data="about-resource"),

    # ── Sidebar + main panel ─────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            dbc.Nav([
                dbc.NavLink("About This Resource", id="tab-about-resource",
                            href="#", active=True, className="mb-1"),
                dbc.NavLink("About This Page", id="tab-about",
                            href="#", className="mb-1"),
                dbc.NavLink("All Players Stats", id="tab-all-players",
                            href="#", className="mb-1"),
                dbc.NavLink("Undervalued Players", id="tab-undervalued",
                            href="#", className="mb-1"),
                dbc.NavLink("Model Validation", id="tab-validation",
                            href="#", className="mb-1"),
            ], vertical=True, pills=True, className="flex-column"),
        ], md=2, lg=2, className="mb-3"),

        dbc.Col([
            html.Div(id="controls-wrapper", style={"display": "none"}, children=[
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Season:", className="fw-bold mb-1",
                                           style={"fontSize": "13px"}),
                                dcc.Dropdown(
                                    id="season-dropdown",
                                    options=SEASON_OPTIONS,
                                    value=BENCHMARK_SEASON,
                                    clearable=False,
                                    style={"fontSize": "13px"},
                                ),
                            ], md=2),
                            dbc.Col([
                                html.Label("Search Player:", className="fw-bold mb-1",
                                           style={"fontSize": "13px"}),
                                dbc.Input(
                                    id="player-search",
                                    type="search",
                                    placeholder="Search by name (e.g. Ohtani, Judge, Soto…)",
                                    debounce=True,
                                    style={"fontSize": "13px"},
                                ),
                                html.Small(id="search-result-count", className="text-muted"),
                            ], md=3),
                            dbc.Col([
                                html.Label("Number of Players:", className="fw-bold mb-1",
                                           style={"fontSize": "13px"}),
                                dcc.Slider(
                                    id="top-n-slider",
                                    min=10, max=MAX_PLAYERS_2025, step=10,
                                    value=MAX_PLAYERS_2025,
                                    marks={10: "10", 50: "50", 100: "100",
                                           200: "200", MAX_PLAYERS_2025: "All 350"},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ], md=7),
                        ], align="center"),
                    ])
                ], className="mb-3"),
            ]),

            html.Div(id="glossary-wrapper", style={"display": "none"}, children=[
                dbc.Collapse(
                    id="metrics-glossary-collapse", is_open=False,
                    children=_metrics_glossary_card(),
                ),
                dbc.Button("Show / Hide Metrics Glossary",
                           id="metrics-glossary-toggle",
                           color="secondary", outline=True, className="mb-3",
                           style={"fontSize": "13px"}),
                html.Hr(),
            ]),

            dcc.Loading(id="loading", type="default",
                        children=html.Div(id="loading-output")),
            html.Div(id="main-content"),
        ], md=10, lg=10),
    ]),

    # ── Footer ────────────────────────────────────────────────────────────
    html.Hr(),
    dbc.Row([dbc.Col([
        html.P("Data: Baseball Savant (Statcast) · FanGraphs · pybaseball  |  "
               "Analysis: Undervaluation Score (UVS) — advanced statistical model",
               className="text-center text-muted small")
    ])], className="mt-3 mb-4"),

], fluid=True, style={"maxWidth": "1900px", "padding": "20px"})




# ── Callbacks ───────────────────────────────────────────────────────────────

@app.callback(
    Output("metrics-glossary-collapse", "is_open"),
    Input("metrics-glossary-toggle", "n_clicks"),
    State("metrics-glossary-collapse", "is_open"),
)
def toggle_glossary(n, is_open):
    return not is_open if n else is_open


@app.callback(
    Output("active-tab-store", "data"),
    Output("tab-about-resource", "active"),
    Output("tab-about", "active"),
    Output("tab-all-players", "active"),
    Output("tab-undervalued", "active"),
    Output("tab-validation", "active"),
    Input("tab-about-resource", "n_clicks"),
    Input("tab-about", "n_clicks"),
    Input("tab-all-players", "n_clicks"),
    Input("tab-undervalued", "n_clicks"),
    Input("tab-validation", "n_clicks"),
    State("active-tab-store", "data"),
    prevent_initial_call=True,
)
def switch_tab(n_res, n_about, n_all, n_uv, n_val, current):
    tab_map = {
        "tab-about-resource": "about-resource",
        "tab-about": "about",
        "tab-all-players": "all-players",
        "tab-undervalued": "undervalued",
        "tab-validation": "validation",
    }
    triggered = dash.callback_context.triggered_id
    active = tab_map.get(triggered, current or "about-resource")
    return (
        active,
        active == "about-resource",
        active == "about",
        active == "all-players",
        active == "undervalued",
        active == "validation",
    )


@app.callback(
    Output("controls-wrapper", "style"),
    Output("glossary-wrapper", "style"),
    Input("active-tab-store", "data"),
)
def toggle_tab_chrome(active_tab):
    if active_tab in ("about-resource", "validation"):
        return {"display": "none"}, {"display": "none"}
    return {}, {}


@app.callback(
    Output("season-store", "data"),
    Output("top-n-slider", "max"),
    Output("top-n-slider", "value"),
    Output("top-n-slider", "marks"),
    Input("season-dropdown", "value"),
)
def sync_season(val):
    season = coerce_season(val)
    if season == CURRENT_SEASON:
        top = MAX_PLAYERS_2026
        marks = {10: "10", 50: "50", 100: "100", 200: "200",
                 350: "350", top: "All"}
    else:
        top = MAX_PLAYERS_2025
        marks = {10: "10", 50: "50", 100: "100", 200: "200", top: "All 350"}
    return season, top, top, marks


@app.callback(
    [Output("main-content", "children"),
     Output("app-title", "children"),
     Output("app-subtitle", "children"),
     Output("last-updated-display", "children"),
     Output("loading-output", "children"),
     Output("search-result-count", "children")],
    [Input("top-n-slider", "value"),
     Input("active-tab-store", "data"),
     Input("season-store", "data"),
     Input("player-search", "value"),
     Input("data-poll-interval", "n_intervals")],
)
def update_main_content(top_n, active_tab, season, search_query, _poll_n):
    season = coerce_season(season)
    top_n  = top_n or MAX_PLAYERS_2025

    title = (
        "Live Undervalued MLB Hitters Analysis"
        if season == CURRENT_SEASON
        else "Undervalued MLB Hitters Analysis"
    )

    if season == CURRENT_SEASON:
        subtitle = (
            "🔴 2026 Live Season  |  All hitters  |  Same UVS formula as 2025  |  "
            + refresh_status_label()
        )
    else:
        subtitle = "2025 Full Season  |  All hitters with ≥ 200 PA  |  Full-season benchmark"

    lu = _last_updated(season)
    last_upd_parts = [
        html.Span("Last updated: ", style={"fontWeight": "600"}),
        html.Span(lu),
    ]
    if season == CURRENT_SEASON and live_refresh_enabled():
        last_upd_parts.extend([
            html.Span("  ·  ", style={"color": "#aaa"}),
            html.Span(
                f"Dashboard checks for new data every {poll_interval_ms() // 60_000} min",
                style={"color": "#666"},
            ),
        ])
    last_upd = html.Span(last_upd_parts)

    if active_tab == "about-resource":
        resource_title = "Live Undervalued MLB Hitters Analysis"
        resource_subtitle = (
            "A live hitting value index for the 2026 MLB season · "
            "Full 2025 benchmark included"
        )
        mlb_label = mlb_refresh_interval_label()
        refresh_label = refresh_interval_label()
        poll_min = poll_interval_ms() // 60_000
        resource_last_upd = html.Span([
            html.Span("Live through the ", style={"fontWeight": "600"}),
            html.Span("2026 regular season"),
            html.Span("  ·  ", style={"color": "#aaa"}),
            html.Span(
                f"PA live every {mlb_label} · Statcast advanced every {refresh_label} · "
                f"UI checks every {poll_min} min"
            ),
        ])
        return (
            _about_resource_content(),
            resource_title,
            resource_subtitle,
            resource_last_upd,
            "",
            "",
        )

    if active_tab == "about":
        return _about_content(season), title, subtitle, last_upd, "", ""

    if active_tab == "validation":
        val_title = "Undervalued MLB Hitters — Model Validation"
        val_subtitle = (
            "2025 full-season UVS vs 2026 live follow-up  |  "
            "Out-of-sample check on the same formula cohort"
        )
        val_last_upd = html.Span([
            html.Span("Follow-up data: ", style={"fontWeight": "600"}),
            html.Span(_last_updated(CURRENT_SEASON)),
            html.Span("  ·  ", style={"color": "#aaa"}),
            html.Span("Benchmark scores frozen from 2025 Full Season"),
        ])
        return _validation_tab(), val_title, val_subtitle, val_last_upd, "", ""

    try:
        df = load_season_data(season)
    except Exception as exc:
        err = dbc.Alert(f"Error loading data: {exc}", color="danger")
        return err, title, subtitle, last_upd, "", ""

    if df.empty:
        msg = (
            f"No 2026 data yet. Run:  python scripts/fetch_2026_data.py"
            if season == CURRENT_SEASON
            else "No data found. Please run the data pipeline."
        )
        return dbc.Alert(msg, color="warning"), title, subtitle, last_upd, "", ""

    total_before_search = len(df)
    df = apply_search(df, search_query)
    search_count = ""
    if search_query and str(search_query).strip():
        search_count = f"  {len(df)} of {total_before_search} hitters match"

    df = df.head(min(top_n, len(df)))

    if active_tab == "all-players":
        content = _all_players_tab(df, season)
    elif active_tab == "undervalued":
        content = _undervalued_tab(df, season)
    else:
        content = _about_content(season)

    return content, title, subtitle, last_upd, "", search_count


# ── Tab renderers ────────────────────────────────────────────────────────────

def _about_resource_content():
    """App-wide overview — no season filters; includes full metrics glossary."""
    p = {"fontSize": "16px", "lineHeight": "1.85", "marginBottom": "22px"}
    refresh_label = refresh_interval_label()
    mlb_label = mlb_refresh_interval_label()
    poll_min = poll_interval_ms() // 60_000
    return dbc.Container([
        dbc.Row([dbc.Col([
            dbc.Card([dbc.CardBody([
                html.P([
                    "This dashboard uses a custom ",
                    html.Strong("Undervaluation Score (UVS)"),
                    " to evaluate MLB hitters through a formula that includes advanced Statcast "
                    "and traditional metrics — expected performance, contact quality, plate "
                    "discipline, run production, salary efficiency, and luck adjustment. The goal "
                    "is to surface hitters whose underlying skill exceeds their box-score results "
                    "and pay, helping you spot players who may be undervalued relative to their "
                    "true talent."
                ], style=p),
                html.P([
                    "But UVS is not only a discovery tool for overlooked names. Because the "
                    "formula rewards genuine offensive excellence, it also ranks established "
                    "stars near the top — giving you a way to appreciate the greatness of "
                    "already well-known hitters. In that sense, this app doubles as a ",
                    html.Strong("true hitting value index"),
                    ": a single framework for comparing every qualified hitter, from breakout "
                    "candidates to household names."
                ], style=p),
                html.P([
                    "The app is intended to run ",
                    html.Strong("live throughout the 2026 MLB regular season"),
                    ". Counting stats (PA, HR, R, RBI, BA, OBP, SLG) refresh from the ",
                    html.Strong("MLB Stats API"),
                    " every ",
                    html.Strong(mlb_label),
                    ". Advanced Statcast metrics used in UVS — xwOBA, Barrel%, HardHit%, ",
                    "Chase%, Z-Contact%, spray splits, and more — refresh every ",
                    html.Strong(refresh_label),
                    ". The dashboard checks for file updates every ",
                    html.Strong(f"{poll_min} minutes"),
                    ". Switch to ",
                    html.Strong("2026 Live Season"),
                    " to follow the current leaderboard as it evolves."
                ], style=p),
                html.P([
                    "Full leaderboards for the ",
                    html.Strong("2025 season"),
                    " are also included as a complete benchmark — every hitter with at least "
                    "200 plate appearances, ranked with the same formula and table format. "
                    "Use it to see how players performed over a full year and to compare "
                    "against the live 2026 view as the new season progresses."
                ], style=p),
                html.P(
                    "For season-specific methodology, sample-size notes, and formula details, "
                    "open About This Page and select either the 2025 Full Season or 2026 Live "
                    "Season from the season dropdown. The All Players Stats and Undervalued "
                    "Players tabs contain the full sortable leaderboards that have all the "
                    "statistics included in the formula, while the Undervalued Players tab "
                    "contains the UVS rankings for either the 2025 or 2026 Live season. "
                    "The Model Validation tab tracks how 2025 UVS scores lined up with "
                    "2026 follow-up performance — calibration charts, baselines, and "
                    "player-level outcomes for intellectual honesty about what the model "
                    "did and did not predict.",
                    style={**p, "marginBottom": "28px"},
                ),
                _metrics_glossary_card(),
            ])], className="mb-4"),
        ], width=12)]),
    ], fluid=True)


def _about_content(season: int):
    if coerce_season(season) == CURRENT_SEASON:
        return dbc.Container([dbc.Row([dbc.Col([dbc.Card([dbc.CardBody([
            html.P([
                "Welcome to the ", html.Strong("2026 Live Season"), " view. This dashboard tracks every MLB "
                "hitter in real time as the 2026 season unfolds, using the ", html.Strong("same UVS formula"),
                " and table format as the 2025 full-season benchmark. There is no plate-appearance minimum — "
                "all hitters with Statcast data are included."
            ], style={"fontSize": "16px", "lineHeight": "1.8", "marginBottom": "20px"}),
            _sample_size_notice(CURRENT_SEASON),
            _data_availability_notice(CURRENT_SEASON),
            html.P([
                "Using advanced statistics, the ", html.Strong("Undervaluation Score (UVS)"),
                " identifies hitters whose underlying performance — expected metrics, contact quality, "
                "and efficiency — exceeds their surface stats and salary."
            ], style={"fontSize": "16px", "lineHeight": "1.8", "marginBottom": "15px"}),
            html.P([html.Strong("UVS = 0.25·EPI + 0.20·CQI + 0.15·PDI + 0.15·RPI + 0.10·SE + 0.10·LA")],
                   style={"fontFamily": "monospace", "fontSize": "14px",
                          "backgroundColor": "#f0f0f0", "padding": "12px",
                          "borderRadius": "6px"}),
        ])], className="mb-4")], width=10, className="mx-auto")])], fluid=True)

    # ── 2025 About — exact original text ──────────────────────────────
    p = {"fontSize": "16px", "lineHeight": "1.8", "marginBottom": "25px"}
    return dbc.Container([dbc.Row([dbc.Col([dbc.Card([dbc.CardBody([html.Div([

        html.P(
            "This website examines all the 350 hitters during the 2025 MLB regular season that had a "
            "good enough sample size with at least 200 plate appearances. However, this page doesn't "
            "just include the basic statistics of these players as most of the statistics are advanced "
            "statistics, like Hard Hit% and LD%. Additionally, expected performance is also tracked "
            "with statistics like wOBA and xSLG that are explained on the next tab in the metrics "
            "glossary.",
            style=p,
        ),

        html.P([
            "Using those advanced statistics, I came up with a formula to find the most undervalued "
            "hitters in the 2025 regular season. A player is undervalued when their underlying "
            "performance, meaning their expected metrics, quality of contact, and efficiency, is "
            "better than their surface stats and salary. ",
            html.Strong("As a result, this undervaluation score (UVS) formula rewards:"),
        ], style={**p, "marginBottom": "15px"}),

        html.Ul([
            html.Li("Great expected performance"),
            html.Li("Strong plate discipline and contact quality"),
            html.Li("High WAR per $1M"),
            html.Li("Positive luck differential"),
            html.Li("Strong run creation"),
            html.Li("Moderate to low salary or small market exposure"),
        ], style={"fontSize": "16px", "lineHeight": "1.8", "marginBottom": "25px",
                  "marginLeft": "20px"}),

        html.P(
            "There are 7 indexes created that go into the final UVS calculation which is seen on the "
            "third tab titled \"Undervalued Players.\" Each index consists of a collection of both "
            "advanced and basic metrics seen in the metrics glossary. However, not each of those "
            "indexes are weighted the same. Due to the main goal being finding undervalued performance, "
            "an expected performance index is weighted the most to find out which hitters have been "
            "unlucky. Then, a contact quality index is weighted the second highest followed by a plate "
            "discipline index and a run production index. The plate discipline index is weighted "
            "slightly more than the run production index because another aim of this formula is to "
            "predict future performance and plate discipline is a stronger predictor of future "
            "performance, while run production mostly measures past results. Then, a salary efficiency "
            "index is included to see their efficiency per $1M before a luck adjustment index "
            "concludes the final dynamic of the formula, to see how unlucky a player has been with "
            "their balls.",
            style=p,
        ),

        html.P(
            "Overall, this website aims to provide MLB fans an easily accessible source to see a "
            "bunch of basic and advanced metrics on hitters with a solid sample size during the 2025 "
            "regular season. All this is set up for the primary goal of this website, which is to "
            "rank how undervalued all 350 hitters with at least 200 plate appearances were. From "
            "this, users can learn new names of players who weren't talked about enough this past "
            "year and keep an eye out for them becoming possible stars/superstars in the coming "
            "years. From the ranking, users can also gain a deeper appreciation for the greatness of "
            "already well-recognized superstars!",
            style={**p, "marginBottom": "0"},
        ),

    ])]) ], className="mb-4")], width=10, className="mx-auto")])], fluid=True)


def _all_players_tab(df: pd.DataFrame, season: int):
    n = len(df)
    label = f"{n} hitter{'s' if n != 1 else ''} shown"
    table = build_datatable(df, season, table_id="stats-datatable")
    return dbc.Row([dbc.Col([
        _sample_size_notice(season, df),
        _data_availability_notice(season),
        dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col(html.H5("Player Statistics", className="mb-0"), md=6),
                    dbc.Col(html.Small(label, className="text-muted"), md=6,
                            className="text-end"),
                ])
            ]),
            dbc.CardBody([table]),
        ])
    ], width=12)])


def _uvs_formula_section():
    """Original UVS formula breakdown + z-score explanation (same for both seasons)."""
    mono = {"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "5px"}
    return html.Div([
        html.P([
            html.Strong("UVS = "),
            "0.25(EPI) + 0.20(CQI) + 0.15(PDI) + 0.15(RPI) + 0.10(SE) + 0.10(LA)",
        ], style={"fontFamily": "monospace", "fontSize": "14px", "marginBottom": "20px"}),
        html.Div([
            html.P([html.Strong("EPI (Expected Performance Index):"),
                    " mean(z(xwOBA), z(xSLG), z(xBA), z(xISO))"], style=mono),
            html.P([html.Strong("CQI (Contact Quality Index):"),
                    " mean(z(Barrel%), z(HardHit%), z(Exit Velo), z(Sweet Spot%))"], style=mono),
            html.P([html.Strong("PDI (Plate Discipline Index):"),
                    " z(BB%) - z(K%) - z(O-Swing%) + z(Z-Contact%) + z(Contact%)"], style=mono),
            html.P([html.Strong("RPI (Run Production Index):"),
                    " mean(z(wRC+), z(wOBA), z(OPS), z(ISO), z(R), z(RBI))"], style=mono),
            html.P([html.Strong("SE (Salary Efficiency):"),
                    " z(WAR per $1M)"], style=mono),
            html.P([html.Strong("LA (Luck Adjustment):"),
                    " mean(z(xwOBA - wOBA), z(xBA - BA), z(xSLG - SLG))"], style=mono),
        ], style={"backgroundColor": "#f8f9fa", "padding": "15px",
                  "borderRadius": "5px", "marginBottom": "20px"}),
        html.Div([
            html.H6("The Formula", style={"marginTop": "20px", "marginBottom": "15px"}),
            html.P(
                html.Strong("z = (x - x̄) / σ",
                            style={"fontFamily": "monospace", "fontSize": "16px"}),
                style={"textAlign": "center", "marginBottom": "15px"},
            ),
            html.Ul([
                html.Li("x → the individual player's value for a stat "
                        "(e.g., their Barrel% = 12.5)"),
                html.Li(["x̄ (x-bar) → the ", html.Em("mean"),
                         " (average) of that stat across all players "
                         "(e.g., league average Barrel% = 8.0)"]),
                html.Li(["σ (sigma) → the ",
                         html.Strong("standard deviation", style={"color": "#ff6b35"}),
                         " of that stat (how spread out the numbers are)"]),
            ], style={"marginBottom": "15px"}),
            html.P(
                "Then you subtract the average from each player's value, "
                "and divide by how spread out the data is.",
                style={"fontStyle": "italic"},
            ),
        ], style={"marginTop": "20px"}),
    ])


def _undervalued_tab(df: pd.DataFrame, season: int):
    rank_cols = ["undervalued_rank", "name", "uvs", "pa"]
    col_names = {"undervalued_rank": "Rank", "name": "Player",
                 "uvs": "UVS", "pa": "PA"}

    available = [c for c in rank_cols if c in df.columns]
    rank_df = df[available].copy()
    rank_df = rank_df.drop_duplicates(subset=["name"], keep="first")

    for c in available:
        if c not in ("name",):
            rank_df[c] = pd.to_numeric(rank_df[c], errors="coerce")

    low_sample_styles = []
    if coerce_season(season) == CURRENT_SEASON:
        threshold, _ = _low_sample_pa_threshold(df, season)
        low_sample_styles = [
            {"if": {"filter_query": f"{{pa}} < {threshold}"},
             "backgroundColor": "#fff8e6"},
            {"if": {"filter_query": f"{{pa}} < {threshold}", "column_id": "pa"},
             "color": "#b85c00", "fontWeight": "bold"},
        ]

    rank_table = dash_table.DataTable(
        data=rank_df.to_dict("records"),
        columns=[{"name": col_names.get(c, c), "id": c,
                  "format": {"specifier": ".0f" if c in ("undervalued_rank", "pa") else ".3f"},
                  "type": "numeric" if c != "name" else "text"}
                 for c in available],
        style_cell={"textAlign": "left", "padding": "10px", "fontSize": "13px"},
        style_header={"backgroundColor": "#2c3e50", "color": "white",
                      "fontWeight": "bold", "fontSize": "13px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
            {"if": {"column_id": "uvs"}, "color": "#1a7340", "fontWeight": "bold"},
        ] + low_sample_styles,
        sort_action="native",
        filter_action="native",
        page_action="native",
        page_size=50,
        export_format="csv",
        export_headers="display",
    )

    formula_box = dbc.Card([
        dbc.CardHeader(html.H5("UVS (Undervaluation Score)", className="mb-0")),
        dbc.CardBody([
            _uvs_formula_section(),
            rank_table,
        ]),
    ], className="mb-4")

    return dbc.Container([dbc.Row([dbc.Col([formula_box], width=12)])], fluid=True)


def _validation_table(data: pd.DataFrame, columns: list[dict], table_id: str) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=table_id,
        data=data.to_dict("records"),
        columns=columns,
        style_cell={"textAlign": "left", "padding": "10px", "fontSize": "13px"},
        style_header={"backgroundColor": "#2c3e50", "color": "white",
                      "fontWeight": "bold", "fontSize": "13px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        ],
        sort_action="native",
        page_action="native",
        page_size=20,
        export_format="csv",
        export_headers="display",
    )


def _chart_block(title: str, fig: go.Figure, description: str) -> html.Div:
    """HTML title + Plotly chart + plain-language description (titles never clip)."""
    chart_h = int(fig.layout.height) if fig.layout.height else 420
    fig.update_layout(
        title=None,
        autosize=False,
        width=None,
        height=chart_h,
        margin=dict(t=36, b=90, l=55, r=30),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    return html.Div([
        html.H5(title, className="mb-2",
                style={"fontSize": "16px", "fontWeight": "700", "color": "#1a1a1a",
                       "lineHeight": "1.35"}),
        dcc.Graph(
            figure=fig,
            config={"displayModeBar": False, "responsive": False},
            style={"height": f"{chart_h}px", "width": "100%"},
        ),
        html.P(description, style={
            "fontSize": "14px",
            "lineHeight": "1.65",
            "color": "#444",
            "marginTop": "8px",
            "marginBottom": "0",
        }),
    ], style={
        "backgroundColor": "#fff",
        "border": "1px solid #e8e8e8",
        "borderRadius": "8px",
        "padding": "16px 16px 18px",
        # Do NOT use height:100% — Plotly resize + percent height grows forever
    })


def _validation_tab():
    """2025 UVS scores vs 2026 follow-up — calibration and baseline comparison."""
    p = {"fontSize": "15px", "lineHeight": "1.75", "marginBottom": "18px"}
    cohort = load_validation_cohort(PROJECT_ROOT)
    if cohort.empty:
        return dbc.Alert(
            "Validation data unavailable. Need both comprehensive_stats_2025.csv and "
            "comprehensive_stats_2026.csv in data/processed/.",
            color="warning",
        )

    summary = validation_summary(cohort)
    calibration = build_tier_calibration(cohort, n_tiers=5)
    baseline = build_baseline_comparison(cohort, top_n=35)
    top_picks = build_top_picks_table(cohort, top_n=35)

    # ── Charts ────────────────────────────────────────────────────────────
    tier_x = calibration["uvs_tier"].astype(str).tolist()

    fig_delta_woba = go.Figure()
    fig_delta_woba.add_trace(go.Bar(
        x=tier_x,
        y=calibration["mean_delta_woba"],
        marker_color=["#c0392b" if v < 0 else "#27ae60"
                      for v in calibration["mean_delta_woba"]],
        text=[f"{v:+.3f}" for v in calibration["mean_delta_woba"]],
        textposition="outside",
        cliponaxis=False,
        showlegend=False,
    ))
    fig_delta_woba.add_hline(y=0, line_dash="dash", line_color="#666")
    fig_delta_woba.update_layout(
        xaxis_title="2025 UVS tier (equal-sized groups)",
        yaxis_title="Δ wOBA",
        height=380,
        plot_bgcolor="#fafafa",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )

    fig_wrc = go.Figure()
    if "mean_wrc_plus_2025" in calibration.columns:
        fig_wrc.add_trace(go.Bar(
            name="2025 wRC+",
            x=tier_x,
            y=calibration["mean_wrc_plus_2025"],
            marker_color="#95a5a6",
        ))
    fig_wrc.add_trace(go.Bar(
        name="2026 wRC+",
        x=tier_x,
        y=calibration["mean_wrc_plus_2026"],
        marker_color="#2980b9",
    ))
    fig_wrc.update_layout(
        barmode="group",
        xaxis_title="2025 UVS tier",
        yaxis_title="wRC+",
        height=380,
        plot_bgcolor="#fafafa",
    )

    scatter = cohort.copy()
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=scatter["uvs_2025"],
        y=scatter["delta_woba"],
        mode="markers",
        marker=dict(size=7, color=scatter["undervalued_rank"],
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="2025 rank")),
        text=scatter["name"],
        hovertemplate=(
            "%{text}<br>2025 UVS: %{x:.2f}<br>Δ wOBA: %{y:.3f}<extra></extra>"
        ),
        showlegend=False,
    ))
    if len(scatter) >= 5:
        z = np.polyfit(scatter["uvs_2025"], scatter["delta_woba"], 1)
        x_line = np.linspace(scatter["uvs_2025"].min(), scatter["uvs_2025"].max(), 50)
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=np.poly1d(z)(x_line),
            mode="lines", line=dict(color="#e74c3c", dash="dash"),
            name="Linear fit",
        ))
    fig_scatter.add_hline(y=0, line_dash="dot", line_color="#999")
    fig_scatter.update_layout(
        xaxis_title="2025 UVS (z-score composite)",
        yaxis_title="Δ wOBA (2026 − 2025)",
        height=400,
        plot_bgcolor="#fafafa",
    )

    # 2025 UVS vs 2026 UVS — score stability / talent persistence
    fig_uvs = go.Figure()
    uvs_ok = scatter.dropna(subset=["uvs_2025", "uvs_2026"])
    fig_uvs.add_trace(go.Scatter(
        x=uvs_ok["uvs_2025"],
        y=uvs_ok["uvs_2026"],
        mode="markers",
        marker=dict(size=7, color="#1a7340", opacity=0.65),
        text=uvs_ok["name"],
        hovertemplate=(
            "%{text}<br>2025 UVS: %{x:.2f}<br>2026 UVS: %{y:.2f}<extra></extra>"
        ),
        name="Hitters",
    ))
    if len(uvs_ok) >= 2:
        lo = float(min(uvs_ok["uvs_2025"].min(), uvs_ok["uvs_2026"].min()))
        hi = float(max(uvs_ok["uvs_2025"].max(), uvs_ok["uvs_2026"].max()))
        fig_uvs.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            line=dict(color="#999", dash="dot"),
            name="y = x (perfect persistence)",
        ))
        z_uvs = np.polyfit(uvs_ok["uvs_2025"], uvs_ok["uvs_2026"], 1)
        x_line = np.linspace(uvs_ok["uvs_2025"].min(), uvs_ok["uvs_2025"].max(), 50)
        fig_uvs.add_trace(go.Scatter(
            x=x_line, y=np.poly1d(z_uvs)(x_line),
            mode="lines",
            line=dict(color="#e74c3c", dash="dash"),
            name="Linear fit",
        ))
    corr_uvs = summary.get("corr_uvs_2025_2026")
    fig_uvs.update_layout(
        xaxis_title="2025 UVS",
        yaxis_title="2026 UVS",
        height=400,
        plot_bgcolor="#fafafa",
    )

    fig_uvs_tier = go.Figure()
    if "mean_uvs_2026" in calibration.columns:
        fig_uvs_tier.add_trace(go.Bar(
            name="Mean 2025 UVS",
            x=tier_x,
            y=calibration["mean_uvs_2025"],
            marker_color="#95a5a6",
        ))
        fig_uvs_tier.add_trace(go.Bar(
            name="Mean 2026 UVS",
            x=tier_x,
            y=calibration["mean_uvs_2026"],
            marker_color="#1a7340",
        ))
    fig_uvs_tier.update_layout(
        barmode="group",
        xaxis_title="2025 UVS tier",
        yaxis_title="Mean UVS",
        height=400,
        plot_bgcolor="#fafafa",
    )

    baseline_labels = baseline["group"].tolist()
    fig_baseline = go.Figure()
    fig_baseline.add_trace(go.Bar(
        x=baseline_labels,
        y=baseline["mean_delta_woba"],
        marker_color="#8e44ad",
        text=[f"{v:+.3f}" for v in baseline["mean_delta_woba"]],
        textposition="outside",
        cliponaxis=False,
    ))
    fig_baseline.add_hline(y=0, line_dash="dash", line_color="#666")
    fig_baseline.update_layout(
        xaxis_title="Comparison group",
        yaxis_title="Mean Δ wOBA (2026 − 2025)",
        height=420,
        plot_bgcolor="#fafafa",
        xaxis=dict(tickangle=-20),
    )

    # ── Summary cards ───────────────────────────────────────────────────
    top35_delta = summary.get("mean_delta_woba_top35")
    rest_delta = summary.get("mean_delta_woba_rest")
    all_delta = summary.get("mean_delta_woba_all")
    corr_uvs_yoy = summary.get("corr_uvs_2025_2026")
    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Matched hitters", className="text-muted"),
            html.H3(f"{summary['n_matched']}", className="mb-0"),
            html.Small("2025 ≥200 PA · 2026 ≥50 PA", className="text-muted"),
        ])), md=True),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Top 35 Δ wOBA", className="text-muted"),
            html.H3(f"{top35_delta:+.3f}" if top35_delta is not None else "—",
                    className="mb-0",
                    style={"color": "#c0392b" if top35_delta and top35_delta < 0 else "#27ae60"}),
            html.Small("2025 UVS rank 1–35", className="text-muted"),
        ])), md=True),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Everyone else Δ wOBA", className="text-muted"),
            html.H3(f"{rest_delta:+.3f}" if rest_delta is not None else "—", className="mb-0"),
            html.Small("Baseline: non-top-35 cohort", className="text-muted"),
        ])), md=True),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("UVS ↔ Δ wOBA", className="text-muted"),
            html.H3(f"{summary.get('corr_uvs_delta_woba', 0):+.2f}", className="mb-0"),
            html.Small(f"Cohort mean Δ = {all_delta:+.3f}", className="text-muted"),
        ])), md=True),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("2025 UVS ↔ 2026 UVS", className="text-muted"),
            html.H3(
                f"{corr_uvs_yoy:+.2f}" if corr_uvs_yoy is not None else "—",
                className="mb-0",
                style={"color": "#1a7340"},
            ),
            html.Small("Score persistence", className="text-muted"),
        ])), md=True),
    ], className="mb-4 g-3")

    baseline_cols = [
        {"name": "Group", "id": "group", "type": "text"},
        {"name": "N", "id": "n", "type": "numeric"},
        {"name": "Mean 2025 UVS", "id": "mean_uvs_2025", "type": "numeric",
         "format": {"specifier": ".3f"}},
        {"name": "2025 wOBA", "id": "mean_woba_2025", "type": "numeric",
         "format": {"specifier": ".3f"}},
        {"name": "2026 wOBA", "id": "mean_woba_2026", "type": "numeric",
         "format": {"specifier": ".3f"}},
        {"name": "Δ wOBA", "id": "mean_delta_woba", "type": "numeric",
         "format": {"specifier": "+.3f"}},
    ]
    if "mean_delta_wrc_plus" in baseline.columns:
        baseline_cols.extend([
            {"name": "Δ wRC+", "id": "mean_delta_wrc_plus", "type": "numeric",
             "format": {"specifier": "+.1f"}},
        ])

    picks_cols = [
        {"name": "2025 Rank", "id": "undervalued_rank", "type": "numeric"},
        {"name": "Player", "id": "name", "type": "text"},
        {"name": "2025 UVS", "id": "uvs_2025", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "2026 UVS", "id": "uvs_2026", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "2025 wOBA", "id": "woba_2025", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "2026 wOBA", "id": "woba_2026", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "Δ wOBA", "id": "delta_woba", "type": "numeric", "format": {"specifier": "+.3f"}},
    ]
    if "wrc_plus_2025" in top_picks.columns:
        picks_cols.extend([
            {"name": "2025 wRC+", "id": "wrc_plus_2025", "type": "numeric",
             "format": {"specifier": ".0f"}},
            {"name": "2026 wRC+", "id": "wrc_plus_2026", "type": "numeric",
             "format": {"specifier": ".0f"}},
            {"name": "Δ wRC+", "id": "delta_wrc_plus", "type": "numeric",
             "format": {"specifier": "+.0f"}},
        ])
    if "pa_2026" in top_picks.columns:
        picks_cols.append({"name": "2026 PA", "id": "pa_2026", "type": "numeric"})

    cal_display = calibration.copy()
    cal_cols = [
        {"name": "2025 UVS tier", "id": "uvs_tier", "type": "text"},
        {"name": "N", "id": "n", "type": "numeric"},
        {"name": "Mean 2025 UVS", "id": "mean_uvs_2025", "type": "numeric",
         "format": {"specifier": ".3f"}},
    ]
    if "mean_uvs_2026" in cal_display.columns:
        cal_cols.append({"name": "Mean 2026 UVS", "id": "mean_uvs_2026",
                         "type": "numeric", "format": {"specifier": ".3f"}})
    cal_cols.append({"name": "Mean Δ wOBA", "id": "mean_delta_woba", "type": "numeric",
                     "format": {"specifier": "+.3f"}})
    if "mean_delta_wrc_plus" in cal_display.columns:
        cal_cols.append({"name": "Mean Δ wRC+", "id": "mean_delta_wrc_plus",
                         "type": "numeric", "format": {"specifier": "+.1f"}})
    if "mean_wrc_plus_2026" in cal_display.columns:
        cal_cols.append({"name": "Mean 2026 wRC+", "id": "mean_wrc_plus_2026",
                         "type": "numeric", "format": {"specifier": ".1f"}})

    interpretation = html.Div([
        html.P([
            "In this matched sample, higher 2025 UVS tiers ",
            html.Strong("did not outperform"),
            " on year-over-year wOBA improvement — top-ranked hitters saw a larger ",
            f"mean wOBA drop ({top35_delta:+.3f}) than the rest of the cohort ",
            f"({rest_delta:+.3f}). That pattern is consistent with ",
            html.Em("regression to the mean"),
            " and with the Luck Adjustment (LA) component flagging players who were ",
            "already outperforming their expected stats in 2025."
        ], style=p) if top35_delta is not None and rest_delta is not None else html.Div(),
        html.P([
            "Higher UVS tiers still posted stronger ",
            html.Strong("absolute"),
            " 2026 wRC+ levels (see quintile chart), and 2025 UVS correlates with ",
            html.Strong("2026 UVS"),
            (f" (r = {corr_uvs_yoy:+.2f})" if corr_uvs_yoy is not None else ""),
            " — evidence the score persists as a ",
            html.Strong("talent / value index"),
            ", even when year-over-year wOBA gains do not. It is not a guaranteed "
            "breakout predictor over a single offseason."
        ], style={**p, "marginBottom": "0"}),
    ], style={"backgroundColor": "#f8f9fa", "padding": "16px 20px",
              "borderRadius": "8px", "borderLeft": "4px solid #3498db"})

    return dbc.Container([
        dbc.Card([dbc.CardBody([
            html.H4("Did 2025 undervalued hitters outperform afterward?", className="mb-3"),
            html.P([
                "This tab tracks hitters who qualified in the ",
                html.Strong("2025 full-season benchmark"),
                " (≥200 PA) and reappear in ",
                html.Strong("2026 live data"),
                " (≥50 PA). We compare each player's ",
                html.Strong("frozen 2025 UVS score"),
                " to ",
                html.Strong("actual 2026 performance"),
                " (wOBA, wRC+) and show ",
                html.Strong("calibration by UVS tier"),
                " plus ",
                html.Strong("simple baselines"),
                " (top-35 picks vs everyone else vs bottom quintile)."
            ], style=p),
            html.P([
                html.Strong("Limitation: "),
                "We do not have a dedicated 2025 second-half split in this pipeline, so ",
                "follow-up uses ",
                html.Strong("2026 season-to-date"),
                " rather than 2025 H2 only. That mixes true talent change with a new season ",
                "and different run environments — interpret year-over-year deltas cautiously."
            ], style={**p, "marginBottom": "0", "fontSize": "14px", "color": "#555"}),
        ])], className="mb-4"),

        cards,
        interpretation,

        dbc.Row([
            dbc.Col(_chart_block(
                f"2025 UVS vs 2026 UVS"
                + (f" (r = {corr_uvs:+.2f})" if corr_uvs is not None else ""),
                fig_uvs,
                "Each point is one hitter. The dotted gray line is perfect year-to-year "
                "persistence (same UVS both seasons). The red dashed line is the actual "
                "relationship. A positive correlation means high-UVS hitters in 2025 "
                "tended to stay relatively high in 2026 — UVS is sticky as a talent/"
                "value signal, even if individual scores regress toward the middle.",
            ), md=6),
            dbc.Col(_chart_block(
                "Mean UVS by 2025 quintile (2025 score vs 2026 score)",
                fig_uvs_tier,
                "Hitters are split into five equal groups by 2025 UVS. Gray bars are "
                "each group's average 2025 score; green bars are that same group's "
                "average 2026 UVS. Q5 stays highest and Q1 stays lowest in 2026, but "
                "the gap shrinks — classic regression to the mean, not a full reshuffle "
                "of the rankings.",
            ), md=6),
        ], className="mb-3 g-3"),

        dbc.Row([
            dbc.Col(_chart_block(
                "Mean wOBA change (2026 − 2025) by 2025 UVS quintile",
                fig_delta_woba,
                "This asks whether higher 2025 UVS predicted better offense next year. "
                "Green bars mean that quintile's average wOBA rose in 2026; red bars "
                "mean it fell. If UVS were a clean breakout predictor, Q5 would be "
                "green and largest. Here the opposite pattern appears: the highest "
                "2025 UVS group saw the biggest wOBA drop on average.",
            ), md=6),
            dbc.Col(_chart_block(
                "wRC+ by 2025 UVS quintile (2025 vs 2026)",
                fig_wrc,
                "Same quintiles, but looking at production levels instead of change. "
                "Gray is 2025 wRC+; blue is 2026 wRC+. Higher 2025 UVS groups still "
                "tend to have higher absolute 2026 wRC+ — so UVS tracks who the better "
                "hitters are, even when year-over-year gains do not favor the top tier.",
            ), md=6),
        ], className="mb-3 g-3"),

        dbc.Row([
            dbc.Col(_chart_block(
                "2025 UVS vs wOBA change into 2026",
                fig_scatter,
                "Player-level view of the same idea as the Δ wOBA quintile chart. "
                "X is 2025 UVS; Y is how much wOBA changed into 2026. Points above "
                "zero improved; below zero got worse. Color is 2025 UVS rank. A "
                "downward-sloping red fit means higher 2025 UVS was associated with "
                "worse, not better, wOBA change in this sample.",
            ), md=6),
            dbc.Col(_chart_block(
                "Δ wOBA vs simple baselines",
                fig_baseline,
                "A head-to-head check against naive groups. Compare the top 35 by "
                "2025 UVS rank to: everyone else in the matched sample, the bottom "
                "UVS quintile, and the full cohort average. If the model found true "
                "buy-lows, the top-35 bar should beat the baselines. Here it does "
                "not — the top picks had the weakest mean Δ wOBA.",
            ), md=6),
        ], className="mb-4 g-3"),

        dbc.Card([dbc.CardHeader(html.H5("Calibration by 2025 UVS quintile", className="mb-0")),
                  dbc.CardBody([
                      html.P(
                          "Equal-sized groups sorted by 2025 UVS. Shows whether higher scores "
                          "translated into better follow-up gains.",
                          className="text-muted small mb-3",
                      ),
                      _validation_table(cal_display, cal_cols, "validation-calibration-table"),
                  ])], className="mb-4"),

        dbc.Card([dbc.CardHeader(html.H5("Baseline comparison", className="mb-0")),
                  dbc.CardBody([
                      html.P([
                          "Simple checks against non-model groups: all other matched hitters, ",
                          "bottom UVS quintile (Q1), and the full cohort average."
                      ], className="text-muted small mb-3"),
                      _validation_table(baseline, baseline_cols, "validation-baseline-table"),
                  ])], className="mb-4"),

        dbc.Card([dbc.CardHeader(html.H5("Top 35 by 2025 UVS — player-level follow-up",
                                         className="mb-0")),
                  dbc.CardBody([
                      html.P(
                          "The same cutoff as the headline undervalued leaderboard (~top decile). "
                          "Sort or export to inspect individual hits and misses.",
                          className="text-muted small mb-3",
                      ),
                      _validation_table(top_picks, picks_cols, "validation-top-picks-table"),
                  ])], className="mb-4"),
    ], fluid=True)


# ── Run ─────────────────────────────────────────────────────────────────────

def _warm_cache() -> None:
    """Pre-load both seasons once at startup for fast tab/search switching."""
    if live_refresh_enabled():
        pull_latest_committed_data()
    for s in (BENCHMARK_SEASON, CURRENT_SEASON):
        try:
            load_season_data(s)
        except Exception:
            pass


_warm_cache()
start_background_scheduler()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print("\n" + "=" * 60)
    print("Undervalued MLB Hitters Dashboard")
    print(f"http://localhost:{port}")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
