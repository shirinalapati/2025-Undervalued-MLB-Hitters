"""
Undervalued MLB Hitters Dashboard — dual-mode (2025 full season / 2026 live season).

Tabs:
  1. About This Page   — methodology overview
  2. All Players Stats — full sortable/filterable table with CSV export
  3. Undervalued Players — UVS leaderboard

Season modes:
  2025 Full Season  → static benchmark; ≥350 PA; ranks by raw UVS
  2026 Live Season  → live data; no hard PA floor; reliability-weighted score
                      adjusted_uvs = w*uvs_norm + (1-w)*50   where w = PA/(PA+120)
"""

import os
import sys
import traceback
import requests
from pathlib import Path
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from numpy import integer as np_integer, floating as np_floating

# ── Paths & config ─────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))

PROJECT_ROOT  = Path(__file__).parent.parent
DATA_DIR      = PROJECT_ROOT / "data" / "processed"
API_BASE_URL  = "http://localhost:8000"

BENCHMARK_SEASON  = 2025
CURRENT_SEASON    = 2026
RELIABILITY_K     = 120
LEAGUE_AVG_SCORE  = 50.0
MIN_PA_2025       = 200   # original 2025 filter shown in UI; pipeline already pre-filters
MIN_PA_2026       = 10    # show everyone in 2026; reliability handles noise

SEASON_OPTIONS = [
    {"label": "2025 Full Season", "value": 2025},
    {"label": "2026 Live Season", "value": 2026},
]

# ── App init ───────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server          # WSGI entry-point for PythonAnywhere / Render
app.title = "Undervalued MLB Hitters"


# ── Reliability helpers (inline so dashboard is self-contained) ─────────────

def _compute_reliability(pa: pd.Series) -> pd.Series:
    pa = pa.fillna(0).clip(lower=0)
    return pa / (pa + RELIABILITY_K)


def _normalize_uvs(uvs: pd.Series) -> pd.Series:
    uvs = uvs.fillna(uvs.mean() if uvs.notna().any() else 0)
    lo, hi = uvs.min(), uvs.max()
    return (uvs - lo) / (hi - lo) * 100.0 if hi > lo else pd.Series(LEAGUE_AVG_SCORE, index=uvs.index)


def _apply_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Add reliability, uvs_normalized, and adjusted_uvs columns."""
    df = df.copy()
    pa_col = next((c for c in ["pa", "PA"] if c in df.columns), None)
    pa = df[pa_col].fillna(0) if pa_col else pd.Series(0, index=df.index)
    df["reliability"]     = _compute_reliability(pa)
    df["reliability_pct"] = (df["reliability"] * 100).round(1)
    df["uvs_normalized"]  = _normalize_uvs(df["uvs"]) if "uvs" in df.columns and df["uvs"].notna().any() else LEAGUE_AVG_SCORE
    df["adjusted_uvs"]    = (df["reliability"] * df["uvs_normalized"] + (1 - df["reliability"]) * LEAGUE_AVG_SCORE).round(2)
    return df


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
    ts_file = DATA_DIR / f"last_updated_{season}.txt"
    if ts_file.exists():
        return ts_file.read_text().strip()
    # Try alternate cwd
    alt = Path(os.getcwd()) / "data" / "processed" / f"last_updated_{season}.txt"
    if alt.exists():
        return alt.read_text().strip()
    return "Unknown"


def load_season_data(season: int) -> pd.DataFrame:
    """
    Load processed CSV for the given season, apply position filter, PA filter,
    compute advanced metrics, and (for 2026) add reliability columns.
    """
    path = _csv_path(season)
    if path is None:
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Keep hitters only
    if "position_type" in df.columns:
        df = df[df["position_type"] == "Hitter"].copy()

    # PA filter
    pa_col = next((c for c in ["pa", "PA"] if c in df.columns), None)
    min_pa = MIN_PA_2025 if season == BENCHMARK_SEASON else MIN_PA_2026
    if pa_col:
        df = df[df[pa_col] >= min_pa].copy()

    # Compute / refresh metrics
    try:
        from src.utils.metrics import calculate_all_advanced_metrics
        df = calculate_all_advanced_metrics(df)
    except Exception:
        pass
    try:
        from src.utils.uvs_metrics import calculate_all_uvs_metrics
        df = calculate_all_uvs_metrics(df)
    except Exception:
        pass
    try:
        from src.utils.tova_metrics import calculate_all_composite_metrics
        df = calculate_all_composite_metrics(df)
    except Exception:
        pass

    # Derived columns
    df = _fill_missing_columns(df, season)

    if season == CURRENT_SEASON:
        df = _apply_reliability(df)
        sort_col = "adjusted_uvs"
    else:
        sort_col = "uvs" if "uvs" in df.columns and df["uvs"].notna().any() else "undervalued_rank"

    # Rank
    if sort_col in df.columns and df[sort_col].notna().any():
        asc = sort_col == "undervalued_rank"
        df = df.sort_values(sort_col, ascending=asc, na_position="last")
    df["undervalued_rank"] = range(1, len(df) + 1)

    return df


def _fill_missing_columns(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Fill / map derived columns needed by the table."""
    df = df.copy()

    # Player name
    for col in ["last_name, first_name", "Name", "name"]:
        if col in df.columns and df[col].notna().any():
            df["name"] = df[col]
            break
    if "name" not in df.columns:
        df["name"] = "Unknown"
    df["name"] = df["name"].fillna("Unknown")

    # xwOBA
    for col in ["est_woba", "xwoba", "xwOBA"]:
        if col in df.columns and df[col].notna().any():
            df["xwoba"] = df[col]
            break
    # xBA / xSLG
    for src, dst in [("est_ba", "xba"), ("est_slg", "xslg")]:
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    # xISO
    if "xiso" not in df.columns and "xslg" in df.columns and "xba" in df.columns:
        df["xiso"] = df["xslg"] - df["xba"]

    # BA
    if "ba" not in df.columns:
        if "BA" in df.columns:
            df["ba"] = df["BA"]
        elif "H" in df.columns and "AB" in df.columns:
            df["ba"] = (df["H"] / df["AB"].replace(0, np.nan)).round(3)

    # Salary column (season-specific)
    salary_col = f"salary_{season}"
    if salary_col not in df.columns:
        for alt in [f"salary_{season}_x", "salary_2025", "salary_2025_x", "salary"]:
            if alt in df.columns and df[alt].notna().any():
                df[salary_col] = df[alt]
                break
    # Unify to 'salary_display' for table
    df["salary_display"] = df.get(salary_col, pd.Series(np.nan, index=df.index))

    # xHR
    for col in ["est_hr", "xHR", "xhr"]:
        if col in df.columns:
            df["xhr"] = df[col]
            break

    # WAR/$1M
    if "war_per_salary" not in df.columns:
        war = df.get("WAR", df.get("war"))
        sal = df.get(salary_col)
        if war is not None and sal is not None:
            valid = sal.notna() & (sal > 0)
            df["war_per_salary"] = np.where(valid, war / (sal + 0.1), np.nan)

    # Lowercase counting stats
    for upper, lower in [("AB","ab"),("H","h"),("R","r"),("RBI","rbi"),
                          ("HR","hr"),("BB","bb"),("K","k"),("OBP","obp"),
                          ("SLG","slg"),("ISO","iso"),("OPS","ops"),
                          ("BABIP","babip"),("BA","ba"),("WAR","war")]:
        if lower not in df.columns and upper in df.columns:
            df[lower] = df[upper]
    # K fallback to SO
    if "k" not in df.columns or df["k"].isna().all():
        for alt in ["K","SO","so","Strikes"]:
            if alt in df.columns and df[alt].notna().any():
                df["k"] = df[alt]
                break

    # wRC+
    for col in ["wRC+", "wrc_plus", "wRCplus"]:
        if col in df.columns:
            df["wrc_plus"] = df[col]
            break

    return df


def apply_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Filter dataframe rows by player name substring (case-insensitive)."""
    if not query or not query.strip():
        return df
    q = query.strip().lower()
    mask = df["name"].str.lower().str.contains(q, na=False)
    return df[mask]


# ── Column definitions ──────────────────────────────────────────────────────

def _hitter_columns_2025():
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


def _hitter_columns_2026():
    """2026 columns — identical to 2025 but with reliability columns inserted after Rank."""
    base = _hitter_columns_2025()
    # Insert reliability columns right after Rank and before UVS
    reliability_cols = [
        {"name": "Adj. UVS",    "id": "adjusted_uvs",    "type": "numeric"},
        {"name": "Raw UVS",     "id": "uvs_normalized",  "type": "numeric"},
        {"name": "Reliability", "id": "reliability_pct", "type": "numeric"},
    ]
    # Remove UVS from base (we replace it with adj/raw pair)
    base = [c for c in base if c["id"] != "uvs"]
    # Insert after Rank + Player
    result = base[:2] + reliability_cols + base[2:]
    return result


# ── Table column formatting ──────────────────────────────────────────────────

_PCT_IDS = {
    "barrel_batted_rate","hard_hit_percent","sweet_spot_percent",
    "bb_percent","k_percent","o_swing_percent","z_contact_percent",
    "contact_percent","gb_percent","fb_percent","ld_percent",
    "pull_percent","oppo_percent","reliability_pct",
}
_THREE_DP  = {"uvs","woba","xwoba","xba","xslg","xiso","adjusted_uvs","uvs_normalized","ba","obp","slg","iso","ops","babip"}
_TWO_DP    = {"war","war_per_salary","salary_display","xhr"}
_ONE_DP    = {"avg_exit_velocity","exit_velocity"}
_ZERO_DP   = {"undervalued_rank","pa","ab","h","r","rbi","hr","bb","k","wrc_plus"}


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
    "z_contact_percent":  ["Z-Contact%", "z_contact_percent"],
    "contact_percent":    ["Contact%", "contact_percent"],
    "gb_percent":         ["GB%", "gb_percent"],
    "fb_percent":         ["FB%", "fb_percent"],
    "ld_percent":         ["LD%", "ld_percent"],
    "pull_percent":       ["Pull%", "pull_percent"],
    "oppo_percent":       ["Oppo%", "oppo_percent"],
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


def _resolve(col_id: str, df: pd.DataFrame) -> str | None:
    """Return the actual DataFrame column name for a table column ID."""
    if col_id in df.columns:
        return col_id
    for alt in _ALIASES.get(col_id, []):
        if alt in df.columns:
            return alt
    return None


def _get_val(row, col_id: str, df: pd.DataFrame) -> object:
    src = _resolve(col_id, df)
    if src is None:
        return ""
    val = row.get(src, "")
    if pd.isna(val) or val is None:
        return ""
    # Convert decimal fractions to percentages where needed
    if col_id in _PCT_DECIMAL_IDS and isinstance(val, (int, float)) and 0 <= val < 1.5:
        val = val * 100
    # Numpy → Python
    if hasattr(val, "item"):
        val = val.item()
    if col_id == "undervalued_rank":
        return int(val)
    return val


def _build_table_records(df: pd.DataFrame, col_defs: list) -> list:
    df = df.reset_index(drop=True)
    rows = df.to_dict("records")
    result = []
    for row in rows:
        rec = {}
        for col in col_defs:
            cid = col["id"]
            # Direct
            if cid in row and row[cid] not in (None, "") and not (isinstance(row[cid], float) and np.isnan(row[cid])):
                val = row[cid]
                if cid in _PCT_DECIMAL_IDS and isinstance(val, (int, float)) and 0 <= val < 1.5:
                    val = val * 100
                if hasattr(val, "item"):
                    val = val.item()
                if cid == "undervalued_rank":
                    val = int(val)
                rec[cid] = val
            else:
                # Try alias
                found = False
                for alt in _ALIASES.get(cid, []):
                    if alt in row and row[alt] not in (None, "") and not (isinstance(row[alt], float) and np.isnan(row[alt])):
                        val = row[alt]
                        if cid in _PCT_DECIMAL_IDS and isinstance(val, (int, float)) and 0 <= val < 1.5:
                            val = val * 100
                        if hasattr(val, "item"):
                            val = val.item()
                        rec[cid] = val
                        found = True
                        break
                if not found:
                    rec[cid] = ""
        result.append(rec)
    return result


# ── DataTable builder ───────────────────────────────────────────────────────

def build_datatable(df: pd.DataFrame, season: int, table_id: str = "stats-datatable") -> dash_table.DataTable:
    col_defs = _hitter_columns_2026() if season == CURRENT_SEASON else _hitter_columns_2025()
    formatted_cols = [_format_col(c["id"], c["name"]) for c in col_defs]
    records = _build_table_records(df, col_defs)

    style_data_conditional = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        {"if": {"column_id": "name"}, "fontWeight": "bold", "minWidth": "160px", "maxWidth": "210px"},
        {"if": {"column_id": "undervalued_rank"}, "textAlign": "center", "fontWeight": "bold"},
        {"if": {"column_id": "adjusted_uvs"}, "color": "#1a7340", "fontWeight": "bold"},
        {"if": {"column_id": "uvs"}, "color": "#1a7340", "fontWeight": "bold"},
    ]

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


# ── Layout ──────────────────────────────────────────────────────────────────

app.layout = dbc.Container([

    # ── Hidden stores ────────────────────────────────────────────────────
    dcc.Store(id="season-store", data=BENCHMARK_SEASON),

    # ── Header ───────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.H1("Undervalued MLB Hitters Analysis",
                    className="text-center mb-1",
                    style={"color": "#1a1a1a", "fontWeight": "700"}),
            html.Div(id="app-subtitle", className="text-center text-muted mb-1",
                     style={"fontSize": "15px"}),
            html.Div(id="last-updated-display", className="text-center mb-3",
                     style={"fontSize": "12px", "color": "#888"}),
            html.Hr(),
        ])
    ]),

    # ── Controls row ─────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        # Season selector
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

                        # Player search
                        dbc.Col([
                            html.Label("Search Player:", className="fw-bold mb-1",
                                       style={"fontSize": "13px"}),
                            dbc.Input(
                                id="player-search",
                                type="text",
                                placeholder="e.g. Shohei Ohtani…",
                                debounce=True,
                                style={"fontSize": "13px"},
                            ),
                        ], md=3),

                        # Player count slider
                        dbc.Col([
                            html.Label("Players shown:", className="fw-bold mb-1",
                                       style={"fontSize": "13px"}),
                            dcc.Slider(
                                id="top-n-slider",
                                min=10, max=500, step=10, value=500,
                                marks={10: "10", 50: "50", 100: "100",
                                       200: "200", 350: "350", 500: "All"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ], md=7),
                    ], align="center"),
                ])
            ], className="mb-3")
        ])
    ]),

    # ── Metrics glossary ─────────────────────────────────────────────────
    dbc.Row([dbc.Col([
        dbc.Collapse(
            id="metrics-glossary-collapse", is_open=False,
            children=dbc.Card([dbc.CardBody([
                html.H5("📚 Metrics Glossary", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.H6("Contact Quality & Power", className="mt-3 mb-2"),
                        html.P([html.Strong("Barrel%"), " – Ideal EV + launch angle (Statcast barrels)."]),
                        html.P([html.Strong("HardHit%"), " – % of batted balls at ≥ 95 mph EV."]),
                        html.P([html.Strong("Exit Velo"), " – Average exit velocity (mph)."]),
                        html.P([html.Strong("Sweet Spot%"), " – Launch angle 8–32°."]),
                        html.H6("Expected Performance", className="mt-4 mb-2"),
                        html.P([html.Strong("xwOBA"), " – Expected wOBA from contact quality + Ks/BBs."]),
                        html.P([html.Strong("xBA"), " – Expected BA from EV + launch angle."]),
                        html.P([html.Strong("xSLG"), " – Expected SLG from contact quality."]),
                        html.P([html.Strong("xISO"), " – xSLG − xBA; expected isolated power."]),
                        html.H6("Plate Discipline", className="mt-4 mb-2"),
                        html.P([html.Strong("BB%"), " – Walks / PA."]),
                        html.P([html.Strong("K%"), " – Strikeouts / PA."]),
                        html.P([html.Strong("Chase%"), " – Swing % on pitches outside zone."]),
                        html.P([html.Strong("Z-Contact%"), " – Contact % on in-zone swings."]),
                    ], md=6),
                    dbc.Col([
                        html.H6("Batted-Ball Profile", className="mt-3 mb-2"),
                        html.P([html.Strong("GB/FB/LD%"), " – Ground ball / fly ball / line drive rates."]),
                        html.P([html.Strong("Pull% / Oppo%"), " – Pull side / opposite field rates."]),
                        html.H6("Value Metrics", className="mt-4 mb-2"),
                        html.P([html.Strong("wRC+"), " – Park/league-adjusted runs created (100 = avg)."]),
                        html.P([html.Strong("WAR"), " – Wins Above Replacement."]),
                        html.P([html.Strong("WAR/$1M"), " – WAR per million dollars of salary."]),
                        html.H6("2026 Live-Season Metrics", className="mt-4 mb-2"),
                        html.P([html.Strong("Adj. UVS"), " – Reliability-adjusted score. "
                                "= w × UVS_norm + (1−w) × 50,  w = PA / (PA + 120)."]),
                        html.P([html.Strong("Raw UVS"), " – UVS normalized to 0–100 (no shrinkage)."]),
                        html.P([html.Strong("Reliability"), " – Sample confidence %. "
                                "≈50% at 120 PA, ≈74% at 350 PA."]),
                        html.H6("UVS Formula", className="mt-4 mb-2"),
                        html.P("UVS = 0.25·EPI + 0.20·CQI + 0.15·PDI + 0.15·RPI + 0.10·SE + 0.10·LA",
                               style={"fontFamily": "monospace", "fontSize": "12px"}),
                    ], md=6),
                ])
            ])], className="mb-3"),
        ),
        dbc.Button("📚 Show / Hide Metrics Glossary",
                   id="metrics-glossary-toggle",
                   color="secondary", outline=True, className="mb-3",
                   style={"fontSize": "13px"}),
        html.Hr()
    ])]),

    # ── Loading spinner ───────────────────────────────────────────────────
    dcc.Loading(id="loading", type="default",
                children=html.Div(id="loading-output")),

    # ── Tabs ─────────────────────────────────────────────────────────────
    dbc.Row([dbc.Col([
        dbc.Tabs([
            dbc.Tab(label="ℹ️ About This Page",  tab_id="about"),
            dbc.Tab(label="📊 All Players Stats",  tab_id="all-players"),
            dbc.Tab(label="💎 Undervalued Players", tab_id="undervalued"),
        ], id="main-tabs", active_tab="about", className="mb-3")
    ])]),

    # ── Main content ──────────────────────────────────────────────────────
    html.Div(id="main-content"),

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
    Output("season-store", "data"),
    Input("season-dropdown", "value"),
)
def sync_season(val):
    return val or BENCHMARK_SEASON


@app.callback(
    [Output("main-content", "children"),
     Output("app-subtitle", "children"),
     Output("last-updated-display", "children"),
     Output("loading-output", "children")],
    [Input("top-n-slider", "value"),
     Input("main-tabs", "active_tab"),
     Input("season-store", "data"),
     Input("player-search", "value")],
)
def update_main_content(top_n, active_tab, season, search_query):
    season = season or BENCHMARK_SEASON
    top_n  = top_n  or 500

    # Subtitle
    if season == CURRENT_SEASON:
        subtitle = f"🔴 2026 Live Season  |  Sample-size-adjusted rankings  |  All qualified hitters"
    else:
        subtitle = f"2025 Full Season  |  All hitters with ≥ 200 PA  |  Full-season benchmark"

    # Last updated
    lu = _last_updated(season)
    last_upd = html.Span([
        html.Span("Last updated: ", style={"fontWeight": "600"}),
        html.Span(lu),
    ])

    if active_tab == "about":
        return _about_content(season), subtitle, last_upd, ""

    # Load data
    try:
        df = load_season_data(season)
    except Exception as exc:
        err = dbc.Alert(f"Error loading data: {exc}", color="danger")
        return err, subtitle, last_upd, ""

    if df.empty:
        msg = (
            f"No 2026 data yet. Run:  python scripts/fetch_2026_data.py"
            if season == CURRENT_SEASON
            else "No data found. Please run the data pipeline."
        )
        return dbc.Alert(msg, color="warning"), subtitle, last_upd, ""

    # Apply search
    df = apply_search(df, search_query)

    # Respect slider
    df = df.head(min(top_n, len(df)))

    if active_tab == "all-players":
        content = _all_players_tab(df, season)
    elif active_tab == "undervalued":
        content = _undervalued_tab(df, season)
    else:
        content = _about_content(season)

    return content, subtitle, last_upd, ""


# ── Tab renderers ────────────────────────────────────────────────────────────

def _about_content(season: int):
    if season == CURRENT_SEASON:
        intro = html.Div([
            html.P([
                "Welcome to the ", html.Strong("2026 Live Season"), " view. This dashboard tracks every MLB "
                "hitter in real time as the 2026 season unfolds. Unlike the 2025 full-season benchmark, "
                "live rankings use ", html.Strong("sample-size-aware scoring"), " so small early-season "
                "samples don't distort the leaderboard."
            ], style={"fontSize": "16px", "lineHeight": "1.8", "marginBottom": "20px"}),
            dbc.Alert([
                html.Strong("Live-season scores are sample-size adjusted and regressed toward league "
                            "average to reduce early-season noise."),
                html.Br(),
                "Formula:  Adj. UVS = w × UVS_norm + (1 − w) × 50,  where w = PA / (PA + 120).",
            ], color="info", className="mb-4"),
            html.P([
                "As the season progresses and PA accumulate, the reliability weight (w) approaches 1 and the "
                "adjusted score converges to the raw UVS. At 120 PA, the score is 50% reliable; at 350 PA "
                "(the qualifying threshold used in 2025), it reaches ~74% reliability."
            ], style={"fontSize": "15px", "lineHeight": "1.8", "marginBottom": "20px"}),
        ])
    else:
        intro = html.Div([
            html.P([
                "This website examines all 350 hitters from the 2025 MLB regular season with at least 200 "
                "plate appearances. Statistics include both traditional and advanced metrics — Hard Hit%, "
                "xwOBA, LD%, and more — explained in the Metrics Glossary above."
            ], style={"fontSize": "16px", "lineHeight": "1.8", "marginBottom": "20px"}),
        ])

    shared = html.Div([
        html.P([
            "Using advanced statistics, the ", html.Strong("Undervaluation Score (UVS)"),
            " identifies hitters whose underlying performance — expected metrics, contact quality, "
            "and efficiency — exceeds their surface stats and salary."
        ], style={"fontSize": "16px", "lineHeight": "1.8", "marginBottom": "15px"}),
        html.P([html.Strong("UVS rewards:")],
               style={"fontSize": "16px", "marginBottom": "8px"}),
        html.Ul([
            html.Li("Great expected performance (xwOBA, xSLG, xBA)"),
            html.Li("Strong plate discipline and contact quality"),
            html.Li("High WAR per $1M of salary"),
            html.Li("Positive luck differential (expected > actual outcomes)"),
            html.Li("Strong run creation (wRC+)"),
        ], style={"fontSize": "15px", "lineHeight": "1.9", "marginLeft": "20px",
                   "marginBottom": "25px"}),
        html.P(
            "UVS = 0.25·EPI + 0.20·CQI + 0.15·PDI + 0.15·RPI + 0.10·SE + 0.10·LA",
            style={"fontFamily": "monospace", "fontSize": "14px",
                   "backgroundColor": "#f0f0f0", "padding": "12px",
                   "borderRadius": "6px", "marginBottom": "20px"},
        ),
    ])

    return dbc.Container([
        dbc.Row([dbc.Col([
            dbc.Card([dbc.CardBody([intro, shared])], className="mb-4")
        ], width=10, className="mx-auto")])
    ], fluid=True)


def _all_players_tab(df: pd.DataFrame, season: int):
    n = len(df)
    label = f"{n} hitter{'s' if n != 1 else ''} shown"
    table = build_datatable(df, season, table_id="stats-datatable")
    return dbc.Row([dbc.Col([
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


def _undervalued_tab(df: pd.DataFrame, season: int):
    is_live = (season == CURRENT_SEASON)

    # Build the ranking table
    if is_live:
        rank_cols = ["undervalued_rank", "name", "adjusted_uvs", "uvs_normalized",
                     "reliability_pct", "pa"]
        col_names = {"undervalued_rank": "Rank", "name": "Player",
                     "adjusted_uvs": "Adj. UVS", "uvs_normalized": "Raw UVS",
                     "reliability_pct": "Reliability (%)", "pa": "PA"}
    else:
        rank_cols = ["undervalued_rank", "name", "uvs", "pa"]
        col_names = {"undervalued_rank": "Rank", "name": "Player",
                     "uvs": "UVS", "pa": "PA"}

    available = [c for c in rank_cols if c in df.columns]
    rank_df = df[available].copy()
    rank_df = rank_df.drop_duplicates(subset=["name"], keep="first")

    for c in available:
        if c not in ("name",):
            rank_df[c] = pd.to_numeric(rank_df[c], errors="coerce")

    rank_table = dash_table.DataTable(
        data=rank_df.to_dict("records"),
        columns=[{"name": col_names.get(c, c), "id": c,
                  "format": {"specifier": ".0f" if c in ("undervalued_rank","pa") else ".2f"},
                  "type": "numeric" if c != "name" else "text"}
                 for c in available],
        style_cell={"textAlign": "left", "padding": "10px", "fontSize": "13px"},
        style_header={"backgroundColor": "#2c3e50", "color": "white",
                      "fontWeight": "bold", "fontSize": "13px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
            {"if": {"column_id": "adjusted_uvs"}, "color": "#1a7340", "fontWeight": "bold"},
            {"if": {"column_id": "uvs"}, "color": "#1a7340", "fontWeight": "bold"},
        ],
        sort_action="native",
        filter_action="native",
        page_action="native",
        page_size=50,
        export_format="csv",
        export_headers="display",
    )

    live_note = dbc.Alert(
        "📡 Live-season scores are sample-size adjusted and regressed toward league average "
        "to reduce early-season noise.  Adj. UVS = w × UVS_norm + (1 − w) × 50,  "
        "where w = PA / (PA + 120).",
        color="info", className="mb-3"
    ) if is_live else html.Div()

    formula_box = dbc.Card([
        dbc.CardHeader(html.H5("UVS — Undervaluation Score", className="mb-0")),
        dbc.CardBody([
            live_note,
            html.P("UVS = 0.25·EPI + 0.20·CQI + 0.15·PDI + 0.15·RPI + 0.10·SE + 0.10·LA",
                   style={"fontFamily": "monospace", "fontSize": "14px",
                           "backgroundColor": "#f0f0f0", "padding": "12px",
                           "borderRadius": "6px", "marginBottom": "20px"}),
            html.Div([
                html.P([html.Strong("EPI (Expected Performance Index): "),
                        "mean( z(xwOBA), z(xSLG), z(xBA), z(xISO) )"],
                       style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "4px"}),
                html.P([html.Strong("CQI (Contact Quality Index): "),
                        "mean( z(Barrel%), z(HardHit%), z(Exit Velo), z(Sweet Spot%) )"],
                       style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "4px"}),
                html.P([html.Strong("PDI (Plate Discipline Index): "),
                        "z(BB%) − z(K%) − z(O-Swing%) + z(Z-Contact%) + z(Contact%)"],
                       style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "4px"}),
                html.P([html.Strong("RPI (Run Production Index): "),
                        "mean( z(wRC+), z(wOBA), z(OPS), z(ISO), z(R), z(RBI) )"],
                       style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "4px"}),
                html.P([html.Strong("SE  (Salary Efficiency): "), "z(WAR / $1M)"],
                       style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "4px"}),
                html.P([html.Strong("LA  (Luck Adjustment): "),
                        "mean( z(xwOBA−wOBA), z(xBA−BA), z(xSLG−SLG) )"],
                       style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "4px"}),
            ], style={"backgroundColor": "#f8f9fa", "padding": "14px",
                       "borderRadius": "6px", "marginBottom": "20px"}),
            html.H6("Leaderboard"),
            rank_table,
        ])
    ], className="mb-4")

    return dbc.Container([dbc.Row([dbc.Col([formula_box], width=12)])], fluid=True)


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print("\n" + "=" * 60)
    print("Undervalued MLB Hitters Dashboard")
    print(f"http://localhost:{port}")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
