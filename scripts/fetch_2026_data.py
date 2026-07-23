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

  Source 5 — Baseball Savant custom CSV export
             → BB%, K%, Z-Contact%, Pull%, BABIP

  Source 6 — Statcast pitch-level aggregation (cached daily)
             → Chase%, Contact%, GB%, FB%, LD%, Pull%, Oppo%

  Source 7 — pybaseball.bwar_bat (Baseball Reference WAR + salary)

  Source 8 — comprehensive_stats_2025.csv + bwar_bat(2025) salary fallbacks
  Source 9 — league minimum salary estimate for remaining pre-arb players

All Statcast tables share player_id → clean merge.
BRef matched by normalised player name.

Applies the same UVS formula as the 2025 project (raw UVS ranking,
no sample-size filter — all hitters included).

Output:
  data/processed/comprehensive_stats_2026.csv
  data/processed/last_updated_2026.txt
"""

import sys
import io
import logging
import re
import unicodedata
from pathlib import Path
from datetime import datetime, timezone, date

import pandas as pd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV     = DATA_PROCESSED / "comprehensive_stats_2026.csv"
TIMESTAMP_FILE = DATA_PROCESSED / "last_updated_2026.txt"
STATCAST_CACHE = DATA_RAW / "statcast_aggregates_2026_v3.csv"
SALARY_2025    = DATA_PROCESSED / "salaries_2025.csv"
COMP_STATS_2025 = DATA_PROCESSED / "comprehensive_stats_2025.csv"
LEAGUE_MIN_SALARY_M = 0.78   # 2026 MLB minimum ($780k) for pre-arb estimates

SEASON = 2026
MIN_PA = 0   # include every hitter returned by Statcast (no sample-size floor)

_SWING = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "foul_bunt", "hit_into_play", "missed_bunt",
}
_CONTACT = {"foul", "foul_tip", "foul_bunt", "hit_into_play"}
_SPRAY_HOME_X = 125.42  # Statcast home-plate x coordinate (feet)

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
    """Statcast percentile ranks (0–100 scale) — not actual rate stats."""
    log.info("  Fetching Statcast percentile ranks…")
    try:
        df = pyb.statcast_batter_percentile_ranks(year)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "chase_percent": "chase_percentile",
            "k_percent":     "k_percentile",
            "bb_percent":    "bb_percentile",
            "whiff_percent": "whiff_percentile",
            "sprint_speed":  "sprint_speed",
            "oaa":           "oaa",
        })
        keep = ["player_id", "k_percentile", "bb_percentile", "chase_percentile",
                "whiff_percentile", "sprint_speed", "oaa"]
        df = df[[c for c in keep if c in df.columns]]
        log.info(f"    → {len(df)} rows")
        return df
    except Exception as exc:
        log.warning(f"    Percentile ranks failed: {exc}")
        return pd.DataFrame()


def fetch_savant_custom(year: int) -> pd.DataFrame:
    """Savant custom leaderboard CSV — actual rate stats not in percentile API."""
    log.info("  Fetching Savant custom leaderboard CSV…")
    selections = (
        "player_id,pa,bb_percent,k_percent,iz_contact_percent,"
        "pull_percent,opo_percent,babip"
    )
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/custom?"
        f"type=batter&year={year}&filter=&min=1&selections={selections}"
        f"&chart=false&x=pa&y=ops&r=no&csv=true"
    )
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df = df.loc[:, ~df.columns.duplicated()]
        if "iz_contact_percent" in df.columns:
            df = df.rename(columns={"iz_contact_percent": "z_contact_percent"})
        if "opo_percent" in df.columns:
            df = df.rename(columns={"opo_percent": "oppo_percent"})
        keep = ["player_id", "bb_percent", "k_percent", "z_contact_percent",
                "pull_percent", "oppo_percent", "babip"]
        df = df[[c for c in keep if c in df.columns]]
        log.info(f"    → {len(df)} rows")
        return df
    except Exception as exc:
        log.warning(f"    Savant custom CSV failed: {exc}")
        return pd.DataFrame()


def fetch_statcast_aggregates(pyb, year: int) -> pd.DataFrame:
    """
    Aggregate discipline + batted-ball rates from pitch-level Statcast.
    Cached for 24h because the query is slow (~15–30s).
    """
    if STATCAST_CACHE.exists():
        age = datetime.now(timezone.utc).timestamp() - STATCAST_CACHE.stat().st_mtime
        if age < 86400:
            log.info(f"  Using cached Statcast aggregates ({STATCAST_CACHE.name})")
            return pd.read_csv(STATCAST_CACHE)

    log.info("  Aggregating Statcast pitch data (discipline + GB/FB/LD + Pull/Oppo)…")
    try:
        start = f"{year}-03-01"
        end = date.today().isoformat()
        raw = pyb.statcast(start, end)
        if raw is None or raw.empty:
            return pd.DataFrame()

        raw = raw[raw.get("game_type", "R") == "R"].copy()
        raw["in_zone"] = raw["zone"].between(1, 9, inclusive="both")
        raw["is_swing"] = raw["description"].isin(_SWING)
        raw["is_contact"] = raw["description"].isin(_CONTACT)

        rows = []
        for batter, grp in raw.groupby("batter"):
            outside = ~grp["in_zone"]
            o_pitches = outside.sum()
            o_swings = (outside & grp["is_swing"]).sum()
            in_swings = (grp["in_zone"] & grp["is_swing"]).sum()
            in_contact = (grp["in_zone"] & grp["is_swing"] & grp["is_contact"]).sum()
            all_swings = grp["is_swing"].sum()
            all_contact = (grp["is_swing"] & grp["is_contact"]).sum()

            bbe = grp[grp["bb_type"].notna()]
            bb_counts = bbe["bb_type"].value_counts()
            bbe_total = len(bbe)

            pull_n = oppo_n = 0
            spray_hits = bbe[
                bbe["hc_x"].notna() & bbe["hc_y"].notna() & bbe["stand"].isin(["R", "L"])
            ].copy()
            if len(spray_hits):
                spray_hits["spray_deg"] = np.degrees(
                    np.arctan2(spray_hits["hc_x"] - _SPRAY_HOME_X, spray_hits["hc_y"])
                )
                is_r = spray_hits["stand"] == "R"
                pull_n = int(
                    ((is_r & (spray_hits["spray_deg"] < -15))
                     | (~is_r & (spray_hits["spray_deg"] > 15))).sum()
                )
                oppo_n = int(
                    ((is_r & (spray_hits["spray_deg"] > 15))
                     | (~is_r & (spray_hits["spray_deg"] < -15))).sum()
                )

            rows.append({
                "player_id": int(batter),
                "o_swing_percent": round(o_swings / o_pitches * 100, 1) if o_pitches else np.nan,
                "z_contact_percent_sc": round(in_contact / in_swings * 100, 1) if in_swings else np.nan,
                "contact_percent": round(all_contact / all_swings * 100, 1) if all_swings else np.nan,
                "gb_percent": round(bb_counts.get("ground_ball", 0) / bbe_total * 100, 1) if bbe_total else np.nan,
                "fb_percent": round(bb_counts.get("fly_ball", 0) / bbe_total * 100, 1) if bbe_total else np.nan,
                "ld_percent": round(bb_counts.get("line_drive", 0) / bbe_total * 100, 1) if bbe_total else np.nan,
                "pull_percent_sc": round(pull_n / bbe_total * 100, 1) if bbe_total else np.nan,
                "oppo_percent_sc": round(oppo_n / bbe_total * 100, 1) if bbe_total else np.nan,
            })

        out = pd.DataFrame(rows)
        out.to_csv(STATCAST_CACHE, index=False)
        log.info(f"    → {len(out)} batters aggregated (cached)")
        return out
    except Exception as exc:
        log.warning(f"    Statcast aggregation failed: {exc}")
        if STATCAST_CACHE.exists():
            log.info("    Falling back to stale Statcast cache")
            return pd.read_csv(STATCAST_CACHE)
        return pd.DataFrame()


def fetch_bref_war(pyb, year: int) -> pd.DataFrame:
    """Baseball Reference WAR + salary via pybaseball.bwar_bat (aggregated by player)."""
    log.info("  Fetching Baseball Reference WAR + salary (bwar_bat)…")
    try:
        raw = pyb.bwar_bat()
        if raw is None or raw.empty:
            return pd.DataFrame()
        hitters = raw[(raw["year_ID"] == year) & (raw["pitcher"] == "N")].copy()
        if hitters.empty:
            return pd.DataFrame()
        hitters["player_id"] = hitters["mlb_ID"].astype(int)
        agg = hitters.groupby("player_id", as_index=False).agg(
            war=("WAR", "sum"),
            salary_bref=("salary", lambda s: s.dropna().max() if s.notna().any() else np.nan),
        )
        agg["salary_2026"] = (agg["salary_bref"] / 1_000_000).round(3)
        agg["war"] = agg["war"].round(2)
        agg.drop(columns=["salary_bref"], inplace=True)
        log.info(f"    → {len(agg)} rows  |  WAR filled {agg['war'].notna().sum()}  |  salary filled {agg['salary_2026'].notna().sum()}")
        return agg
    except Exception as exc:
        log.warning(f"    bwar_bat failed: {exc}")
        return pd.DataFrame()


def load_salary_fallback() -> pd.DataFrame:
    """2025 full-season salaries (millions) from comprehensive_stats_2025.csv."""
    path = COMP_STATS_2025 if COMP_STATS_2025.exists() else SALARY_2025
    if not path.exists():
        return pd.DataFrame()
    sal = pd.read_csv(path)
    name_col = next((c for c in ["name", "Name"] if c in sal.columns), None)
    sal_col = next((c for c in ["salary_2025", "salary"] if c in sal.columns), None)
    if not name_col or not sal_col:
        return pd.DataFrame()
    sal["_norm_name"] = _norm_name(sal[name_col])
    sal = sal.rename(columns={sal_col: "salary_2026_fallback"})
    out = sal[["_norm_name", "salary_2026_fallback"]].dropna(subset=["salary_2026_fallback"])
    log.info(f"    Salary fallback loaded from {path.name}: {len(out)} rows")
    return out.drop_duplicates("_norm_name")


def fetch_bref_prior_salary(pyb, year: int) -> pd.DataFrame:
    """Prior-year BRef salary by player_id (millions) when current year is blank."""
    log.info(f"  Fetching Baseball Reference {year} salary fallback…")
    try:
        raw = pyb.bwar_bat()
        if raw is None or raw.empty:
            return pd.DataFrame()
        hitters = raw[(raw["year_ID"] == year) & (raw["pitcher"] == "N")].copy()
        if hitters.empty:
            return pd.DataFrame()
        hitters["player_id"] = hitters["mlb_ID"].astype(int)
        agg = hitters.groupby("player_id", as_index=False).agg(
            salary_prior=("salary", lambda s: s.dropna().max() if s.notna().any() else np.nan),
        )
        agg["salary_2026_fallback_bref"] = (agg["salary_prior"] / 1_000_000).round(3)
        agg = agg.dropna(subset=["salary_2026_fallback_bref"])
        log.info(f"    → {len(agg)} prior-year salaries")
        return agg[["player_id", "salary_2026_fallback_bref"]]
    except Exception as exc:
        log.warning(f"    Prior-year BRef salary failed: {exc}")
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
            df["_bref_name"] = _norm_name(df["Name"])
        keep = ["_bref_name", "HR", "R", "RBI", "AB", "H", "2B", "3B",
                "OBP", "SLG", "ISO", "OPS", "BA_bref", "SB", "SF",
                "k_bref", "bb_bref"]
        df = df[[c for c in keep if c in df.columns]]
        log.info(f"    → {len(df)} rows")
        return df
    except Exception as exc:
        log.warning(f"    BRef fetch failed: {exc}")
        return pd.DataFrame()


# ── Merge ──────────────────────────────────────────────────────────────────

def _fix_encoding(s: str) -> str:
    """Decode pybaseball BRef mojibake (\\xc3\\xa9 → é, \\' → ')."""
    s = str(s).replace("\\'", "'")
    if "\\x" in s:
        try:
            s = s.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass
    try:
        s = s.encode("latin-1").decode("utf-8")
    except Exception:
        pass
    return s


def _norm_name(s: pd.Series) -> pd.Series:
    """'Alvarez, Yordan' / 'Jos\\xc3\\xa9 Ram\\xc3\\xadrez' → 'yordan alvarez'."""

    def _flip(x):
        x = _fix_encoding(x).strip().lower()
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
        if ", " in x:
            parts = x.split(", ", 1)
            x = f"{parts[1]} {parts[0]}"
        x = re.sub(r"\b(jr|sr|ii|iii|iv)\.?\b", "", x)
        x = x.replace("'", "").replace(".", "")
        return re.sub(r"\s+", " ", x).strip()

    return s.apply(_flip)


def build_dataset(expected, ev, percentile, bref, savant, sc_agg, bref_war, salary_fb, salary_prior) -> pd.DataFrame:
    """Merge all sources on player_id (Statcast) then name-match BRef/salary."""
    if expected.empty:
        log.error("No expected stats — cannot build dataset.")
        return pd.DataFrame()

    df = expected.copy()

    if not ev.empty and "player_id" in ev.columns:
        df = df.merge(ev, on="player_id", how="left")
        log.info(f"  After EV merge: {len(df)} rows")

    if not percentile.empty and "player_id" in percentile.columns:
        df = df.merge(percentile, on="player_id", how="left")
        log.info(f"  After percentile merge: {len(df)} rows")

    if not savant.empty and "player_id" in savant.columns:
        df = df.merge(savant, on="player_id", how="left", suffixes=("", "_sav"))
        log.info(f"  After Savant custom merge: {len(df)} rows")

    if not sc_agg.empty and "player_id" in sc_agg.columns:
        df = df.merge(sc_agg, on="player_id", how="left", suffixes=("", "_sc"))
        log.info(f"  After Statcast aggregate merge: {len(df)} rows")

    if not bref_war.empty and "player_id" in bref_war.columns:
        df = df.merge(bref_war, on="player_id", how="left", suffixes=("", "_bwar"))
        log.info(f"  After BRef WAR merge: {df['war'].notna().sum()} WAR values")

    if "last_name, first_name" in df.columns:
        df["_norm_name"] = _norm_name(df["last_name, first_name"])
    elif "player_name" in df.columns:
        df["_norm_name"] = df["player_name"].str.lower().str.strip()
    else:
        df["_norm_name"] = ""

    if not bref.empty and "_bref_name" in bref.columns:
        df = df.merge(bref, left_on="_norm_name", right_on="_bref_name", how="left")
        log.info(f"  After BRef merge: {len(df)} rows")

    if not salary_fb.empty and "_norm_name" in salary_fb.columns:
        df = df.merge(salary_fb, on="_norm_name", how="left")
        if "salary_2026_fallback" in df.columns:
            if "salary_2026" not in df.columns:
                df["salary_2026"] = df["salary_2026_fallback"]
            else:
                df["salary_2026"] = df["salary_2026"].fillna(df["salary_2026_fallback"])
            df.drop(columns=["salary_2026_fallback"], errors="ignore")
        log.info(f"  After comp-2025 salary merge: {df['salary_2026'].notna().sum()} salaries")

    if not salary_prior.empty and "player_id" in salary_prior.columns:
        df = df.merge(salary_prior, on="player_id", how="left")
        if "salary_2026_fallback_bref" in df.columns:
            df["salary_2026"] = df["salary_2026"].fillna(df["salary_2026_fallback_bref"])
            df.drop(columns=["salary_2026_fallback_bref"], errors="ignore")
        log.info(f"  After prior-year BRef salary merge: {df['salary_2026'].notna().sum()} salaries")

    if "salary_2026" in df.columns:
        missing_sal = df["salary_2026"].isna()
        has_pa = df.get("pa", pd.Series(True, index=df.index)).fillna(0) > 0
        df.loc[missing_sal & has_pa, "salary_2026"] = LEAGUE_MIN_SALARY_M
        log.info(f"  After league-minimum fill: {df['salary_2026'].notna().sum()} salaries "
                 f"({(missing_sal & has_pa).sum()} estimated at ${LEAGUE_MIN_SALARY_M}M)")

    df.drop(columns=["_norm_name", "_bref_name"], errors="ignore", inplace=True)
    return df


# ── Derived columns ────────────────────────────────────────────────────────

def _fill_zero_bbe_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Default missing fields for tiny samples or zero balls in play."""
    df = df.copy()
    if "bip" in df.columns:
        bip = pd.to_numeric(df["bip"], errors="coerce").fillna(0)
    else:
        bip = pd.Series(0, index=df.index)
    ab = pd.to_numeric(df.get("ab", df.get("AB")), errors="coerce").fillna(0)
    pa = pd.to_numeric(df.get("pa", df.get("PA")), errors="coerce").fillna(0)
    h = pd.to_numeric(df.get("h", df.get("H")), errors="coerce").fillna(0)
    sparse = bip.le(0) | pa.lt(5)

    if "pull_percent_sc" in df.columns and "pull_percent" in df.columns:
        df["pull_percent"] = df["pull_percent"].fillna(df["pull_percent_sc"])
    if "oppo_percent_sc" in df.columns and "oppo_percent" in df.columns:
        df["oppo_percent"] = df["oppo_percent"].fillna(df["oppo_percent_sc"])
    if "z_contact_percent_sc" in df.columns and "z_contact_percent" in df.columns:
        df["z_contact_percent"] = df["z_contact_percent"].fillna(df["z_contact_percent_sc"])

    for col in [
        "barrel_batted_rate", "hard_hit_percent", "avg_exit_velocity",
        "sweet_spot_percent", "woba", "xwoba", "xba", "xslg", "xiso", "xhr",
        "z_contact_percent", "contact_percent", "o_swing_percent",
        "gb_percent", "fb_percent", "ld_percent", "pull_percent", "oppo_percent",
        "babip", "wrc_plus",
    ]:
        if col in df.columns:
            df.loc[sparse, col] = df.loc[sparse, col].fillna(0.0)

    if "xiso" in df.columns and "xslg" in df.columns and "xba" in df.columns:
        need_xiso = df["xiso"].isna() & df["xslg"].notna() & df["xba"].notna()
        df.loc[need_xiso, "xiso"] = (df.loc[need_xiso, "xslg"] - df.loc[need_xiso, "xba"]).round(3)

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

    # SLG / ISO / OPS (lowercase for dashboard)
    if "SLG" in df.columns:
        df["slg"] = df["SLG"]
    if "ISO" in df.columns:
        df["iso"] = df["ISO"]
    elif "slg" in df.columns and "ba" in df.columns:
        df["iso"] = (df["slg"] - df["ba"]).round(3)
    if "OPS" in df.columns:
        df["ops"] = df["OPS"]
    elif "obp" in df.columns and "slg" in df.columns:
        df["ops"] = (df["obp"] + df["slg"]).round(3)

    # Unified HR/R/RBI/AB/H
    for upper, lower in [("HR","hr"),("R","r"),("RBI","rbi"),("AB","ab"),("H","h")]:
        if lower not in df.columns and upper in df.columns:
            df[lower] = df[upper]

    # K / BB counting (from BRef)
    if "k" not in df.columns and "k_bref" in df.columns:
        df["k"] = df["k_bref"]
    if "bb" not in df.columns and "bb_bref" in df.columns:
        df["bb"] = df["bb_bref"]

    # Prefer Savant / Statcast aggregate columns; fill gaps from BRef counts
    pa = df.get("pa", pd.Series(np.nan, index=df.index))
    bb = df.get("bb", df.get("bb_bref"))
    k = df.get("k", df.get("k_bref"))
    if "bb_percent" not in df.columns or df["bb_percent"].isna().sum() > len(df) * 0.3:
        if bb is not None:
            df["bb_percent"] = (bb / pa.replace(0, np.nan) * 100).round(1)
    if "k_percent" not in df.columns or df["k_percent"].isna().sum() > len(df) * 0.3:
        if k is not None:
            df["k_percent"] = (k / pa.replace(0, np.nan) * 100).round(1)

    # Pull%: Savant official rate first, Statcast spray aggregation as fallback
    if "pull_percent_sc" in df.columns:
        if "pull_percent" not in df.columns:
            df["pull_percent"] = df["pull_percent_sc"]
        else:
            df["pull_percent"] = df["pull_percent"].fillna(df["pull_percent_sc"])
        df.drop(columns=["pull_percent_sc"], errors="ignore")

    # Oppo%: Savant first, Statcast spray aggregation as fallback
    if "oppo_percent_sc" in df.columns:
        if "oppo_percent" not in df.columns:
            df["oppo_percent"] = df["oppo_percent_sc"]
        else:
            df["oppo_percent"] = df["oppo_percent"].fillna(df["oppo_percent_sc"])
        df.drop(columns=["oppo_percent_sc"], errors="ignore")

    # Z-Contact%: Savant first, pitch-level Statcast fallback
    if "z_contact_percent_sc" in df.columns:
        if "z_contact_percent" not in df.columns:
            df["z_contact_percent"] = df["z_contact_percent_sc"]
        else:
            df["z_contact_percent"] = df["z_contact_percent"].fillna(
                df["z_contact_percent_sc"]
            )
        df.drop(columns=["z_contact_percent_sc"], errors="ignore")

    # Discipline / batted-ball from Statcast aggregates when still missing
    for col in ["o_swing_percent", "contact_percent", "gb_percent", "fb_percent",
                "ld_percent"]:
        sc_col = f"{col}_sc" if f"{col}_sc" in df.columns else col
        if sc_col in df.columns and sc_col != col:
            if col not in df.columns or df[col].isna().all():
                df[col] = df[sc_col]
            else:
                df[col] = df[col].fillna(df[sc_col])
            df.drop(columns=[sc_col], errors="ignore")

    # BABIP: prefer Savant, else compute from BRef
    if "babip" not in df.columns or df["babip"].isna().all():
        if all(c in df.columns for c in ["H", "HR", "AB"]):
            h, hr, ab = df["H"], df["HR"], df["AB"]
            so = df.get("k_bref", df.get("k", pd.Series(0, index=df.index))).fillna(0)
            sf = df.get("SF", pd.Series(0, index=df.index)).fillna(0)
            denom = ab - so - hr + sf
            df["babip"] = ((h - hr) / denom.replace(0, np.nan)).round(3)

    # xHR ≈ HR × (xSLG / SLG); fallback when SLG is 0/missing
    hr = df.get("HR", df.get("hr"))
    xslg = df.get("xslg")
    slg = df.get("SLG", df.get("slg"))
    if hr is not None and xslg is not None:
        df["xhr"] = np.nan
        if slg is not None:
            valid = hr.notna() & xslg.notna() & slg.notna() & (slg > 0)
            df.loc[valid, "xhr"] = (hr[valid] * (xslg[valid] / slg[valid])).round(2)
        zero_slg = hr.notna() & xslg.notna() & (slg.isna() | (slg <= 0))
        df.loc[zero_slg, "xhr"] = (hr[zero_slg] * (xslg[zero_slg] / 0.410)).round(2)
        df.loc[hr.notna() & (hr == 0), "xhr"] = 0.0

    # wRC+ from wOBA when FanGraphs unavailable (Statcast wOBA vs league avg)
    LG_WOBA = 0.316
    if "woba" in df.columns and ("wrc_plus" not in df.columns or df["wrc_plus"].isna().all()):
        df["wrc_plus"] = ((df["woba"] / LG_WOBA) * 100).round(1)

    # WAR / $1M — uses Baseball Reference WAR + salary (millions)
    if "salary_2026" in df.columns and "war" in df.columns:
        valid = df["salary_2026"].notna() & (df["salary_2026"] > 0) & df["war"].notna()
        df["war_per_salary"] = np.where(
            valid, (df["war"] / (df["salary_2026"] + 0.1)).round(3), np.nan
        )

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

    return _fill_zero_bbe_defaults(df)


# ── Main ───────────────────────────────────────────────────────────────────

def write_timestamp() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    TIMESTAMP_FILE.write_text(ts)
    log.info(f"Timestamp written: {ts}")


def main() -> None:
    log.info("=" * 60)
    log.info("2026 MLB Live Season Data Pipeline")
    log.info("Sources: Statcast + Savant + BRef WAR + pitch-level aggregates")
    log.info("=" * 60)

    pyb = _pyb()

    log.info("Step 1: Fetching data…")
    expected   = fetch_expected_stats(pyb, SEASON)
    ev         = fetch_exit_velo(pyb, SEASON)
    percentile = fetch_percentile_ranks(pyb, SEASON)
    savant     = fetch_savant_custom(SEASON)
    sc_agg     = fetch_statcast_aggregates(pyb, SEASON)
    bref       = fetch_bref_batting(pyb, SEASON)
    bref_war   = fetch_bref_war(pyb, SEASON)
    salary_fb  = load_salary_fallback()
    salary_prior = fetch_bref_prior_salary(pyb, SEASON - 1)

    if expected.empty:
        log.error("Cannot continue without expected stats. Aborting.")
        sys.exit(1)

    log.info("Step 2: Merging sources…")
    df = build_dataset(expected, ev, percentile, bref, savant, sc_agg, bref_war, salary_fb, salary_prior)

    # Optional PA filter (disabled — include all hitters)
    if MIN_PA > 0 and "pa" in df.columns:
        df = df[df["pa"] >= MIN_PA].copy()
    log.info(f"  → {len(df)} hitters (no sample-size filter)")

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

    # ── Step 4: Sort & rank (same raw UVS as 2025) ─────────────────────
    sort_col = "uvs"
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
               "o_swing_percent","z_contact_percent","contact_percent",
               "gb_percent","fb_percent","ld_percent","pull_percent","oppo_percent",
               "xhr","babip","war","salary_2026","wrc_plus"] if c in df.columns}
    for col, cnt in filled.items():
        log.info(f"  {col}: {cnt}/{len(df)} filled")

    top5 = df.head(5)
    name_col = next((c for c in ["name","last_name, first_name"] if c in top5.columns), None)
    if name_col:
        log.info(f"\n  Top 5 ({sort_col}):")
        for _, row in top5.iterrows():
            pa  = int(row["pa"]) if "pa" in row else "?"
            score = round(row.get(sort_col, 0), 3)
            log.info(f"    {row[name_col]}  PA={pa}  {sort_col}={score}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
