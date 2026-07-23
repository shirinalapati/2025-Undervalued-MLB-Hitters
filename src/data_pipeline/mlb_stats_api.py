"""
Fetch season hitting stats from the public MLB Stats API (statsapi.mlb.com).

No API key required. Used for near-live PA, counting stats, and rate stats
while Statcast advanced metrics refresh on a slower schedule.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"

# Batch size for /people?personIds=…&hydrate=stats(…)
PEOPLE_BATCH_SIZE = 50

# MLB API stat key → our CSV column (lowercase dashboard id)
MLB_TO_COL = {
    "plateAppearances": "pa",
    "atBats": "ab",
    "hits": "h",
    "doubles": "doubles",
    "triples": "triples",
    "homeRuns": "hr",
    "runs": "r",
    "rbi": "rbi",
    "baseOnBalls": "bb",
    "strikeOuts": "k",
    "stolenBases": "sb",
    "hitByPitch": "hbp",
    "sacFlies": "sf",
    "sacBunts": "sh",
    "avg": "ba",
    "obp": "obp",
    "slg": "slg",
    "ops": "ops",
    "babip": "babip",
}

# Also mirror to uppercase where the 2025 pipeline expects them
UPPER_MIRROR = {
    "ab": "AB", "h": "H", "hr": "HR", "r": "R", "rbi": "RBI",
    "bb": "BB", "k": "SO", "obp": "OBP", "slg": "SLG", "ops": "OPS",
    "babip": "BABIP", "ba": "BA",
}


def _parse_split(split: dict[str, Any]) -> dict[str, Any] | None:
    player = split.get("player") or {}
    pid = player.get("id")
    if pid is None:
        return None
    row: dict[str, Any] = {
        "player_id": int(pid),
        "mlb_full_name": player.get("fullName"),
    }
    stat = split.get("stat") or {}
    for mlb_key, col in MLB_TO_COL.items():
        val = stat.get(mlb_key)
        if val is not None and val != "":
            row[col] = val
    return row


def fetch_season_hitting(season: int, sport_id: int = 1) -> pd.DataFrame:
    """
    Pull regular-season hitting stats for all players with pagination.
    Returns DataFrame keyed by player_id.
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            "stats": "season",
            "group": "hitting",
            "season": season,
            "sportId": sport_id,
            "limit": page_size,
            "offset": offset,
        }
        resp = requests.get(MLB_STATS_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        stats = payload.get("stats") or []
        if not stats:
            break
        splits = stats[0].get("splits") or []
        if not splits:
            break
        for split in splits:
            row = _parse_split(split)
            if row:
                rows.append(row)
        if len(splits) < page_size:
            break
        offset += page_size

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["player_id"] = df["player_id"].astype(int)

    # Numeric coercion
    for col in set(MLB_TO_COL.values()):
        if col in df.columns and col not in ("ba", "obp", "slg", "ops", "babip"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("ba", "obp", "slg", "ops", "babip"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("MLB Stats API (season endpoint): %d hitters for %s", len(df), season)
    return df


def fetch_hitting_for_player_ids(
    player_ids: list[int],
    season: int,
) -> pd.DataFrame:
    """
    Fetch season hitting stats for specific MLBAM ids (batch /people hydrate).
    Uses the existing Statcast player list so every tracked hitter gets live PA.
    """
    ids = sorted({int(x) for x in player_ids if pd.notna(x) and int(x) > 0})
    if not ids:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    hydrate = f"stats(group=hitting,type=season,season={season})"

    for i in range(0, len(ids), PEOPLE_BATCH_SIZE):
        batch = ids[i : i + PEOPLE_BATCH_SIZE]
        resp = requests.get(
            MLB_PEOPLE_URL,
            params={"personIds": ",".join(str(x) for x in batch), "hydrate": hydrate},
            timeout=90,
        )
        resp.raise_for_status()
        for person in resp.json().get("people") or []:
            pid = person.get("id")
            if pid is None:
                continue
            stats_list = person.get("stats") or []
            if not stats_list:
                continue
            splits = stats_list[0].get("splits") or []
            if not splits:
                continue
            split = {"player": {"id": pid, "fullName": person.get("fullName")}, "stat": splits[0].get("stat") or {}}
            row = _parse_split(split)
            if row and row.get("pa", 0):
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["player_id"] = df["player_id"].astype(int)
    for col in set(MLB_TO_COL.values()):
        if col in df.columns and col not in ("ba", "obp", "slg", "ops", "babip"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("ba", "obp", "slg", "ops", "babip"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("MLB Stats API (by player_id): %d/%d hitters with PA for %s", len(df), len(ids), season)
    return df


def fetch_live_hitting(season: int, player_ids: list[int] | None = None) -> pd.DataFrame:
    """Prefer batched player-id lookup; fall back to season leaderboard endpoint."""
    if player_ids:
        df = fetch_hitting_for_player_ids(player_ids, season)
        if not df.empty:
            return df
    return fetch_season_hitting(season)


def merge_live_stats(existing: pd.DataFrame, mlb: pd.DataFrame) -> pd.DataFrame:
    """
    Update counting / rate stats from MLB API without overwriting Statcast
    advanced columns (xwOBA, Barrel%, discipline, spray, etc.).
    """
    if existing.empty or mlb.empty or "player_id" not in existing.columns:
        return existing

    df = existing.copy()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    mlb = mlb.copy()
    mlb["player_id"] = pd.to_numeric(mlb["player_id"], errors="coerce")

    live_cols = [c for c in MLB_TO_COL.values() if c in mlb.columns]
    mlb_sub = mlb[["player_id"] + live_cols].drop_duplicates("player_id", keep="last")
    merged = df.merge(mlb_sub, on="player_id", how="left", suffixes= ("", "_mlb_live"))

    updated = 0
    for col in live_cols:
        live_col = f"{col}_mlb_live"
        if live_col not in merged.columns:
            continue
        mask = merged[live_col].notna()
        merged.loc[mask, col] = merged.loc[mask, live_col]
        updated += int(mask.sum())
        merged.drop(columns=[live_col], inplace=True)

        upper = UPPER_MIRROR.get(col)
        if upper:
            merged[upper] = merged[col]

    # ISO from updated SLG − BA
    if "slg" in merged.columns and "ba" in merged.columns:
        merged["iso"] = (merged["slg"] - merged["ba"]).round(3)
        merged["ISO"] = merged["iso"]

    # BB% / K% from counting stats when PA available
    pa = pd.to_numeric(merged.get("pa"), errors="coerce")
    bb = pd.to_numeric(merged.get("bb"), errors="coerce")
    k = pd.to_numeric(merged.get("k"), errors="coerce")
    valid_pa = pa.notna() & pa.gt(0)
    if valid_pa.any():
        merged.loc[valid_pa, "bb_percent"] = (bb[valid_pa] / pa[valid_pa] * 100).round(1)
        merged.loc[valid_pa, "k_percent"] = (k[valid_pa] / pa[valid_pa] * 100).round(1)

    # Luck differentials refresh when actuals move but expected stay from Statcast
    for actual, expected, out in [
        ("ba", "xba", "xba_minus_ba"),
        ("slg", "xslg", "xslg_minus_slg"),
        ("woba", "xwoba", "xwoba_minus_woba"),
    ]:
        if actual in merged.columns and expected in merged.columns:
            merged[out] = (merged[expected] - merged[actual]).round(3)

    # wRC+ estimate from Statcast wOBA (wOBA itself lags until full Statcast refresh)
    lg_woba = 0.316
    if "woba" in merged.columns:
        merged["wrc_plus"] = ((merged["woba"] / lg_woba) * 100).round(1)

    log.info("MLB live merge: refreshed counting stats for %d player-stat updates", updated)
    return merged
