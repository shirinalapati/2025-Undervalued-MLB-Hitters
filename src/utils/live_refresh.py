"""
Background scheduler for 2026 live-season data refresh.

Two-tier refresh while the dashboard is running:
  1. MLB Stats API (fast) — PA, HR, R, RBI, BA, OBP, SLG every ~45 min
  2. Full Statcast scrape — xwOBA, Barrel%, discipline, spray every ~2 h

Disable with LIVE_REFRESH_ENABLED=0.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FULL_FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_2026_data.py"
MLB_LIVE_SCRIPT = PROJECT_ROOT / "scripts" / "refresh_mlb_live_stats.py"
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "comprehensive_stats_2026.csv"
DATA_TS = PROJECT_ROOT / "data" / "processed" / "last_updated_2026.txt"
MLB_LIVE_TS = PROJECT_ROOT / "data" / "processed" / "last_updated_mlb_live_2026.txt"
SEASON_START = date(2026, 3, 1)
SEASON_END = date(2026, 10, 5)

_refresh_lock = threading.Lock()
_last_full_refresh_utc: datetime | None = None
_last_full_refresh_ok: bool | None = None
_last_mlb_refresh_utc: datetime | None = None
_last_mlb_refresh_ok: bool | None = None
_scheduler_started = False
_git_sync_done = False


def is_enabled() -> bool:
    return os.environ.get("LIVE_REFRESH_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def git_sync_enabled() -> bool:
    return os.environ.get("LIVE_GIT_SYNC", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def refresh_interval_seconds() -> int:
    hours = float(os.environ.get("LIVE_REFRESH_HOURS", "2"))
    return max(int(hours * 3600), 1800)


def mlb_refresh_interval_seconds() -> int:
    minutes = float(os.environ.get("LIVE_MLB_REFRESH_MINUTES", "45"))
    return max(int(minutes * 60), 900)


def poll_interval_ms() -> int:
    minutes = float(os.environ.get("LIVE_DATA_POLL_MINUTES", "2"))
    return max(int(minutes * 60_000), 60_000)


def in_season() -> bool:
    today = date.today()
    return SEASON_START <= today <= SEASON_END


def _git_remote() -> str | None:
    for name in ("hitters", "origin"):
        try:
            r = subprocess.run(
                ["git", "remote", "get-url", name],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                return name
        except Exception:
            continue
    return None


def pull_latest_committed_data() -> bool:
    if not git_sync_enabled() or not (PROJECT_ROOT / ".git").exists():
        return False
    remote = _git_remote()
    if not remote:
        return False
    try:
        fetch = subprocess.run(
            ["git", "fetch", remote, "main"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=90,
        )
        if fetch.returncode != 0:
            return False
        checkout = subprocess.run(
            [
                "git", "checkout", f"{remote}/main", "--",
                "data/processed/comprehensive_stats_2026.csv",
                "data/processed/last_updated_2026.txt",
                "data/processed/last_updated_mlb_live_2026.txt",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if checkout.returncode == 0 and DATA_TS.exists():
            log.info("Synced 2026 data from %s/main", remote)
            return True
        return False
    except Exception as exc:
        log.debug("git sync failed: %s", exc)
        return False


def _run_script(script: Path, label: str) -> bool:
    if not script.exists():
        log.error("%s script not found: %s", label, script)
        return False
    if date.today().year > 2026 or (date.today().year == 2026 and not in_season()):
        log.info("Outside 2026 season — skipping %s", label)
        return False
    if not _refresh_lock.acquire(blocking=False):
        log.info("Refresh already in progress — skipping %s", label)
        return False
    try:
        log.info("Starting %s…", label)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=900,
        )
        ok = result.returncode == 0
        if ok:
            log.info("%s completed", label)
        else:
            tail = (result.stderr or result.stdout or "")[-800:]
            log.error("%s failed (exit %s): %s", label, result.returncode, tail)
        return ok
    except Exception as exc:
        log.exception("%s error: %s", label, exc)
        return False
    finally:
        _refresh_lock.release()


def run_fetch_pipeline() -> bool:
    global _last_full_refresh_utc, _last_full_refresh_ok
    ok = _run_script(FULL_FETCH_SCRIPT, "full Statcast refresh")
    _last_full_refresh_utc = datetime.now(timezone.utc)
    _last_full_refresh_ok = ok
    return ok


def run_mlb_live_refresh() -> bool:
    global _last_mlb_refresh_utc, _last_mlb_refresh_ok
    ok = _run_script(MLB_LIVE_SCRIPT, "MLB live stats refresh")
    _last_mlb_refresh_utc = datetime.now(timezone.utc)
    _last_mlb_refresh_ok = ok
    return ok


def _refresh_loop() -> None:
    global _git_sync_done

    if git_sync_enabled() and not _git_sync_done:
        pull_latest_committed_data()
        _git_sync_done = True

    delay = int(os.environ.get("LIVE_REFRESH_STARTUP_DELAY", "15"))
    if delay > 0:
        log.info("First MLB live refresh in %ds, full Statcast shortly after…", delay)
        time.sleep(delay)

    if in_season():
        run_mlb_live_refresh()

    last_mlb = time.time()
    last_full = time.time()
    mlb_iv = mlb_refresh_interval_seconds()
    full_iv = refresh_interval_seconds()

    while is_enabled():
        if not in_season():
            log.info("Waiting for 2026 season window (%s – %s)", SEASON_START, SEASON_END)
            time.sleep(3600)
            continue

        now = time.time()
        if now - last_mlb >= mlb_iv:
            run_mlb_live_refresh()
            last_mlb = now
        if now - last_full >= full_iv:
            run_fetch_pipeline()
            last_full = now

        time.sleep(60)


def start_background_scheduler() -> None:
    global _scheduler_started
    if _scheduler_started or not is_enabled():
        return
    _scheduler_started = True
    thread = threading.Thread(target=_refresh_loop, daemon=True, name="live-2026-refresh")
    thread.start()
    log.info(
        "Live refresh — MLB API every %dm, Statcast every %s, UI poll every %dm",
        mlb_refresh_interval_seconds() // 60,
        refresh_interval_label(),
        poll_interval_ms() // 60_000,
    )


def refresh_interval_label() -> str:
    s = refresh_interval_seconds()
    if s >= 3600:
        h = s / 3600
        return f"{int(h)}h" if h == int(h) else f"{h:.1f}h"
    return f"{max(s // 60, 1)}m"


def mlb_refresh_interval_label() -> str:
    return f"{mlb_refresh_interval_seconds() // 60}m"


def refresh_status_label() -> str:
    if not is_enabled():
        return "manual refresh only"
    mlb = mlb_refresh_interval_label()
    full = refresh_interval_label()
    parts = [f"PA live every {mlb}", f"Statcast every {full}"]
    if _last_mlb_refresh_utc and _last_mlb_refresh_ok:
        parts.append(f"PA @ {_last_mlb_refresh_utc.strftime('%H:%M UTC')} ✓")
    if _last_full_refresh_utc and _last_full_refresh_ok:
        parts.append(f"adv @ {_last_full_refresh_utc.strftime('%H:%M UTC')} ✓")
    return " · ".join(parts)
