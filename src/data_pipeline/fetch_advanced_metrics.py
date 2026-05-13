"""
Enhanced data pipeline for fetching comprehensive Statcast and advanced metrics.

This module collects all advanced metrics needed for identifying undervalued hitters:
- Quality of contact metrics (Barrel%, HardHit%, Exit Velocity, Sweet Spot%)
- Expected outcomes (xBA, xSLG, xwOBA, xISO)
- Plate discipline (Chase%, BB%, K%, Z-Contact%, Contact%)
- Batted ball profile (GB%, FB%, LD%, Pull%, Oppo%)
- Run production (wRC+, wOBA, OPS, ISO, R, RBI)
- Value metrics (WAR, WAR per $1M, Salary)

The module enforces a minimum of 200 plate appearances for all hitters
and merges data from multiple sources (Statcast, FanGraphs, salary databases).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))

try:
    import pybaseball as pyb
    from pybaseball import statcast_batter, statcast_pitcher
except ImportError:
    print("Warning: pybaseball not installed. Install with: pip install pybaseball")
    pyb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def fetch_comprehensive_batting_stats(year: int = 2025, min_pa: int = 200) -> pd.DataFrame:
    """
    Fetch comprehensive batting statistics including all advanced metrics.
    
    Args:
        year: Season year
        min_pa: Minimum plate appearances to include (default 200 for full-season;
                pass a lower value like 10 for live-season mode where reliability
                weighting handles small-sample noise).
        
    Returns:
        DataFrame with comprehensive batting metrics
    """
    if pyb is None:
        logger.error("pybaseball not available")
        return pd.DataFrame()
    
    logger.info(f"Fetching comprehensive batting stats for {year} (min_pa={min_pa})...")
    
    try:
        # Fetch Statcast expected stats leaderboard (has xwOBA, xBA, etc.)
        # minPA=1 so pybaseball returns all batters; we apply our own PA floor below.
        df = pyb.statcast_batter_expected_stats(year, minPA=1)
        
        if df is None or df.empty:
            logger.warning(f"No Statcast batting data for {year}")
            return pd.DataFrame()
        
        logger.info(f"Fetched {len(df)} hitters from Statcast expected stats")
        
        # Apply configurable PA floor
        if 'pa' in df.columns:
            before_filter = len(df)
            df = df[df['pa'] >= min_pa]
            logger.info(f"Filtered Statcast data: {before_filter} -> {len(df)} players with >= {min_pa} PA")
        
        # Fetch FanGraphs data for WAR, wRC+, and quality of contact metrics
        try:
            # Use the same PA floor for FanGraphs (qual=0 lets all players through)
            fg_batting = pyb.batting_stats(year, qual=min_pa if min_pa > 1 else 0)
            if fg_batting is not None and not fg_batting.empty:
                logger.info(f"Fetched {len(fg_batting)} hitters from FanGraphs")
                
                # Try to merge on multiple possible keys
                merge_keys = None
                for key_set in [['Name', 'player_id'], ['Name'], ['last_name, first_name']]:
                    if all(k in df.columns for k in key_set) and all(k in fg_batting.columns for k in key_set):
                        merge_keys = key_set
                        break
                
                if merge_keys:
                    df = pd.merge(
                        df,
                        fg_batting,
                        on=merge_keys,
                        how='left',
                        suffixes=('', '_fg')
                    )
                    logger.info(f"Merged FanGraphs data using keys: {merge_keys}")
                else:
                    # Try fuzzy matching on names
                    logger.info("Attempting name-based merge...")
                    # Create a name column for matching
                    if 'last_name, first_name' in df.columns:
                        # Convert "Last, First" to "First Last" format to match FanGraphs
                        df['merge_name'] = df['last_name, first_name'].apply(
                            lambda x: ' '.join(reversed(x.split(', '))) if pd.notna(x) and ', ' in str(x) else str(x)
                        ).str.lower().str.strip()
                    elif 'Name' in df.columns:
                        df['merge_name'] = df['Name'].str.lower().str.strip()
                    
                    if 'Name' in fg_batting.columns:
                        fg_batting['merge_name'] = fg_batting['Name'].str.lower().str.strip()
                        
                        # Merge on the normalized name
                        df = pd.merge(df, fg_batting, on='merge_name', how='left', suffixes=('', '_fg'))
                        
                        # Check merge success
                        merged_cols = [c for c in df.columns if c.endswith('_fg')]
                        if merged_cols:
                            logger.info(f"Successfully merged {len(merged_cols)} columns from FanGraphs")
                            # Check if key metrics have data
                            for col in ['Barrel%', 'HardHit%', 'BB%', 'K%']:
                                if col in df.columns:
                                    non_null = df[col].notna().sum()
                                    logger.info(f"  {col}: {non_null}/{len(df)} non-null")
                        
                        df = df.drop(columns=['merge_name'], errors='ignore')
        except Exception as e:
            logger.warning(f"Could not fetch FanGraphs data: {e}")
        
        # Ensure we have all key columns (fill missing with NaN)
        key_columns = [
            'Name', 'player_id',
            # Quality of Contact
            'barrel_batted_rate', 'hard_hit_percent', 'avg_exit_velocity', 'sweet_spot_percent',
            # Expected Outcomes
            'xba', 'xslg', 'xwoba', 'xiso',
            # Plate Discipline
            'o_swing_percent', 'z_swing_percent', 'swing_percent', 'contact_percent',
            'z_contact_percent', 'o_contact_percent', 'zone_percent',
            # Batted Ball Profile
            'launch_angle', 'launch_speed', 'pull_percent', 'centered_percent', 'oppo_percent',
            # Run Value
            'woba', 'wrc_plus', 'war',
            # Traditional
            'ab', 'pa', 'h', 'bb', 'k', 'hr', 'sb', 'r', 'rbi'
        ]
        
        # Add missing columns
        for col in key_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        logger.info(f"Fetched {len(df)} batting records with {len(df.columns)} columns")
        
        # Save raw data
        output_path = DATA_RAW / f"comprehensive_batting_{year}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching comprehensive batting stats: {e}")
        return pd.DataFrame()


def fetch_comprehensive_pitching_stats(year: int = 2025) -> pd.DataFrame:
    """
    Fetch comprehensive pitching statistics including all advanced metrics.
    
    Args:
        year: Season year
        
    Returns:
        DataFrame with comprehensive pitching metrics
    """
    if pyb is None:
        logger.error("pybaseball not available")
        return pd.DataFrame()
    
    logger.info(f"Fetching comprehensive pitching stats for {year}...")
    
    try:
        # Try to get pitching stats - use batting_stats approach or date range
        # For now, use FanGraphs pitching stats which has advanced metrics
        df = pyb.pitching_stats(year, qual=50)
        
        if df is None or df.empty:
            logger.warning(f"No Statcast pitching data for {year}")
            return pd.DataFrame()
        
        # Also fetch FanGraphs data for WAR, FIP, SIERA, etc.
        try:
            fg_pitching = pyb.pitching_stats(year, qual=50)
            if fg_pitching is not None and not fg_pitching.empty:
                df = pd.merge(
                    df,
                    fg_pitching,
                    on=['Name', 'player_id'],
                    how='left',
                    suffixes=('', '_fg')
                )
        except Exception as e:
            logger.warning(f"Could not fetch FanGraphs data: {e}")
        
        # Ensure we have all key columns
        key_columns = [
            'Name', 'player_id',
            # Expected vs Actual
            'era', 'xera', 'xfip', 'siera', 'fip',
            # Contact Suppression
            'barrel_batted_rate', 'hard_hit_percent', 'avg_exit_velocity',
            # Pitch Movement (if available)
            'release_speed', 'spin_rate',
            # Whiff Ability
            'whiff_percent', 'csw_percent', 'k_percent', 'bb_percent', 'k_bb_percent',
            # Contextual Luck
            'babip', 'lob_percent', 'hr_fb_percent',
            # Traditional
            'ip', 'h', 'er', 'bb', 'so', 'hr', 'w', 'l', 'sv'
        ]
        
        # Add missing columns
        for col in key_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        logger.info(f"Fetched {len(df)} pitching records with {len(df.columns)} columns")
        
        # Save raw data
        output_path = DATA_RAW / f"comprehensive_pitching_{year}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching comprehensive pitching stats: {e}")
        return pd.DataFrame()


def fetch_salary_data(year: int = 2025) -> pd.DataFrame:
    """
    Attempt to fetch salary data from CSV file.
    
    Note: Salary data is not available in pybaseball. This function loads
    salary data from a manually created CSV file (see fetch_salaries.py).
    
    Args:
        year: Season year
        
    Returns:
        DataFrame with salary information (columns: 'name', 'salary_2025')
    """
    logger.info("Attempting to fetch salary data...")
    
    try:
        from fetch_salary_data import get_salary_data
        
        salary_path = DATA_PROCESSED / f"salaries_{year}.csv"
        salary_df = get_salary_data(year=year, source='csv', csv_path=salary_path)
        
        if salary_df.empty:
            logger.info("No salary data found. Run 'python fetch_salaries.py --create-template' to create a template.")
        
        return salary_df
    except ImportError:
        logger.warning("Could not import salary fetching module")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not fetch salary data: {e}")
        return pd.DataFrame()


def filter_pitchers_by_ip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter pitchers based on innings pitched requirements:
    - Starters: At least 50 innings pitched
    - Relievers: At least 20 innings pitched
    
    Args:
        df: DataFrame with pitching data
        
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    
    # Try to find IP column (could be 'IP', 'ip', 'IP_', etc.)
    ip_col = None
    for col in ['IP', 'ip', 'IP_', 'Innings Pitched', 'innings_pitched']:
        if col in df.columns:
            ip_col = col
            break
    
    if ip_col is None:
        logger.warning("Could not find IP column for pitcher filtering")
        return df
    
    # Try to identify starters vs relievers
    # Starters typically have GS (games started) > 0 or high IP/G ratio
    gs_col = None
    for col in ['GS', 'gs', 'Games Started', 'games_started']:
        if col in df.columns:
            gs_col = col
            break
    
    # Identify starters (GS > 0 or IP/G > 1.5) and relievers (GS = 0 or IP/G <= 1.5)
    if gs_col and gs_col in df.columns:
        starters = df[df[gs_col] > 0].copy()
        relievers = df[df[gs_col] == 0].copy()
    else:
        # Fallback: Use IP/G ratio (starters typically > 1.5 IP per game)
        g_col = None
        for col in ['G', 'g', 'Games', 'games']:
            if col in df.columns:
                g_col = col
                break
        
        if g_col and g_col in df.columns:
            df['ip_per_game'] = df[ip_col] / (df[g_col] + 1e-10)
            starters = df[df['ip_per_game'] > 1.5].copy()
            relievers = df[df['ip_per_game'] <= 1.5].copy()
        else:
            # If we can't identify, assume all are starters and use 50 IP minimum
            logger.warning("Could not identify starters vs relievers, using 50 IP minimum for all")
            return df[df[ip_col] >= 50]
    
    # Filter starters: >= 50 IP
    starters_filtered = starters[starters[ip_col] >= 50]
    
    # Filter relievers: >= 20 IP
    relievers_filtered = relievers[relievers[ip_col] >= 20]
    
    # Combine
    filtered = pd.concat([starters_filtered, relievers_filtered], ignore_index=True)
    
    logger.info(f"Filtered pitchers: {len(starters_filtered)} starters (>=50 IP), {len(relievers_filtered)} relievers (>=20 IP)")
    
    return filtered


def calculate_batted_ball_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate batted ball profile percentages (GB%, FB%, LD%).
    
    Args:
        df: DataFrame with launch angle data
        
    Returns:
        DataFrame with added batted ball percentages
    """
    df = df.copy()
    
    if 'launch_angle' not in df.columns:
        logger.warning("No launch_angle data available for batted ball calculations")
        return df
    
    # Define launch angle ranges (approximate)
    # Ground balls: < 10 degrees
    # Line drives: 10-25 degrees
    # Fly balls: > 25 degrees
    
    total_batted_balls = df.get('ab', 0).fillna(0) - df.get('k', 0).fillna(0)
    
    # This is simplified - in practice you'd need actual batted ball event data
    # For now, we'll use launch angle if available
    if 'launch_angle' in df.columns:
        df['gb_percent'] = (df['launch_angle'] < 10).astype(int) * 100
        df['ld_percent'] = ((df['launch_angle'] >= 10) & (df['launch_angle'] <= 25)).astype(int) * 100
        df['fb_percent'] = (df['launch_angle'] > 25).astype(int) * 100
    
    return df


def combine_all_data(year: int = 2025, min_pa: int = 200) -> pd.DataFrame:
    """
    Combine all data sources into a unified dataset.
    
    Args:
        year: Season year
        min_pa: Minimum plate appearances for hitters (200 for full season,
                lower for live-season mode).
        
    Returns:
        Combined DataFrame with all metrics
    """
    logger.info("Combining all data sources...")
    
    # Fetch all data
    batting = fetch_comprehensive_batting_stats(year, min_pa=min_pa)
    pitching = fetch_comprehensive_pitching_stats(year)
    salary = fetch_salary_data(year)
    
    # Process batting data
    if not batting.empty:
        batting = calculate_batted_ball_percentages(batting)
        batting['position_type'] = 'Hitter'
        # Filter: Minimum 200 plate appearances
        # Use Statcast PA if available (more inclusive), otherwise use FanGraphs PA
        pa_col = None
        if 'pa' in batting.columns:
            pa_col = 'pa'
        elif 'PA' in batting.columns:
            pa_col = 'PA'
        elif 'PA_fg' in batting.columns:
            pa_col = 'PA_fg'
        
        if pa_col:
            before_count = len(batting)
            batting = batting[batting[pa_col] >= min_pa]
            logger.info(f"Filtered hitters: {before_count} -> {len(batting)} with >= {min_pa} PA (using {pa_col})")
        else:
            logger.warning("No PA column found - cannot filter by plate appearances")
    
    # Process pitching data
    if not pitching.empty:
        pitching['position_type'] = 'Pitcher'
        # Filter pitchers: 50 IP for starters, 20 IP for relievers
        pitching = filter_pitchers_by_ip(pitching)
    
    # Combine
    if not batting.empty and not pitching.empty:
        combined = pd.concat([batting, pitching], ignore_index=True)
    elif not batting.empty:
        combined = batting
    elif not pitching.empty:
        combined = pitching
    else:
        combined = pd.DataFrame()
    
    # Merge salary data if available
    if not salary.empty and not combined.empty:
        combined = pd.merge(
            combined,
            salary,
            on=['Name', 'player_id'],
            how='left'
        )
    
    # Save combined dataset
    if not combined.empty:
        output_path = DATA_PROCESSED / f"comprehensive_stats_{year}.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved combined dataset to {output_path} ({len(combined)} records)")
    
    return combined


def main():
    """Main function to run the enhanced data pipeline."""
    year = 2025
    
    logger.info(f"Starting enhanced data pipeline for {year} MLB season...")
    
    # Ensure directories exist
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Fetch and combine all data
    combined = combine_all_data(year)
    
    logger.info("Enhanced data pipeline completed!")
    return combined


if __name__ == "__main__":
    main()

