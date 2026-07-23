"""
Undervaluation Score (UVS) calculation module.

This module implements the core UVS formula used to identify undervalued MLB hitters.
The UVS combines 6 weighted indexes to create a comprehensive score that identifies
players whose expected performance, contact quality, and efficiency exceed their
actual results and salary.

Formula:
UVS = 0.25(EPI) + 0.20(CQI) + 0.15(PDI) + 0.15(RPI) + 0.10(SE) + 0.10(LA)

Component Indexes:
- EPI (Expected Performance Index) - 25%: Mean of z-scores for xwOBA, xSLG, xBA, xISO
- CQI (Contact Quality Index) - 20%: Mean of z-scores for Barrel%, HardHit%, Exit Velo, Sweet Spot%
- PDI (Plate Discipline Index) - 15%: z(BB%) - z(K%) - z(O-Swing%) + z(Z-Contact%) + z(Contact%)
- RPI (Run Production Index) - 15%: Mean of z-scores for wRC+, wOBA, OPS, ISO, R, RBI
- SE (Salary Efficiency) - 10%: z(WAR per $1M)
- LA (Luck Adjustment) - 10%: Mean of z-scores for (xwOBA-wOBA), (xBA-BA), (xSLG-SLG)

All components use z-scores: z = (x - x̄) / σ

Higher UVS = More undervalued (better underlying performance relative to results/salary)
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_z_score(series: pd.Series) -> pd.Series:
    """
    Calculate z-scores (standardized values) for a series.
    
    z = (x - x̄) / σ
    """
    if series.std() == 0 or series.std() is None or pd.isna(series.std()):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / series.std()


def get_column(df: pd.DataFrame, possible_names: list, default: float = 0.0) -> pd.Series:
    """Helper to get column with fallbacks."""
    for col_name in possible_names:
        if col_name in df.columns and df[col_name].notna().sum() > 0:
            return df[col_name].fillna(default)
    return pd.Series(default, index=df.index)


def calculate_epi(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Expected Performance Index (EPI).
    
    EPI = mean(z(xwOBA), z(xSLG), z(xBA), z(xISO))
    """
    xwoba = get_column(df, ['est_woba', 'xwoba', 'xwOBA'])
    xslg = get_column(df, ['est_slg', 'xslg', 'xSLG'])
    xba = get_column(df, ['est_ba', 'xba', 'xBA'])
    xiso = get_column(df, ['xiso', 'xISO'])
    
    # Calculate xISO if not available (xISO = xSLG - xBA)
    if (xiso == 0).all() or xiso.isna().all():
        xiso = xslg - xba
    
    # Calculate z-scores and take mean
    z_xwoba = calculate_z_score(xwoba)
    z_xslg = calculate_z_score(xslg)
    z_xba = calculate_z_score(xba)
    z_xiso = calculate_z_score(xiso)
    
    epi = (z_xwoba + z_xslg + z_xba + z_xiso) / 4.0
    
    return epi


def calculate_cqi(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Contact Quality Index (CQI).
    
    CQI = mean(z(Barrel%), z(HardHit%), z(Exit Velo), z(Sweet Spot%))
    """
    barrel = get_column(df, ['Barrel%', 'barrel_batted_rate'])
    hardhit = get_column(df, ['HardHit%', 'hard_hit_percent'])
    exit_velo = get_column(df, ['avg_exit_velocity', 'exit_velocity'])
    sweet_spot = get_column(df, ['sweet_spot_percent', 'sweet_spot'])
    
    # Normalize percentages if needed (convert 0-100 to 0-1)
    if barrel.max() > 1:
        barrel = barrel / 100
    if hardhit.max() > 1:
        hardhit = hardhit / 100
    if sweet_spot.max() > 1:
        sweet_spot = sweet_spot / 100
    
    # Calculate z-scores and take mean
    z_barrel = calculate_z_score(barrel)
    z_hardhit = calculate_z_score(hardhit)
    z_exit_velo = calculate_z_score(exit_velo)
    z_sweet_spot = calculate_z_score(sweet_spot)
    
    cqi = (z_barrel + z_hardhit + z_exit_velo + z_sweet_spot) / 4.0
    
    return cqi


def calculate_pdi(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Plate Discipline Index (PDI).
    
    PDI = z(BB%) – z(K%) – z(O-Swing%) + z(Z-Contact%) + z(Contact%)
    """
    bb_pct = get_column(df, ['BB%', 'bb_percent'])
    k_pct = get_column(df, ['K%', 'k_percent'])
    o_swing = get_column(df, ['O-Swing%', 'o_swing_percent', 'Chase%'])
    z_contact = get_column(df, ['Z-Contact%', 'z_contact_percent'])
    contact_pct = get_column(df, ['Contact%', 'contact_percent'])
    
    # Normalize percentages if needed
    if bb_pct.max() > 1:
        bb_pct = bb_pct / 100
    if k_pct.max() > 1:
        k_pct = k_pct / 100
    if o_swing.max() > 1:
        o_swing = o_swing / 100
    if z_contact.max() > 1:
        z_contact = z_contact / 100
    if contact_pct.max() > 1:
        contact_pct = contact_pct / 100
    
    # Calculate z-scores
    z_bb = calculate_z_score(bb_pct)
    z_k = calculate_z_score(k_pct)
    z_o_swing = calculate_z_score(o_swing)
    z_z_contact = calculate_z_score(z_contact)
    z_contact_pct = calculate_z_score(contact_pct)
    
    pdi = z_bb - z_k - z_o_swing + z_z_contact + z_contact_pct
    
    return pdi


def calculate_rpi(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Run Production Index (RPI).
    
    RPI = mean(z(wRC+), z(wOBA), z(OPS), z(ISO), z(R), z(RBI))
    """
    wrc_plus = get_column(df, ['wRC+', 'wrc_plus'])
    woba = get_column(df, ['woba', 'wOBA'])
    ops = get_column(df, ['OPS', 'ops'])
    iso = get_column(df, ['ISO', 'iso'])
    runs = get_column(df, ['R', 'r'])
    rbi = get_column(df, ['RBI', 'rbi'])
    
    # Calculate z-scores and take mean
    z_wrc_plus = calculate_z_score(wrc_plus)
    z_woba = calculate_z_score(woba)
    z_ops = calculate_z_score(ops)
    z_iso = calculate_z_score(iso)
    z_runs = calculate_z_score(runs)
    z_rbi = calculate_z_score(rbi)
    
    rpi = (z_wrc_plus + z_woba + z_ops + z_iso + z_runs + z_rbi) / 6.0
    
    return rpi


def calculate_se(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Salary Efficiency (SE).
    
    SE = z(WAR per $1M)
    """
    # Try to get WAR per $1M
    war_per_salary = get_column(df, ['war_per_salary'])
    
    # If not available, calculate it
    if (war_per_salary == 0).all() or war_per_salary.isna().all():
        war = get_column(df, ['WAR', 'war'])
        salary = get_column(df, ['salary_2026', 'salary_2025', 'salary_2025_x', 'salary_2025_y'], default=np.inf)
        war_per_salary = war / (salary + 0.1)  # Add small value to avoid division by zero
    
    # Calculate z-score
    se = calculate_z_score(war_per_salary)
    
    return se


def calculate_la(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Luck Adjustment (LA).
    
    LA = mean(z(xwOBA – wOBA), z(xBA – BA), z(xSLG – SLG))
    
    Positive values mean unlucky (expected > actual), which is good for undervaluation.
    """
    # Calculate differences
    xwoba = get_column(df, ['est_woba', 'xwoba', 'xwOBA'])
    woba = get_column(df, ['woba', 'wOBA'])
    xwoba_diff = xwoba - woba
    
    xba = get_column(df, ['est_ba', 'xba', 'xBA'])
    ba = get_column(df, ['BA', 'ba'])
    xba_diff = xba - ba
    
    xslg = get_column(df, ['est_slg', 'xslg', 'xSLG'])
    slg = get_column(df, ['SLG', 'slg'])
    xslg_diff = xslg - slg
    
    # Try to use pre-calculated differences if available
    if 'est_woba_minus_woba_diff' in df.columns and df['est_woba_minus_woba_diff'].notna().sum() > 0:
        xwoba_diff = df['est_woba_minus_woba_diff'].fillna(xwoba_diff)
    if 'est_ba_minus_ba_diff' in df.columns and df['est_ba_minus_ba_diff'].notna().sum() > 0:
        xba_diff = df['est_ba_minus_ba_diff'].fillna(xba_diff)
    if 'est_slg_minus_slg_diff' in df.columns and df['est_slg_minus_slg_diff'].notna().sum() > 0:
        xslg_diff = df['est_slg_minus_slg_diff'].fillna(xslg_diff)
    
    # Calculate z-scores and take mean
    z_xwoba_diff = calculate_z_score(xwoba_diff)
    z_xba_diff = calculate_z_score(xba_diff)
    z_xslg_diff = calculate_z_score(xslg_diff)
    
    la = (z_xwoba_diff + z_xba_diff + z_xslg_diff) / 3.0
    
    return la


def calculate_uvs(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Undervaluation Score (UVS).
    
    UVS = 0.25(EPI) + 0.20(CQI) + 0.15(PDI) + 0.15(RPI) + 0.10(SE) + 0.10(LA)
    
    This score rewards:
    - Great expected performance (xwOBA, xSLG, xBA)
    - Strong plate discipline and contact quality
    - High WAR per $1M
    - Positive luck differential (xwOBA > wOBA means unlucky, should improve)
    - Strong run creation (wRC+)
    """
    # Calculate all component indices
    epi = calculate_epi(df)
    cqi = calculate_cqi(df)
    pdi = calculate_pdi(df)
    rpi = calculate_rpi(df)
    se = calculate_se(df)
    la = calculate_la(df)
    
    # Combine with weights
    uvs = (
        0.25 * epi +
        0.20 * cqi +
        0.15 * pdi +
        0.15 * rpi +
        0.10 * se +
        0.10 * la
    )
    
    return uvs


def calculate_all_uvs_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all UVS-related metrics and add them to the dataframe.
    
    Returns dataframe with added columns:
    - epi (Expected Performance Index)
    - cqi (Contact Quality Index)
    - pdi (Plate Discipline Index)
    - rpi (Run Production Index)
    - se (Salary Efficiency)
    - la (Luck Adjustment)
    - uvs (Undervaluation Score)
    """
    df = df.copy()
    
    df['epi'] = calculate_epi(df)
    df['cqi'] = calculate_cqi(df)
    df['pdi'] = calculate_pdi(df)
    df['rpi'] = calculate_rpi(df)
    df['se'] = calculate_se(df)
    df['la'] = calculate_la(df)
    df['uvs'] = calculate_uvs(df)
    
    return df

