# White Paper: 2025 Undervalued MLB Hitters
## A Statcast-Driven Composite Valuation Framework for Identifying Market Inefficiencies at the Plate

**Author:** Shirin Alapati  
**Project:** 2025 Undervalued MLB Hitters  
**Live Dashboard:** [salapati.pythonanywhere.com](https://salapati.pythonanywhere.com/)  
**Date:** February 2026

---

## Executive Summary

This paper presents the methodology, data infrastructure, and analytical findings behind a multi-dimensional hitter valuation model built for the 2025 MLB season. The central output is the **Undervaluation Score (UVS)** — a six-component composite metric that identifies hitters whose underlying performance quality, contact profile, and plate discipline outpace their observed results and salary cost.

The model evaluates **350 qualified hitters** (≥200 plate appearances) using 30+ advanced statistics sourced from Baseball Savant (Statcast) and FanGraphs. All components are standardized via z-scores and weighted to reflect their predictive relevance to sustainable offensive production.

Key findings demonstrate that expected metrics (xwOBA, xBA, xSLG) and contact quality indicators (Barrel%, Exit Velocity) are systematically stronger predictors of future performance than traditional counting stats. Players whose expected metrics substantially exceed their realized results represent acquisition opportunities often mispriced by the market.

---

## 1. Motivation

### The Problem with Traditional Hitter Evaluation

Conventional front office evaluation historically over-indexed on surface statistics: batting average, home runs, RBI, and OPS. These metrics share a fundamental flaw — they capture what happened, not the quality of contact that drove it.

A hitter who posts a .240 BA while barreling 15% of his balls in play and averaging 95 mph exit velocity is not the same player as one posting .240 BA via weak contact finding holes. BABIP variance, park factors, lineup protection, and sequencing luck routinely create a gap of 30-50 points between a player's expected and observed production across a single season.

This project operationalizes that gap. The UVS model quantifies luck, isolates contact quality from results, and incorporates salary efficiency — producing a single, interpretable ranking of which hitters the market is most likely to be mispricing downward.

### Relevance to a Modern Front Office

Front offices increasingly use expected metrics to:
- Project second-half breakouts for waiver wire and trade deadline decisions
- Set offseason free agent valuations ahead of the market
- Identify pre-arbitration players whose salary cost will remain well below production
- Build internal rankings to challenge the consensus market price on controllable assets

This model is designed to serve those exact functions.

---

## 2. Data Sources and Retrieval Pipeline

### 2.1 Primary Data Sources

| Source | Data Retrieved | Tool/Method |
|--------|---------------|-------------|
| Baseball Savant (Statcast) | xwOBA, xBA, xSLG, Barrel%, HardHit%, Exit Velo, Sweet Spot% | `pybaseball.statcast_batter_expected_stats()` |
| FanGraphs | wRC+, wOBA, OPS, ISO, BB%, K%, Z-Contact%, O-Swing%, GB%, LD%, WAR | `pybaseball.batting_stats()` |
| Spotrac (web-scraped) | 2025 salary data | `requests` + `BeautifulSoup` |

### 2.2 Pipeline Architecture

The data pipeline is implemented in Python using a modular structure under `src/data_pipeline/`:

1. **`fetch_comprehensive.py`** — Orchestrates the full data collection pass, calling Statcast and FanGraphs endpoints for all hitters with ≥200 PA
2. **`fetch_statcast_pitch_data.py`** — Retrieves granular Statcast batted-ball metrics (barrel rate, exit velocity, sweet spot%)
3. **`fetch_salary_data.py`** — Scrapes and normalizes salary data from Spotrac, used to compute WAR/salary efficiency
4. **`fetch_advanced_metrics.py`** — Pulls FanGraphs plate discipline and batted ball profile data

All raw data is persisted as CSV to `data/raw/` before being merged and cleaned into `data/processed/comprehensive_stats_2025.csv`, which serves as the single source of truth for the UVS calculation and the interactive dashboard.

### 2.3 Data Quality

- **350 qualified hitters** retained after applying the ≥200 PA threshold
- Multi-source merges performed on player name and player ID to minimize join errors
- Missing salary entries defaulted to league minimum ($740,000) with a flag, preserving players from the salary efficiency calculation without fabricating data
- Percentage columns from different sources normalized to a consistent 0–100 scale before z-score computation

---

## 3. Model Architecture: Undervaluation Score (UVS)

### 3.1 Formula

```
UVS = 0.25(EPI) + 0.20(CQI) + 0.15(PDI) + 0.15(RPI) + 0.10(SE) + 0.10(LA)
```

Where each component is a z-score-standardized index (mean = 0, standard deviation = 1 across all 350 hitters). A higher UVS indicates a player whose underlying production quality is meaningfully underrepresented by their current statistics and/or salary.

### 3.2 Component Index Definitions

---

#### EPI — Expected Performance Index (Weight: 25%)

```
EPI = mean( z(xwOBA), z(xSLG), z(xBA), z(xISO) )
```

**What it captures:** The ceiling of a hitter's offensive output based on the quality of contact made, stripped of BABIP luck and sequencing noise.

**Why 25%:** Expected metrics are the most predictive inputs in the model. xwOBA in particular has among the highest year-over-year correlation of any offensive statistic (~0.70), compared to ~0.50 for observed wOBA. Weighting EPI highest ensures the model rewards genuine contact quality rather than results that may not repeat.

**Key inputs:**
- **xwOBA** (Baseball Savant): Expected weighted on-base average based on exit velocity, launch angle, and sprint speed
- **xSLG**: Expected slugging percentage derived from batted ball quality
- **xBA**: Expected batting average; filters out defensive positioning and BABIP variance
- **xISO**: Derived as xSLG − xBA; measures raw expected power output

---

#### CQI — Contact Quality Index (Weight: 20%)

```
CQI = mean( z(Barrel%), z(HardHit%), z(Exit Velo), z(Sweet Spot%) )
```

**What it captures:** How hard and optimally a hitter is consistently making contact, independent of whether those balls fell for hits.

**Why 20%:** Contact quality is the mechanical input that determines xwOBA and future expected production. Two players with identical xwOBA driven by different contact profiles carry different risk profiles. A high Barrel% hitter has consistently positive outcomes when healthy; a player relying on "soft contact BABIP" does not.

**Key inputs:**
- **Barrel%**: Batted balls matching the optimal exit velocity × launch angle combination (90+ mph, 26–30° angle typically), associated with a ~1.000 xBA and ~2.800 xSLG
- **HardHit%**: Percentage of batted balls at 95+ mph exit velocity
- **Exit Velocity (avg)**: Mean exit velocity across all batted ball events; a strong indicator of physical bat speed and contact efficiency
- **Sweet Spot%**: Launch angle between 8° and 32°; approximates the range associated with line drives and hard fly balls

---

#### PDI — Plate Discipline Index (Weight: 15%)

```
PDI = z(BB%) − z(K%) − z(O-Swing%) + z(Z-Contact%) + z(Contact%)
```

**What it captures:** The cognitive and mechanical skill of a hitter at the plate — their ability to draw walks, avoid strikeouts, lay off pitches outside the zone, and make contact when swinging.

**Why 15%:** Plate discipline is a highly stable, skill-driven metric. Walk rates and strikeout rates have high year-over-year correlations (~0.75–0.85). A disciplined hitter generates more favorable counts, sees more fastballs, and sustains production even through slumps. The inclusion of O-Swing% penalizes chase-heavy approaches even if current results appear acceptable.

**Key inputs:**
- **BB%**: Walk rate; rewarded as an indicator of pitch recognition and zone awareness
- **K%**: Strikeout rate; penalized as a drag on both run production and lineup efficiency
- **O-Swing%** (Chase%): Swing rate on pitches outside the strike zone; high values penalized
- **Z-Contact%**: Contact rate on pitches inside the strike zone; high values rewarded
- **Contact%**: Overall swing-to-contact rate; rewards bat control

---

#### RPI — Run Production Index (Weight: 15%)

```
RPI = mean( z(wRC+), z(wOBA), z(OPS), z(ISO), z(R), z(RBI) )
```

**What it captures:** Observed offensive production in the current season, measuring the player's actual run-creation contribution to their team.

**Why 15%:** While observed production carries BABIP noise, it remains a necessary anchor. A hitter can have elite expected metrics but if their actual production is severely depressed, it may indicate mechanical issues or injury the model cannot detect via Statcast alone. The RPI keeps the model honest by penalizing a total disconnect between expected and observed production.

**Key inputs:**
- **wRC+**: Park and league adjusted runs created; 100 = league average, values above indicate above-average production
- **wOBA**: Observed weighted on-base average using standard run weights (BB: 0.69, 1B: 0.89, 2B: 1.27, 3B: 1.62, HR: 2.10)
- **OPS**: On-base plus slugging; widely understood reference point
- **ISO**: Isolated slugging (SLG − BA); raw power measure
- **R / RBI**: Volume counting stats included to capture lineup context and clutch production

---

#### SE — Salary Efficiency (Weight: 10%)

```
SE = z( WAR / Salary_in_$M )
```

**What it captures:** How much Wins Above Replacement a team is receiving per million dollars of salary committed to this player.

**Why 10%:** The front office application of any undervaluation model is ultimately a resource allocation problem. A player generating 3.0 WAR on a $3M salary is a different acquisition target than one generating 3.0 WAR on a $25M salary. The SE component directly surfaces this market inefficiency.

**Practical note:** Pre-arbitration players earning near the league minimum ($740K) while producing 2.0+ WAR will score exceptionally on SE. This makes the model particularly relevant for identifying trade sell-high candidates and extension negotiations.

---

#### LA — Luck Adjustment (Weight: 10%)

```
LA = mean( z(xwOBA − wOBA), z(xBA − BA), z(xSLG − SLG) )
```

**What it captures:** The degree to which a player's observed statistics have underperformed their expected metrics — i.e., how "unlucky" they have been.

**Why 10%:** Players with large positive LA (xwOBA >> wOBA, xBA >> BA) have produced quality contact that has not yet resulted in commensurate outcomes. This gap typically closes over larger sample sizes and is often more predictive of future performance than the observed statistics themselves. Identifying these players before the market corrects for the gap is the core "alpha" of the model.

**Example:** A player posting a .240 BA with .330 xBA and a .310 wOBA with .380 xwOBA is almost certainly underperforming due to BABIP luck, defensive positioning, or a hot-spot-heavy infield shift. The LA component flags them as a high-priority target.

---

### 3.3 Normalization Methodology

All component indices use **z-score standardization**:

```
z = (x − x̄) / σ
```

Where x̄ is the mean and σ is the standard deviation across all 350 qualified hitters. This approach:

- Centers the distribution at zero, making positive scores immediately interpretable as above-average
- Is scale-invariant, allowing metrics like Barrel% (0–20%) and Exit Velocity (70–115 mph) to contribute equally without artificial inflation from unit differences
- Handles skewed distributions more robustly than min-max scaling when outliers exist (e.g., elite exit velocity seasons)

Missing values are filled with the population mean before z-scoring, which equates to a z-score of zero — a conservative assumption that neither rewards nor penalizes players with missing data.

---

## 4. Supplementary Composite Metrics

Beyond UVS, the model calculates five additional metrics offering alternative lenses on player value:

| Metric | Formula Summary | Use Case |
|--------|----------------|----------|
| **TOVA+** (True Offensive Value Added Plus) | 10-component z-score composite (xwOBA 15%, xSLG 10%, xISO 10%, Contact Quality 10%, Plate Discipline 10%, Luck Index 10%, Run Value 10%, Balance Score 10%, wRC+ 5%, xBA 10%) | Holistic offensive profile; analogous to a pitcher's FIP− |
| **UPI** (Ultimate Performance Index) | 8-component composite emphasizing expected vs. actual gaps (xwOBA−wOBA diff 25%, contact quality 15%, plate discipline 15%) | Regression candidate identification; short-term performance forecasting |
| **OPS 2.0** | xwOBA (40%) + xSLG (20%) + Barrel% (15%) + BB% (10%) − K% (10%) + Sweet Spot% (5%) | Machine-readable replacement for traditional OPS; scout-friendly one-number summary |
| **BOV** (Best Overall Value) | Expected Production (40%) + Contact Quality (25%) + Run Value (15%) + Plate Discipline (10%) + Balance Score (5%) + Luck Index (5%) | Balanced value metric for general roster construction |
| **TOVA$** | TOVA+ / Salary in $M | Cost-efficiency metric; ideal for identifying pre-arb or team-controlled value |

These supplementary metrics are accessible on the "All Players Stats" tab of the interactive dashboard and allow evaluators to cross-validate UVS findings from multiple analytical angles.

---

## 5. Front Office Applications

### 5.1 Trade Deadline and Waiver Wire Targeting

The model is designed to surface players whose second-half production is likely to exceed their first-half results. High LA players (large positive xwOBA − wOBA gap) entering the trade deadline at a depressed trade value represent buy-low opportunities. Teams can cross-reference UVS rankings with available salary data to identify players worth acquiring ahead of a BABIP correction.

**Decision filter example:**
- UVS rank ≤ 30
- LA component in top quartile (xwOBA − wOBA > +0.020)
- Under $8M annual salary or pre-arbitration
- wRC+ between 90–115 (depressed but not broken)

This profile targets "hidden value" players available at a discount, typical of contending teams eager to shed salary for prospects.

### 5.2 Free Agent Valuation and Contract Negotiations

The market typically prices free agents on trailing two or three-year averages weighted toward recent performance. A player who had an unlucky final contract year — suppressed BA on high exit velocity, low BABIP despite strong Barrel% — will likely be underpriced relative to their true talent.

The UVS model provides a quantitative basis for:
- Projecting year-one expected production from xwOBA, not observed wOBA
- Calculating WAR/$ at fair market value (~$8M/WAR) vs. likely contract cost
- Identifying the "right price" range using the SE component

### 5.3 Lineup Construction and Platoon Decisions

Contact quality and plate discipline indices (CQI, PDI) are independent of lineup spot and opponent handedness effects. Teams can use these components to:
- Identify high-discipline, low-chase hitters for leadoff roles
- Target high-Barrel%, high-xISO bats for protection behind marquee hitters
- Quantify the true cost/benefit of platoon splits when considering full-season production vs. platoon-leveraged roles

### 5.4 Extension Negotiations and Arbitration

Pre-arbitration players generating high SE scores are at risk of contract undervaluation in traditional arbitration, which is driven by service time and observed stats. Teams can use UVS data to:
- Proactively offer extensions to high-UVS, low-salary players before arbitration eligibility
- Prepare arbitration cases that incorporate xwOBA over observed wOBA for players with favorable luck differentials
- Set internal valuations grounded in expected performance, not trailing averages

### 5.5 Draft and International Market Crossover

While the model is built on MLB season data, the underlying logic maps directly to amateur scouting:
- Barrel% and exit velocity are increasingly tracked in the draft and international signing market via Trackman and Hawk-Eye
- Players with strong contact quality profiles but modest counting stats in lower leagues may reflect the same market inefficiency the model finds in the majors

---

## 6. Model Limitations

### 6.1 Sample Size Sensitivity

The 200 PA threshold provides a minimum sample but is not sufficient for stable z-score estimates of all components. Plate discipline metrics (BB%, K%, O-Swing%) stabilize around 200 PA; batted ball profile metrics (Barrel%, LD%) require 400+ PA for reliable estimates. Players near the PA cutoff should be interpreted with caution.

### 6.2 Positional and Defensive Context

The model is offense-only. A catcher generating a 90th percentile UVS score has different roster implications than a first baseman at the same rank. Evaluators should overlay UVS rankings with defensive WAR components and positional scarcity before making roster decisions.

### 6.3 BABIP Regression Assumptions

The LA component assumes that all positive xwOBA − wOBA gaps will close in the player's favor. In reality, some players consistently post below-expected BABIP due to below-average speed, extreme pull-heavy batted ball profiles, or defensive alignment. A positive luck adjustment does not guarantee future production improvement without corroborating sprint speed and batted ball direction data.

### 6.4 Salary Data Completeness

Players on minor league contracts, split contracts, or international deals may have incomplete or unavailable salary entries. Missing salary defaults to league minimum, which may artificially elevate SE scores for players on non-standard contracts. Manual verification is recommended for players whose SE rank diverges sharply from their overall UVS rank.

### 6.5 Static Annual Model

The model is recalculated once on the full season dataset. It does not update in real-time or incorporate rolling window analysis. Mid-season applications (trade deadline) would benefit from a rolling 60-game or 90-game recalculation to capture recent form more accurately.

### 6.6 Lack of Contextual Weighting

The model treats all 350 hitters as one population without adjusting for park factors, lineup environment, or injury history. A player generating elite xwOBA in a depressed offensive park (e.g., Petco Park, pre-2022 Oracle Park) is more impressive than the same metric in Coors Field. Park-factor normalization would improve the precision of cross-team comparisons.

---

## 7. Interpretation Guide

### Reading a UVS Score

UVS is a z-score-based composite. The reference points below apply across the full 350-player population:

| UVS Range | Interpretation |
|-----------|---------------|
| > +1.5 | Significantly undervalued; expected performance substantially exceeds results and/or salary |
| +0.5 to +1.5 | Moderately undervalued; above-average value relative to market price |
| −0.5 to +0.5 | Fairly priced; production and salary roughly in alignment |
| −0.5 to −1.5 | Potentially overvalued; observed results likely exceed repeatable skill level |
| < −1.5 | Significantly overvalued; regression risk or production tied to luck-driven outcomes |

### Red Flags to Cross-Check

A high UVS rank should prompt investigation of the following before acting:
- Sprint speed: Low sprint speed reduces BABIP even on hard contact (reduces LA reliability)
- Injury history: Early season injury may explain low observed stats, not luck
- Pull% / Oppo%: Extreme pull-heavy batters face infield shifts that suppress BABIP independent of exit velocity
- wRC+ below 80 despite high UVS: May indicate a broken mechanical approach not yet captured in Statcast averages

---

## 8. Conclusion

The 2025 Undervalued MLB Hitters model provides a systematic, data-driven framework for identifying offensive players whose underlying production quality is mispriced by the market. By anchoring on expected metrics derived from Statcast exit velocity and launch angle data, and incorporating salary efficiency as a first-class input, the model moves beyond the surface statistics that traditional evaluation overweights.

The six-component UVS formula is deliberately interpretable: each index has a clear analytic purpose, a defensible weight, and a direct application in front office decision-making. The supplementary metrics (TOVA+, UPI, OPS 2.0, BOV, TOVA$) provide additional lenses for analysts who want to cross-validate findings or isolate specific aspects of player value.

As Statcast data quality continues to improve and expected metrics expand in scope, models of this type will become increasingly central to how teams price free agents, evaluate trade targets, and manage roster construction. This project demonstrates the analytical foundation required to build such a system and the front office literacy to apply it in practice.

---

## Appendix A: Full UVS Formula Reference

```
UVS = 0.25(EPI) + 0.20(CQI) + 0.15(PDI) + 0.15(RPI) + 0.10(SE) + 0.10(LA)

EPI = mean( z(xwOBA), z(xSLG), z(xBA), z(xISO) )
      where xISO = xSLG − xBA if not directly available

CQI = mean( z(Barrel%), z(HardHit%), z(Exit Velo), z(Sweet Spot%) )

PDI = z(BB%) − z(K%) − z(O-Swing%) + z(Z-Contact%) + z(Contact%)

RPI = mean( z(wRC+), z(wOBA), z(OPS), z(ISO), z(R), z(RBI) )

SE  = z( WAR / Salary_$M )

LA  = mean( z(xwOBA − wOBA), z(xBA − BA), z(xSLG − SLG) )

z(x) = (x − x̄) / σ   [calculated across all 350 qualified hitters]
```

---

## Appendix B: Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Retrieval | Python, pybaseball, requests, BeautifulSoup |
| Data Processing | pandas, NumPy |
| Statistical Modeling | Custom z-score composites (NumPy) |
| Dashboard | Plotly Dash, Dash Bootstrap Components |
| Deployment | PythonAnywhere (WSGI) |
| Version Control | Git, GitHub |

---

## Appendix C: Data Dictionary (Key Columns)

| Column | Source | Description |
|--------|--------|-------------|
| `est_woba` / `xwOBA` | Statcast | Expected weighted on-base average |
| `est_slg` / `xSLG` | Statcast | Expected slugging percentage |
| `est_ba` / `xBA` | Statcast | Expected batting average |
| `barrel_batted_rate` | Statcast | Barrel percentage |
| `hard_hit_percent` | Statcast | Hard hit percentage (95+ mph EV) |
| `avg_exit_velocity` | Statcast | Average exit velocity |
| `sweet_spot_percent` | Statcast | Sweet spot percentage (8°–32° LA) |
| `wRC+` | FanGraphs | Park/league adjusted runs created |
| `woba` | FanGraphs | Observed wOBA |
| `BB%` / `bb_percent` | FanGraphs | Walk percentage |
| `K%` / `k_percent` | FanGraphs | Strikeout percentage |
| `O-Swing%` | FanGraphs | Chase rate (swing% outside zone) |
| `Z-Contact%` | FanGraphs | Contact rate inside zone |
| `WAR` | FanGraphs | Wins Above Replacement |
| `salary_2025` | Spotrac | 2025 AAV in millions |
| `uvs` | Model | Undervaluation Score (composite) |
| `epi` | Model | Expected Performance Index |
| `cqi` | Model | Contact Quality Index |
| `pdi` | Model | Plate Discipline Index |
| `rpi` | Model | Run Production Index |
| `se` | Model | Salary Efficiency |
| `la` | Model | Luck Adjustment |
