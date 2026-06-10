"""
Interactive web dashboard for the Undervalued MLB Players project.

This Plotly Dash application provides a comprehensive interface for exploring
350 MLB hitters from the 2025 season (all with ≥200 PA) and their advanced statistics.

Dashboard Features:
- Three main tabs:
  1. About This Page: Project overview, methodology, and UVS formula explanation
  2. All Players Stats: Comprehensive table with 30+ statistics per player
  3. Undervalued Players: UVS formula breakdown and ranked player list

- Interactive features:
  - Sortable and filterable tables
  - CSV export functionality
  - Metrics glossary (expandable)
  - Responsive design with Bootstrap styling

The dashboard loads data directly from data/processed/comprehensive_stats_2025.csv
and displays all advanced metrics including expected performance, contact quality,
plate discipline, batted ball profile, run production, and value metrics.
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from numpy import integer as np_integer, floating as np_floating
import requests
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# API base URL
API_BASE_URL = "http://localhost:8000"

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "Undervalued MLB Players | 2025 Season"

# App layout - Simple table-focused design
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1(
                "Undervalued MLB Players Analysis",
                className="text-center mb-2",
                style={'color': '#1a1a1a', 'fontWeight': '700'}
            ),
            html.P(
                "2025 Regular Season | All hitters with ≥200 PA",
                className="text-center text-muted mb-4"
            ),
            # Metrics Glossary
            dbc.Collapse([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("📚 Metrics Glossary", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.H6("Contact Quality & Power", className="mt-3 mb-2"),
                                html.P([
                                    html.Strong("Barrel %"), " – % of batted balls hit with ideal exit velocity + launch angle (Statcast \"barrels\")."
                                ]),
                                html.P([
                                    html.Strong("Hard Hit %"), " – % of batted balls hit ≥ 95 mph Exit Velocity."
                                ]),
                                html.P([
                                    html.Strong("Exit Velocity"), " – Average speed (mph) of all batted balls."
                                ]),
                                html.P([
                                    html.Strong("Sweet Spot %"), " – % of batted balls with launch angle 8–32°."
                                ]),
                                html.P([
                                    html.Strong("xHR"), " – Expected home runs based on launch angle + Exit Velocity of each batted ball."
                                ]),
                                html.H6("Expected Performance (Statcast)", className="mt-4 mb-2"),
                                html.P([
                                    html.Strong("wOBA"), " – Weighted On-Base Avg = (0.69×BB + 0.89×1B + 1.27×2B + 1.62×3B + 2.10×HR) / PA."
                                ]),
                                html.P([
                                    html.Strong("xwOBA"), " – Expected wOBA using quality of contact and Ks/BBs, not actual outcomes."
                                ]),
                                html.P([
                                    html.Strong("xBA"), " – Expected batting average based on exit velocity + launch angle."
                                ]),
                                html.P([
                                    html.Strong("xSLG"), " – Expected slugging % from contact quality."
                                ]),
                                html.P([
                                    html.Strong("xISO"), " – xSLG − xBA; expected isolated power."
                                ]),
                                html.H6("Plate Discipline", className="mt-4 mb-2"),
                                html.P([
                                    html.Strong("BB %"), " – Walks / Plate Appearances."
                                ]),
                                html.P([
                                    html.Strong("K %"), " – Strikeouts / Plate Appearances."
                                ]),
                                html.P([
                                    html.Strong("Chase % (O-Swing %)"), " – % of swings at pitches outside strike zone."
                                ]),
                                html.P([
                                    html.Strong("Z-Contact %"), " – % of swings on in-zone pitches that make contact."
                                ]),
                                html.P([
                                    html.Strong("Contact %"), " – % of all swings that make contact."
                                ]),
                            ], md=6),
                            dbc.Col([
                                html.H6("Batted-Ball Profile", className="mt-3 mb-2"),
                                html.P([
                                    html.Strong("GB %"), " – % of batted balls that are grounders."
                                ]),
                                html.P([
                                    html.Strong("FB %"), " – % that are fly balls."
                                ]),
                                html.P([
                                    html.Strong("LD %"), " – % that are line drives."
                                ]),
                                html.P([
                                    html.Strong("Pull %"), " – % of balls hit to pull side."
                                ]),
                                html.P([
                                    html.Strong("Oppo %"), " – % hit to opposite field."
                                ]),
                                html.H6("Run Production / Value", className="mt-4 mb-2"),
                                html.P([
                                    html.Strong("wRC+"), " – Weighted Runs Created Plus; 100 = league avg (offense adjusted for park/league)."
                                ]),
                                html.P([
                                    html.Strong("WAR"), " – Wins Above Replacement; total value in wins over replacement player."
                                ]),
                                html.P([
                                    html.Strong("WAR/$1M"), " – WAR divided by salary (in millions); efficiency per payroll dollar."
                                ]),
                                html.P([
                                    html.Strong("Salary ($M)"), " – 2025 salary in millions."
                                ]),
                                html.H6("Traditional Counting Stats", className="mt-4 mb-2"),
                                html.P([
                                    html.Strong("PA"), " – Plate appearances."
                                ]),
                                html.P([
                                    html.Strong("AB"), " – At-bats (PA minus walks, HBP, sac flies/bunts, etc.)."
                                ]),
                                html.P([
                                    html.Strong("H"), " – Hits."
                                ]),
                                html.P([
                                    html.Strong("R"), " – Runs scored."
                                ]),
                                html.P([
                                    html.Strong("RBI"), " – Runs batted in."
                                ]),
                                html.P([
                                    html.Strong("HR"), " – Home runs."
                                ]),
                                html.P([
                                    html.Strong("BB"), " – Walks drawn."
                                ]),
                                html.P([
                                    html.Strong("K"), " – Strikeouts."
                                ]),
                                html.H6("Rate / Slash-Line Stats", className="mt-4 mb-2"),
                                html.P([
                                    html.Strong("BA"), " – Batting Avg = H / AB."
                                ]),
                                html.P([
                                    html.Strong("OBP"), " – On-Base % = (H + BB + HBP) / (AB + BB + HBP + SF)."
                                ]),
                                html.P([
                                    html.Strong("SLG"), " – Slugging % = Total Bases / AB."
                                ]),
                                html.P([
                                    html.Strong("ISO"), " – Isolated Power = SLG − BA."
                                ]),
                                html.P([
                                    html.Strong("OPS"), " – On-Base + Slugging = OBP + SLG."
                                ]),
                                html.P([
                                    html.Strong("BABIP"), " – Batting Avg on Balls in Play = (H − HR) / (AB − K − HR + SF)."
                                ]),
                            ], md=6)
                        ])
                    ])
                ], className="mb-3")
            ], id="metrics-glossary-collapse", is_open=False),
            dbc.Button(
                "📚 Show/Hide Metrics Glossary",
                id="metrics-glossary-toggle",
                color="secondary",
                outline=True,
                className="mb-3"
            ),
            html.Hr()
        ])
    ]),
    
    # Filters
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Number of Players:", className="fw-bold mb-2"),
                            dcc.Slider(
                                id='top-n-slider',
                                min=10,
                                max=350,
                                step=10,
                                value=350,  # Default to showing all 350 players
                                marks={10: '10', 50: '50', 100: '100', 200: '200', 350: 'All 350'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=12)
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Loading indicator
    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div(id="loading-output")
    ),
    
    # Tabs for different views
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(
                    label="ℹ️ About This Page",
                    tab_id="about",
                    children=html.Div(id="about-content")
                ),
                dbc.Tab(
                    label="📊 All Players Stats",
                    tab_id="all-players",
                    children=html.Div(id="all-players-content")
                ),
                dbc.Tab(
                    label="💎 Undervalued Players",
                    tab_id="undervalued",
                    children=html.Div(id="undervalued-content")
                ),
            ], id="main-tabs", active_tab="about", className="mb-3")
        ])
    ]),
    
    # Main Content (switches based on tab)
    html.Div(id='main-content', children=[
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Player Statistics", className="mb-0"),
                        html.Small("Click column headers to sort", className="text-muted")
                    ]),
                    dbc.CardBody([
                        html.Div(id='stats-table-container', children=[
                            dbc.Alert("Loading data...", color="info")
                        ])
                    ])
                ])
            ], width=12)
        ])
    ]),
    
    # Footer
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.P(
                "Data from Baseball Savant via pybaseball | Analysis by Advanced Statistical Metrics",
                className="text-center text-muted small"
            )
        ])
    ], className="mt-4 mb-4")
    
], fluid=True, style={'maxWidth': '1800px', 'padding': '20px'})


@app.callback(
    Output("metrics-glossary-collapse", "is_open"),
    [Input("metrics-glossary-toggle", "n_clicks")],
    [dash.dependencies.State("metrics-glossary-collapse", "is_open")],
)
def toggle_metrics_glossary(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output('main-content', 'children'),
    [Input('top-n-slider', 'value'),
     Input('main-tabs', 'active_tab')]
)
def update_main_content(top_n, active_tab):
    """Update main content based on active tab."""
    if active_tab == "about":
        return update_about_page()
    elif active_tab == "all-players":
        table, _ = update_table(top_n)
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Player Statistics", className="mb-0"),
                        html.Small("Click column headers to sort", className="text-muted")
                    ]),
                    dbc.CardBody([
                        table
                    ])
                ])
            ], width=12)
        ])
    elif active_tab == "undervalued":
        return update_undervalued_players()
    return html.Div("Select a tab")


@app.callback(
    [Output('stats-table-container', 'children'),
     Output('loading-output', 'children')],
    [Input('top-n-slider', 'value')]
)
def update_table(top_n):
    """Update the statistics table."""
    import traceback
    try:
        # All players are hitters
        position_type = 'Hitter'
        df = fetch_player_data(position_type, top_n)
        
        if df is None or df.empty:
            error_msg = html.Div([
                dbc.Alert(
                    "No data available. Please run the data pipeline first.",
                    color="warning"
                )
            ])
            return error_msg, ""
        
        # Create comprehensive table with all stats
        table = create_comprehensive_table(df, position_type)
        
        return table, ""
    
    except requests.exceptions.ConnectionError:
        # Try loading from file directly
        try:
            df = fetch_player_data(position_type, top_n)
            if df is not None and not df.empty:
                table = create_comprehensive_table(df, position_type)
                return table, ""
        except:
            pass
        
        error_msg = html.Div([
            dbc.Alert(
                "Cannot connect to API. Please ensure the API server is running or check data files.",
                color="info"
            )
        ])
        return error_msg, ""
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in update_table: {error_trace}")  # Debug output
        error_msg = html.Div([
            dbc.Alert([
                html.H5("Error loading data"),
                html.P(str(e)),
                html.Pre(error_trace[:500], style={'fontSize': '10px'})
            ], color="danger")
        ])
        return error_msg, ""


def fetch_player_data(position_type, top_n, sort_metric='uvs'):
    """Fetch player data from API or local file."""
    try:
        # Try API first
        if position_type:
            endpoint = f"{API_BASE_URL}/api/undervalued/{'hitters' if position_type == 'Hitter' else 'pitchers'}"
        else:
            endpoint = f"{API_BASE_URL}/api/undervalued"
        
        params = {'top_n': top_n, 'year': 2025}
        response = requests.get(endpoint, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            players = data.get('players', [])
            return pd.DataFrame(players)
    except:
        pass
    
    # Fallback: Load from local file
    try:
        # Try multiple path options for different deployment scenarios
        base_paths = [
            Path(__file__).parent.parent,  # Relative to this file
            Path.cwd(),  # Current working directory (set by WSGI)
            Path(os.getcwd()),  # Explicit current working directory
        ]
        
        data_path = None
        for base_path in base_paths:
            potential_path = base_path / "data" / "processed" / "comprehensive_stats_2025.csv"
            if potential_path.exists():
                data_path = potential_path
                break
        
        if data_path and data_path.exists():
            df = pd.read_csv(data_path)
            
            # Filter by position
            if position_type:
                df = df[df.get('position_type', '') == position_type]
            
            # Apply PA/IP filters
            if position_type == 'Hitter':
                pa_col = 'pa' if 'pa' in df.columns else 'PA'
                if pa_col in df.columns:
                    df = df[df[pa_col] >= 200]
            elif position_type == 'Pitcher':
                ip_col = 'IP' if 'IP' in df.columns else 'ip'
                if ip_col in df.columns:
                    gs_col = 'GS' if 'GS' in df.columns else 'gs'
                    if gs_col in df.columns:
                        starters = df[df[gs_col] > 0]
                        relievers = df[df[gs_col] == 0]
                        starters = starters[starters[ip_col] >= 50]
                        relievers = relievers[relievers[ip_col] >= 20]
                        df = pd.concat([starters, relievers], ignore_index=True)
                    else:
                        df = df[df[ip_col] >= 50]
            
            # Always recalculate all metrics to ensure completeness
            try:
                from src.utils.metrics import calculate_all_advanced_metrics
                df = calculate_all_advanced_metrics(df)
            except Exception as e:
                print(f"Warning: Could not calculate advanced metrics: {e}")
            
            # Calculate missing metrics (handles column mapping, xISO, etc.)
            df = calculate_missing_metrics(df, position_type)
            
            # Ensure composite metrics are calculated for ALL hitters
            if position_type == 'Hitter':
                try:
                    from src.utils.tova_metrics import calculate_all_composite_metrics
                    hitters_only = df[df.get('position_type', '') == 'Hitter'].copy()
                    if len(hitters_only) > 0:
                        hitters_only = calculate_all_composite_metrics(hitters_only)
                        # Update the dataframe
                        df.loc[df.get('position_type', '') == 'Hitter', hitters_only.columns] = hitters_only
                except Exception as e:
                    print(f"Warning: Could not calculate composite metrics: {e}")
                
                # Ensure UVS metrics are calculated for ALL hitters
                try:
                    from src.utils.uvs_metrics import calculate_all_uvs_metrics
                    hitters_only = df[df.get('position_type', '') == 'Hitter'].copy()
                    if len(hitters_only) > 0:
                        hitters_only = calculate_all_uvs_metrics(hitters_only)
                        # Update the dataframe
                        df.loc[df.get('position_type', '') == 'Hitter', hitters_only.columns] = hitters_only
                except Exception as e:
                    print(f"Warning: Could not calculate UVS metrics: {e}")
            
            # Sort by rank by default (ascending - lower rank is better)
            if 'undervalued_rank' in df.columns:
                df = df.sort_values('undervalued_rank', ascending=True, na_position='last')
            elif 'uvs' in df.columns and df['uvs'].notna().any():
                df = df.sort_values('uvs', ascending=False, na_position='last')
            
            # Only limit if top_n is less than total (otherwise show all)
            if top_n < len(df):
                df = df.head(top_n)
            
            return df
    except Exception as e:
        print(f"Error loading local data: {e}")
    
    return pd.DataFrame()


def calculate_missing_metrics(df, position_type):
    """Calculate any missing metrics."""
    df = df.copy()
    
    # Map name column - prioritize columns with actual data
    name_col = None
    for col in ['last_name, first_name', 'Name', 'name']:
        if col in df.columns:
            # Check if this column has non-null values
            if df[col].notna().any():
                name_col = col
                break
    
    if name_col:
        df['name'] = df[name_col]
        # Fill any remaining NaN names with 'Unknown'
        df['name'] = df['name'].fillna('Unknown')
    else:
        df['name'] = 'Unknown'
    
    if position_type == 'Hitter':
        # Fill salary_2025 from salary_2025_x where missing (combine salary sources)
        if 'salary_2025' in df.columns and 'salary_2025_x' in df.columns:
            mask = df['salary_2025'].isna() & df['salary_2025_x'].notna()
            df.loc[mask, 'salary_2025'] = df.loc[mask, 'salary_2025_x']
        
        # Also ensure salary_2025 column exists for all players (even if NaN)
        if 'salary_2025' not in df.columns:
            if 'salary_2025_x' in df.columns:
                df['salary_2025'] = df['salary_2025_x']
            else:
                df['salary_2025'] = pd.Series(index=df.index, dtype=float)
        # Map xwOBA - prioritize est_woba if it has data, otherwise use xwoba
        if 'est_woba' in df.columns and df['est_woba'].notna().any():
            df['xwoba'] = df['est_woba']
        elif 'xwoba' in df.columns and df['xwoba'].notna().any():
            pass  # xwoba already exists and has data
        elif 'xwOBA' in df.columns:
            df['xwoba'] = df['xwOBA']
        
        # Calculate performance_diff - use existing diff column if available
        if 'performance_diff' not in df.columns:
            if 'est_woba_minus_woba_diff' in df.columns:
                df['performance_diff'] = df['est_woba_minus_woba_diff']
            elif 'woba' in df.columns and 'xwoba' in df.columns:
                df['performance_diff'] = df['woba'] - df['xwoba']
            elif 'woba' in df.columns and 'est_woba' in df.columns:
                df['performance_diff'] = df['woba'] - df['est_woba']
        
        # Calculate xISO if missing (xISO = xSLG - xBA)
        if 'xiso' not in df.columns or (df['xiso'].isna().all() if 'xiso' in df.columns else True):
            if 'est_slg' in df.columns and 'est_ba' in df.columns:
                df['xiso'] = df['est_slg'] - df['est_ba']
            elif 'xslg' in df.columns and 'xba' in df.columns:
                df['xiso'] = df['xslg'] - df['xba']
        
        # Map xHR if missing - try to calculate from expected stats if not available
        if 'xhr' not in df.columns or (df['xhr'].isna().all() if 'xhr' in df.columns else True):
            # First try to find existing xHR column
            if 'est_hr' in df.columns:
                df['xhr'] = df['est_hr']
            elif 'xHR' in df.columns:
                df['xhr'] = df['xHR']
            elif 'expected_hr' in df.columns:
                df['xhr'] = df['expected_hr']
            # If not available, estimate xHR from expected SLG and actual HR rate
            # xHR ≈ HR * (xSLG / SLG) as a rough approximation
            elif 'HR' in df.columns and 'est_slg' in df.columns and 'SLG' in df.columns:
                # Only calculate where we have valid data
                valid_mask = df['HR'].notna() & df['est_slg'].notna() & df['SLG'].notna() & (df['SLG'] > 0)
                df['xhr'] = pd.Series(index=df.index, dtype=float)
                df.loc[valid_mask, 'xhr'] = df.loc[valid_mask, 'HR'] * (df.loc[valid_mask, 'est_slg'] / df.loc[valid_mask, 'SLG'])
            elif 'HR' in df.columns and 'xslg' in df.columns and 'slg' in df.columns:
                valid_mask = df['HR'].notna() & df['xslg'].notna() & df['slg'].notna() & (df['slg'] > 0)
                df['xhr'] = pd.Series(index=df.index, dtype=float)
                df.loc[valid_mask, 'xhr'] = df.loc[valid_mask, 'HR'] * (df.loc[valid_mask, 'xslg'] / df.loc[valid_mask, 'slg'])
        
        # Calculate BA if missing (BA = H / AB)
        if 'ba' not in df.columns or (df['ba'].isna().all() if 'ba' in df.columns else True):
            if 'BA' in df.columns:
                df['ba'] = df['BA']
            elif 'H' in df.columns and 'AB' in df.columns:
                valid_mask = df['AB'].notna() & (df['AB'] > 0)
                df['ba'] = pd.Series(index=df.index, dtype=float)
                df.loc[valid_mask, 'ba'] = df.loc[valid_mask, 'H'] / df.loc[valid_mask, 'AB']
        
        # Map K column (strikeouts) - try SO or Strikes if K doesn't exist
        # First, ensure 'K' column exists (for direct mapping)
        if 'K' not in df.columns:
            df['K'] = pd.Series(index=df.index, dtype=float)
        
        # Fill K from available sources (prioritize SO, then Strikes)
        if 'SO' in df.columns:
            mask = df['K'].isna() & df['SO'].notna()
            df.loc[mask, 'K'] = df.loc[mask, 'SO']
        if 'Strikes' in df.columns:
            mask = df['K'].isna() & df['Strikes'].notna()
            df.loc[mask, 'K'] = df.loc[mask, 'Strikes']
        if 'k' in df.columns:
            mask = df['K'].isna() & df['k'].notna()
            df.loc[mask, 'K'] = df.loc[mask, 'k']
        
        # Also check for 'SO' column from FanGraphs merge (might be named differently)
        # Check for any column with 'so' or 'strikeout' in the name
        for col in df.columns:
            if col.lower() in ['so', 'strikeouts', 'strikes'] and col not in ['SO', 'Strikes', 'k', 'K']:
                mask = df['K'].isna() & df[col].notna()
                if mask.any():
                    df.loc[mask, 'K'] = df.loc[mask, col]
                    break
        
        # Then map to lowercase 'k' for table column
        if 'k' not in df.columns:
            df['k'] = pd.Series(index=df.index, dtype=float)
        
        # Fill k from K or other sources
        if 'K' in df.columns:
            mask = df['k'].isna() & df['K'].notna()
            df.loc[mask, 'k'] = df.loc[mask, 'K']
        if 'SO' in df.columns:
            mask = df['k'].isna() & df['SO'].notna()
            df.loc[mask, 'k'] = df.loc[mask, 'SO']
        if 'Strikes' in df.columns:
            mask = df['k'].isna() & df['Strikes'].notna()
            df.loc[mask, 'k'] = df.loc[mask, 'Strikes']
        
        # Calculate WAR per $1M if missing
        if 'war_per_salary' not in df.columns or (df['war_per_salary'].isna().all() if 'war_per_salary' in df.columns else True):
            war_col = None
            if 'WAR' in df.columns:
                war_col = 'WAR'
            elif 'war' in df.columns:
                war_col = 'war'
            
            # Try to find the best salary column (prioritize one with most data)
            salary_col = None
            salary_cols_to_try = ['salary_2025', 'salary_2025_x', 'salary_2025_y', 'Salary ($M)', 'salary']
            best_col = None
            best_count = 0
            
            for col in salary_cols_to_try:
                if col in df.columns:
                    count = df[col].notna().sum()
                    if count > best_count:
                        best_col = col
                        best_count = count
            
            salary_col = best_col
            
            if war_col and salary_col:
                # Initialize column for all players
                if 'war_per_salary' not in df.columns:
                    df['war_per_salary'] = pd.Series(index=df.index, dtype=float)
                
                # Calculate only where both WAR and salary exist
                valid_mask = df[war_col].notna() & df[salary_col].notna() & (df[salary_col] > 0)
                df.loc[valid_mask, 'war_per_salary'] = df.loc[valid_mask, war_col] / (df.loc[valid_mask, salary_col] + 0.1)
                
                # Also try salary_2025_x as fallback for players missing salary_2025
                if salary_col == 'salary_2025' and 'salary_2025_x' in df.columns:
                    # Find players with WAR but missing salary_2025
                    fallback_mask = df[war_col].notna() & df['salary_2025'].isna() & df['salary_2025_x'].notna() & (df['salary_2025_x'] > 0)
                    df.loc[fallback_mask, 'war_per_salary'] = df.loc[fallback_mask, war_col] / (df.loc[fallback_mask, 'salary_2025_x'] + 0.1)
        
        # Calculate value_score
        if 'value_score' not in df.columns:
            if 'performance_diff' in df.columns:
                perf_diff = df['performance_diff'].fillna(0)
                df['value_score'] = (perf_diff - perf_diff.min()) / (perf_diff.max() - perf_diff.min() + 1e-10)
            else:
                df['value_score'] = 0.0
        
        # Calculate rank
        if 'undervalued_rank' not in df.columns:
            if 'value_score' in df.columns:
                df['undervalued_rank'] = df['value_score'].rank(ascending=False, method='min').astype(int)
    
    else:  # Pitcher
        # Map columns
        if 'ERA' in df.columns and 'era' not in df.columns:
            df['era'] = df['ERA']
        if 'xERA' in df.columns and 'xera' not in df.columns:
            df['xera'] = df['xERA']
        if 'CSW%' in df.columns and 'csw_percent' not in df.columns:
            df['csw_percent'] = df['CSW%']
        
        # Calculate performance_diff
        if 'performance_diff' not in df.columns:
            if 'era' in df.columns and 'xera' in df.columns:
                df['performance_diff'] = df['xera'] - df['era']  # Negative = better
        
        # Calculate value_score
        if 'value_score' not in df.columns:
            if 'performance_diff' in df.columns:
                perf_diff = df['performance_diff'].fillna(0)
                df['value_score'] = (perf_diff - perf_diff.min()) / (perf_diff.max() - perf_diff.min() + 1e-10)
            else:
                df['value_score'] = 0.0
        
        # Calculate rank
        if 'undervalued_rank' not in df.columns:
            if 'value_score' in df.columns:
                df['undervalued_rank'] = df['value_score'].rank(ascending=False, method='min').astype(int)
    
    return df


def create_comprehensive_table(df, position_type):
    """Create a comprehensive table with all relevant statistics."""
    
    # Define all columns based on position type - using clear, readable names
    if position_type == 'Hitter':
        all_columns = [
            # Basic Info
            {'name': 'Rank', 'id': 'undervalued_rank', 'type': 'numeric'},
            {'name': 'Player', 'id': 'name', 'type': 'text'},
            
            # Quality of Contact
            {'name': 'Barrel%', 'id': 'barrel_batted_rate', 'type': 'numeric'},
            {'name': 'HardHit%', 'id': 'hard_hit_percent', 'type': 'numeric'},
            {'name': 'Exit Velo', 'id': 'avg_exit_velocity', 'type': 'numeric'},
            {'name': 'Sweet Spot%', 'id': 'sweet_spot_percent', 'type': 'numeric'},
            
            # Expected Outcomes
            {'name': 'wOBA', 'id': 'woba', 'type': 'numeric'},
            {'name': 'xwOBA', 'id': 'xwoba', 'type': 'numeric'},
            {'name': 'xBA', 'id': 'xba', 'type': 'numeric'},
            {'name': 'xSLG', 'id': 'xslg', 'type': 'numeric'},
            {'name': 'xISO', 'id': 'xiso', 'type': 'numeric'},
            {'name': 'xHR', 'id': 'xhr', 'type': 'numeric'},
            
            # Plate Discipline
            {'name': 'BB%', 'id': 'bb_percent', 'type': 'numeric'},
            {'name': 'K%', 'id': 'k_percent', 'type': 'numeric'},
            {'name': 'Chase%', 'id': 'o_swing_percent', 'type': 'numeric'},
            {'name': 'Z-Contact%', 'id': 'z_contact_percent', 'type': 'numeric'},
            {'name': 'Contact%', 'id': 'contact_percent', 'type': 'numeric'},
            
            # Batted Ball Profile
            {'name': 'GB%', 'id': 'gb_percent', 'type': 'numeric'},
            {'name': 'FB%', 'id': 'fb_percent', 'type': 'numeric'},
            {'name': 'LD%', 'id': 'ld_percent', 'type': 'numeric'},
            {'name': 'Pull%', 'id': 'pull_percent', 'type': 'numeric'},
            {'name': 'Oppo%', 'id': 'oppo_percent', 'type': 'numeric'},
            
            # Run Value
            {'name': 'wRC+', 'id': 'wrc_plus', 'type': 'numeric'},
            {'name': 'WAR', 'id': 'war', 'type': 'numeric'},
            {'name': 'WAR/$1M', 'id': 'war_per_salary', 'type': 'numeric'},
            
            # Salary
            {'name': 'Salary ($M)', 'id': 'salary_2025', 'type': 'numeric'},
            
            # Traditional
            {'name': 'PA', 'id': 'pa', 'type': 'numeric'},
            {'name': 'AB', 'id': 'ab', 'type': 'numeric'},
            {'name': 'H', 'id': 'h', 'type': 'numeric'},
            {'name': 'R', 'id': 'r', 'type': 'numeric'},
            {'name': 'RBI', 'id': 'rbi', 'type': 'numeric'},
            {'name': 'HR', 'id': 'hr', 'type': 'numeric'},
            {'name': 'BB', 'id': 'bb', 'type': 'numeric'},
            {'name': 'K', 'id': 'k', 'type': 'numeric'},
            {'name': 'BA', 'id': 'ba', 'type': 'numeric'},
            {'name': 'OBP', 'id': 'obp', 'type': 'numeric'},
            {'name': 'SLG', 'id': 'slg', 'type': 'numeric'},
            {'name': 'ISO', 'id': 'iso', 'type': 'numeric'},
            {'name': 'OPS', 'id': 'ops', 'type': 'numeric'},
            {'name': 'BABIP', 'id': 'babip', 'type': 'numeric'},
        ]
    else:  # Pitcher
        all_columns = [
            # Basic Info
            {'name': 'Rank', 'id': 'undervalued_rank', 'type': 'numeric'},
            {'name': 'Player', 'id': 'name', 'type': 'text'},
            
            # Value Metrics
            {'name': 'Value Score', 'id': 'value_score', 'type': 'numeric'},
            {'name': 'Perf Diff', 'id': 'performance_diff', 'type': 'numeric'},
            
            # Expected vs Actual
            {'name': 'ERA', 'id': 'era', 'type': 'numeric'},
            {'name': 'xERA', 'id': 'xera', 'type': 'numeric'},
            {'name': 'xFIP', 'id': 'xfip', 'type': 'numeric'},
            {'name': 'SIERA', 'id': 'siera', 'type': 'numeric'},
            {'name': 'FIP', 'id': 'fip', 'type': 'numeric'},
            
            # Contact Suppression
            {'name': 'Barrel%', 'id': 'barrel_batted_rate', 'type': 'numeric'},
            {'name': 'HardHit%', 'id': 'hard_hit_percent', 'type': 'numeric'},
            {'name': 'Exit Velo', 'id': 'avg_exit_velocity', 'type': 'numeric'},
            
            # Whiff Ability
            {'name': 'Whiff%', 'id': 'whiff_percent', 'type': 'numeric'},
            {'name': 'CSW%', 'id': 'csw_percent', 'type': 'numeric'},
            {'name': 'K%', 'id': 'k_percent', 'type': 'numeric'},
            {'name': 'BB%', 'id': 'bb_percent', 'type': 'numeric'},
            {'name': 'K-BB%', 'id': 'k_bb_percent', 'type': 'numeric'},
            
            # Contextual Luck
            {'name': 'BABIP', 'id': 'babip', 'type': 'numeric'},
            {'name': 'LOB%', 'id': 'lob_percent', 'type': 'numeric'},
            {'name': 'HR/FB%', 'id': 'hr_fb_percent', 'type': 'numeric'},
            
            # Advanced
            {'name': 'Pitch Decept', 'id': 'pitch_deception_index', 'type': 'numeric'},
            {'name': 'True Value', 'id': 'true_player_value', 'type': 'numeric'},
            {'name': 'ERA-', 'id': 'era_minus', 'type': 'numeric'},
            {'name': 'FIP-', 'id': 'fip_minus', 'type': 'numeric'},
            
            # Traditional
            {'name': 'IP', 'id': 'IP', 'type': 'numeric'},
            {'name': 'H', 'id': 'h', 'type': 'numeric'},
            {'name': 'ER', 'id': 'er', 'type': 'numeric'},
            {'name': 'BB', 'id': 'bb', 'type': 'numeric'},
            {'name': 'SO', 'id': 'so', 'type': 'numeric'},
            {'name': 'HR', 'id': 'hr', 'type': 'numeric'},
            {'name': 'W', 'id': 'w', 'type': 'numeric'},
            {'name': 'L', 'id': 'l', 'type': 'numeric'},
        ]
    
    # Find available columns in the dataframe with better matching
    available_columns = []
    used_columns = set()  # Track which columns we've already matched
    
    # Define column name alternatives/mappings
    # Priority: try exact match first, then alternatives
    column_alternatives = {
        'xwoba': ['est_woba', 'xwoba', 'xwOBA', 'est_wOBA'],  # est_woba has data
        'xba': ['est_ba', 'xba', 'xBA'],  # est_ba might have data
        'xslg': ['est_slg', 'xslg', 'xSLG'],  # est_slg might have data
        'performance_diff': ['est_woba_minus_woba_diff', 'performance_diff', 'est_wOBA_minus_wOBA_diff', 'woba_diff'],
        'bb_percent': ['BB%', 'bb_percent', 'bb%', 'walk_percent'],  # BB% exists
        'k_percent': ['K%', 'k_percent', 'k%', 'strikeout_percent'],  # K% exists
        'k': ['K', 'k', 'SO', 'so', 'Strikes', 'strikes'],  # K might be SO or Strikes
        'o_swing_percent': ['O-Swing%', 'o_swing_percent', 'o_swing%', 'chase_percent'],
        'z_contact_percent': ['Z-Contact%', 'z_contact_percent', 'z_contact%'],
        'gb_percent': ['GB%', 'gb_percent', 'gb%'],
        'fb_percent': ['FB%', 'fb_percent', 'fb%'],
        'ld_percent': ['LD%', 'ld_percent', 'ld%'],
        'xba': ['est_ba', 'xba', 'xBA'],  # est_ba might exist
        'xslg': ['est_slg', 'xslg', 'xSLG'],  # est_slg might exist
        'xiso': ['xiso', 'xISO'],
        'wrc_plus': ['wRC+', 'wrc_plus', 'wRC+', 'wrcplus'],
        'war': ['WAR', 'war'],
        'ab': ['AB', 'ab'],
        'h': ['H', 'h'],
        'hr': ['HR', 'hr'],
        'bb': ['BB', 'bb'],
        'k': ['K', 'k', 'SO', 'so', 'Strikes'],
        'pull_percent': ['Pull%', 'pull_percent', 'pull%'],
        'oppo_percent': ['Oppo%', 'oppo_percent', 'oppo%'],
        # Quality of contact - these exist but may be decimals
        'barrel_batted_rate': ['Barrel%', 'barrel_batted_rate', 'barrel_percent', 'Barrels'],
        'hard_hit_percent': ['HardHit%', 'hard_hit_percent', 'Hard%', 'hard_hit'],
        'avg_exit_velocity': ['avg_exit_velocity', 'Exit Velo', 'exit_velocity', 'launch_speed'],
        'sweet_spot_percent': ['sweet_spot_percent', 'Sweet Spot%', 'sweet_spot'],
    }
    
    for col_config in all_columns:
        col_id = col_config['id']
        found = False
        
        # Try exact match first
        if col_id in df.columns and col_id not in used_columns:
            available_columns.append({
                'name': col_config['name'],
                'id': col_id,
                'type': col_config['type']
            })
            used_columns.add(col_id)
            found = True
        else:
            # Try alternatives
            alternatives = column_alternatives.get(col_id, [col_id])
            for alt in alternatives:
                if alt in df.columns and alt not in used_columns:
                    # Use the original col_id for the table, but map from alt column
                    available_columns.append({
                        'name': col_config['name'],
                        'id': col_id,  # Use original ID for table
                        'source_id': alt,  # Store the actual column name
                        'type': col_config['type']
                    })
                    used_columns.add(alt)
                    found = True
                    break
            
            # If still not found, try case-insensitive and fuzzy matching
            if not found:
                col_id_lower = col_id.lower().replace('_', '').replace('-', '')
                for df_col in df.columns:
                    if df_col in used_columns:
                        continue
                    df_col_lower = df_col.lower().replace('_', '').replace('-', '')
                    if df_col_lower == col_id_lower:
                        available_columns.append({
                            'name': col_config['name'],
                            'id': df_col,
                            'type': col_config['type']
                        })
                        used_columns.add(df_col)
                        found = True
                        break
    
    # Ensure name column exists
    if 'name' not in df.columns:
        for col in ['last_name, first_name', 'Name', 'name']:
            if col in df.columns:
                df['name'] = df[col]
                break
        else:
            df['name'] = 'Unknown'
    
    # Ensure name column is in available_columns
    name_in_cols = any(col['id'] == 'name' for col in available_columns)
    if not name_in_cols:
        # Find the name column
        for col in ['name', 'last_name, first_name', 'Name']:
            if col in df.columns:
                available_columns.insert(1, {
                    'name': 'Player',
                    'id': 'name' if col == 'name' else col,
                    'type': 'text'
                })
                if col != 'name':
                    df['name'] = df[col]
                break
    
    # Format columns for dash_table
    # IMPORTANT: Use all_columns to ensure ALL columns are included, not just available_columns
    # This ensures columns appear even if data matching failed
    formatted_columns = []
    
    # Create a set of IDs from available_columns for quick lookup
    available_ids = {col['id'] for col in available_columns}
    
    # First, add all columns from all_columns (this ensures all columns appear)
    for col_config in all_columns:
        col_id = col_config['id']
        col_def = {
            'name': col_config['name'],
            'id': col_id
        }
        
        # Add formatting for numeric columns
        if col_config['type'] == 'numeric':
            # Special formatting for rank column - show as integer
            if col_id == 'undervalued_rank':
                col_def['format'] = {'specifier': '.0f'}  # Integer format (no decimals)
                col_def['type'] = 'numeric'
            elif 'percent' in col_id.lower() or '%' in col_config['name']:
                col_def['format'] = {'specifier': '.1f'}
                col_def['type'] = 'numeric'
            # Handle decimal percentages (0.088 -> 8.8%)
            elif col_id in ['barrel_batted_rate', 'hard_hit_percent', 'sweet_spot_percent']:
                col_def['format'] = {'specifier': '.1f'}
                col_def['type'] = 'numeric'
            elif 'score' in col_id.lower() or 'index' in col_id.lower() or 'value' in col_id.lower():
                col_def['format'] = {'specifier': '.3f'}
                col_def['type'] = 'numeric'
            elif 'woba' in col_id.lower() or 'ba' in col_id.lower() or 'slg' in col_id.lower():
                col_def['format'] = {'specifier': '.3f'}
                col_def['type'] = 'numeric'
            elif 'era' in col_id.lower() or 'fip' in col_id.lower() or 'siera' in col_id.lower():
                col_def['format'] = {'specifier': '.2f'}
                col_def['type'] = 'numeric'
            elif 'velocity' in col_id.lower() or 'velo' in col_id.lower():
                col_def['format'] = {'specifier': '.1f'}
                col_def['type'] = 'numeric'
            else:
                col_def['format'] = {'specifier': '.2f'}
                col_def['type'] = 'numeric'
        
        formatted_columns.append(col_def)
    
    # Ensure all columns in formatted_columns exist in df
    # Create a clean dataframe with only the columns we need
    table_data = pd.DataFrame(index=df.index)
    
    # Build a comprehensive mapping of table column IDs to source column IDs
    col_mapping = {}
    
    # First, use available_columns which has source_id info
    for col in available_columns:
        table_id = col['id']
        source_id = col.get('source_id', table_id)
        if source_id in df.columns:
            col_mapping[table_id] = source_id
        elif table_id in df.columns:
            col_mapping[table_id] = table_id
    
    # Then, for any columns in formatted_columns, try to find them using alternatives
    for col in formatted_columns:
        col_id = col['id']
        if col_id not in col_mapping:
            # Try alternatives
            alternatives = column_alternatives.get(col_id, [col_id])
            for alt in alternatives:
                if alt in df.columns:
                    col_mapping[col_id] = alt
                    break
            # If still not found, try direct match
            if col_id not in col_mapping and col_id in df.columns:
                col_mapping[col_id] = col_id
    
    # Direct column mapping - bypass complex logic
    # Map table column IDs directly to known data column names
    direct_mapping = {
        'barrel_batted_rate': 'Barrel%',
        'hard_hit_percent': 'HardHit%',
        'bb_percent': 'BB%',
        'k_percent': 'K%',
        'o_swing_percent': 'O-Swing%',
        'z_contact_percent': 'Z-Contact%',
        'gb_percent': 'GB%',
        'fb_percent': 'FB%',
        'ld_percent': 'LD%',
        'pull_percent': 'Pull%',
        'oppo_percent': 'Oppo%',
        'wrc_plus': 'wRC+',
        'war': 'WAR',
        'ab': 'AB',
        'h': 'H',
        'r': 'R',
        'rbi': 'RBI',
        'hr': 'HR',
        'bb': 'BB',
        'k': 'K' if 'K' in df.columns and df['K'].notna().any() else ('SO' if 'SO' in df.columns and df['SO'].notna().any() else ('Strikes' if 'Strikes' in df.columns and df['Strikes'].notna().any() else None)),
        'iso': 'ISO',
        'ops': 'OPS',
        'ba': 'BA' if 'BA' in df.columns else ('ba' if 'ba' in df.columns else None),
        'obp': 'OBP',
        'slg': 'SLG',
        'babip': 'BABIP',
        'xhr': 'est_hr' if 'est_hr' in df.columns else ('xHR' if 'xHR' in df.columns else ('xhr' if 'xhr' in df.columns else None)),
        'contact_percent': 'Contact%' if 'Contact%' in df.columns else ('contact_percent' if 'contact_percent' in df.columns else None),
        'war_per_salary': None,  # Will be calculated
        # These may not exist in data (require Statcast pitch-level data)
        'avg_exit_velocity': 'avg_exit_velocity' if 'avg_exit_velocity' in df.columns else None,
        'sweet_spot_percent': 'sweet_spot_percent' if 'sweet_spot_percent' in df.columns else None,
        'xba': 'est_ba' if 'est_ba' in df.columns else ('xba' if 'xba' in df.columns else None),
        'xslg': 'est_slg' if 'est_slg' in df.columns else ('xslg' if 'xslg' in df.columns else None),
        'xiso': 'xiso' if 'xiso' in df.columns else None,  # xISO is calculated, so it should exist
        'salary_2025': 'salary_2025' if 'salary_2025' in df.columns and df['salary_2025'].notna().any() else ('salary_2025_x' if 'salary_2025_x' in df.columns and df['salary_2025_x'].notna().any() else ('salary_2025_y' if 'salary_2025_y' in df.columns and df['salary_2025_y'].notna().any() else ('Salary ($M)' if 'Salary ($M)' in df.columns and df['Salary ($M)'].notna().any() else None))),
        'tova_plus': 'tova_plus' if 'tova_plus' in df.columns else None,
        'upi': 'upi' if 'upi' in df.columns else None,
        'ops_2_0': 'ops_2_0' if 'ops_2_0' in df.columns else None,
        'tova_dollar': 'tova_dollar' if 'tova_dollar' in df.columns else None,
        'bov': 'bov' if 'bov' in df.columns else None,
        'bov_power': 'bov_power' if 'bov_power' in df.columns else None,
        'uvs': 'uvs' if 'uvs' in df.columns else None,
    }
    # Remove None values
    direct_mapping = {k: v for k, v in direct_mapping.items() if v is not None}
    
    # Now populate table_data using the mapping
    # Build as a list of dicts (what Dash DataTable expects)
    table_records = []
    
    # Reset index to ensure we can iterate properly
    df = df.reset_index(drop=True)
    
    for idx in range(len(df)):
        row_dict = {}
        for col in formatted_columns:
            col_id = col['id']
            
            # Try direct mapping first (this is the most reliable)
            source_col = direct_mapping.get(col_id)
            if not source_col:
                # Try col_mapping (from available_columns)
                source_col = col_mapping.get(col_id)
            if not source_col:
                # Try direct match
                source_col = col_id if col_id in df.columns else None
            if not source_col:
                # Last resort: check column_alternatives
                alternatives = column_alternatives.get(col_id, [])
                for alt in alternatives:
                    if alt in df.columns:
                        source_col = alt
                        break
            
            # Get the value
            if source_col and source_col in df.columns:
                try:
                    value = df.iloc[idx][source_col]
                except (KeyError, IndexError):
                    value = None
                
                # Convert decimal percentages to percentages (0.088 -> 8.8)
                # Check both the table column ID and the source column name
                percent_cols = ['barrel_batted_rate', 'hard_hit_percent', 'sweet_spot_percent',
                               'bb_percent', 'k_percent', 'o_swing_percent', 'z_contact_percent',
                               'gb_percent', 'fb_percent', 'ld_percent', 'pull_percent', 'oppo_percent', 'contact_percent']
                percent_source_cols = ['Barrel%', 'HardHit%', 'BB%', 'K%', 'O-Swing%', 'Z-Contact%',
                                      'GB%', 'FB%', 'LD%', 'Pull%', 'Oppo%', 'Contact%', 'Contact%']
                
                # Check if this is a percentage column that needs conversion
                is_percent_col = (col_id in percent_cols) or (source_col in percent_source_cols) or ('%' in str(source_col))
                
                if is_percent_col:
                    # If value is between 0 and 1.5 (likely a decimal percentage), multiply by 100
                    if pd.notna(value) and isinstance(value, (int, float)):
                        if value < 1.5 and value >= 0:
                            value = value * 100
                
                # Replace NaN/None with empty string for display
                if pd.isna(value) or value is None:
                    row_dict[col_id] = ''
                else:
                    # Convert numpy types to native Python types for Dash compatibility
                    try:
                        if hasattr(value, 'item'):  # numpy scalar (np.float64, np.int64, etc.)
                            value = value.item()
                        elif isinstance(value, (np_integer, np_floating)):
                            value = float(value) if isinstance(value, np_floating) else int(value)
                        elif isinstance(value, pd.Series):
                            value = value.iloc[0] if len(value) > 0 else ''
                        
                        # Special handling for rank column - ensure it's an integer
                        if col_id == 'undervalued_rank' and isinstance(value, (int, float)):
                            value = int(value)
                    except (AttributeError, ValueError, TypeError):
                        pass  # Keep original value if conversion fails
                    row_dict[col_id] = value
            elif col_id in df.columns:
                # Fallback: try direct match
                try:
                    value = df.iloc[idx][col_id]
                except (KeyError, IndexError):
                    value = None
                
                # Convert percentages if needed
                percent_cols = ['barrel_batted_rate', 'hard_hit_percent', 'sweet_spot_percent',
                               'bb_percent', 'k_percent', 'o_swing_percent', 'z_contact_percent',
                               'gb_percent', 'fb_percent', 'ld_percent', 'pull_percent', 'oppo_percent', 'contact_percent']
                if col_id in percent_cols and pd.notna(value) and isinstance(value, (int, float)):
                    if value < 1.5 and value >= 0:
                        value = value * 100
                
                if pd.isna(value) or value is None:
                    row_dict[col_id] = ''
                else:
                    # Convert numpy types to native Python types for Dash compatibility
                    try:
                        if hasattr(value, 'item'):  # numpy scalar (np.float64, np.int64, etc.)
                            value = value.item()
                        elif isinstance(value, (np_integer, np_floating)):
                            value = float(value) if isinstance(value, np_floating) else int(value)
                        elif isinstance(value, pd.Series):
                            value = value.iloc[0] if len(value) > 0 else ''
                        
                        # Special handling for rank column - ensure it's an integer
                        if col_id == 'undervalued_rank' and isinstance(value, (int, float)):
                            value = int(value)
                    except (AttributeError, ValueError, TypeError):
                        pass  # Keep original value if conversion fails
                    row_dict[col_id] = value
            else:
                # Column doesn't exist - use empty string
                row_dict[col_id] = ''
        
        table_records.append(row_dict)
    
    # Debug: Check first row to see what data we have
    if len(table_records) > 0:
        first_row = table_records[0]
        # Count non-empty values
        non_empty = sum(1 for v in first_row.values() if v != '' and v is not None)
        total_cols = len(first_row)
        print(f"DEBUG: First row has {non_empty}/{total_cols} non-empty values")
        # Show sample of what we have
        sample_cols = ['barrel_batted_rate', 'hard_hit_percent', 'bb_percent', 'k_percent', 'wrc_plus', 'war']
        for col_id in sample_cols:
            if col_id in first_row:
                print(f"  {col_id}: {first_row[col_id]} (type: {type(first_row[col_id]).__name__})")
    
    # Create the table - use table_records directly (list of dicts is what Dash expects)
    table = dash_table.DataTable(
        id='stats-datatable',
        columns=formatted_columns,
        data=table_records,
        
        # Styling
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '13px',
            'border': '1px solid #dee2e6',
            'whiteSpace': 'normal',
            'height': 'auto',
            'minWidth': '80px',
            'maxWidth': '120px'
        },
        style_header={
            'backgroundColor': '#2c3e50',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'fontSize': '13px',
            'border': '1px solid #1a252f',
            'position': 'sticky',
            'top': 0,
            'whiteSpace': 'normal',
            'height': 'auto',
            'padding': '12px 8px'
        },
        style_data={
            'backgroundColor': 'white',
            'color': 'black',
            'border': '1px solid #dee2e6'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8f9fa'
            },
            {
                'if': {'column_id': 'value_score'},
                'textAlign': 'center',
                'fontWeight': 'bold',
                'color': '#2e7d32'
            },
            {
                'if': {'column_id': 'Rank'},
                'textAlign': 'center',
                'fontWeight': 'bold',
                'width': '60px'
            },
            {
                'if': {'column_id': 'name'},
                'fontWeight': 'bold',
                'minWidth': '150px',
                'maxWidth': '200px'
            },
            {
                'if': {'column_id': 'Rank'},
                'minWidth': '60px',
                'maxWidth': '80px'
            },
            {
                'if': {'column_id': 'Player'},
                'minWidth': '150px',
                'maxWidth': '200px'
            }
        ],
        
        # Functionality
        sort_action='native',
        filter_action='native',
        page_action='native',
        page_current=0,
        page_size=50,
        
        # Export
        export_format='csv',
        export_headers='display',
        
        # Responsive
        fixed_rows={'headers': True},
        style_table={
            'overflowX': 'auto',
            'maxHeight': '800px',
            'minWidth': '100%'
        },
        
        # Tooltips
        tooltip_duration=None
    )
    
    return table


def update_about_page():
    """Display the About This Page content."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.P([
                                "This website examines all the 350 hitters during the 2025 MLB regular season that had a good enough sample size with at least 200 plate appearances. However, this page doesn't just include the basic statistics of these players as most of the statistics are advanced statistics, like Hard Hit% and LD%. Additionally, expected performance is also tracked with statistics like wOBA and xSLG that are explained on the next tab in the metrics glossary."
                            ], style={'fontSize': '16px', 'lineHeight': '1.8', 'marginBottom': '25px'}),
                            
                            html.P([
                                "Using those advanced statistics, I came up with a formula to find the most undervalued hitters in the 2025 regular season. A player is undervalued when their underlying performance, meaning their expected metrics, quality of contact, and efficiency, is better than their surface stats and salary. ",
                                html.Strong("As a result, this undervaluation score (UVS) formula rewards:"),
                            ], style={'fontSize': '16px', 'lineHeight': '1.8', 'marginBottom': '15px'}),
                            
                            html.Ul([
                                html.Li("Great expected performance"),
                                html.Li("Strong plate discipline and contact quality"),
                                html.Li("High WAR per $1M"),
                                html.Li("Positive luck differential"),
                                html.Li("Strong run creation"),
                                html.Li("Moderate to low salary or small market exposure")
                            ], style={'fontSize': '16px', 'lineHeight': '1.8', 'marginBottom': '25px', 'marginLeft': '20px'}),
                            
                            html.P([
                                "There are 7 indexes created that go into the final UVS calculation which is seen on the third tab titled \"Undervalued Players.\" Each index consists of a collection of both advanced and basic metrics seen in the metrics glossary. However, not each of those indexes are weighted the same. Due to the main goal being finding undervalued performance, an expected performance index is weighted the most to find out which hitters have been unlucky. Then, a contact quality index is weighted the second highest followed by a plate discipline index and a run production index. The plate discipline index is weighted slightly more than the run production index because another aim of this formula is to predict future performance and plate discipline is a stronger predictor of future performance, while run production mostly measures past results. Then, a salary efficiency index is included to see their efficiency per $1M before a luck adjustment index concludes the final dynamic of the formula, to see how unlucky a player has been with their balls."
                            ], style={'fontSize': '16px', 'lineHeight': '1.8', 'marginBottom': '25px'}),
                            
                            html.P([
                                "Overall, this website aims to provide MLB fans an easily accessible source to see a bunch of basic and advanced metrics on hitters with a solid sample size during the 2025 regular season. All this is set up for the primary goal of this website, which is to rank how undervalued all 350 hitters with at least 200 plate appearances were. From this, users can learn new names of players who weren't talked about enough this past year and keep an eye out for them becoming possible stars/superstars in the coming years. From the ranking, users can also gain a deeper appreciation for the greatness of already well-recognized superstars!"
                            ], style={'fontSize': '16px', 'lineHeight': '1.8'})
                        ])
                    ])
                ], className="mb-4")
            ], width=10, className="mx-auto")
        ])
    ], fluid=True)


def update_undervalued_players():
    """Display undervalued players analysis."""
    try:
        # Load main data for UVS scores - try multiple path options
        base_paths = [
            Path(__file__).parent.parent,  # Relative to this file
            Path.cwd(),  # Current working directory (set by WSGI)
            Path(os.getcwd()),  # Explicit current working directory
        ]
        
        main_data_path = None
        for base_path in base_paths:
            potential_path = base_path / "data" / "processed" / "comprehensive_stats_2025.csv"
            if potential_path.exists():
                main_data_path = potential_path
                break
        
        uvs_table = None
        if main_data_path and main_data_path.exists():
            try:
                df_main = pd.read_csv(main_data_path)
                hitters = df_main[df_main.get('position_type', '') == 'Hitter'].copy()
                
                # Get player name
                def get_name(row):
                    if 'name' in row.index and pd.notna(row['name']):
                        return str(row['name'])
                    elif 'Name' in row.index and pd.notna(row['Name']):
                        return str(row['Name'])
                    elif 'last_name, first_name' in row.index and pd.notna(row['last_name, first_name']):
                        return str(row['last_name, first_name'])
                    return 'Unknown'
                
                hitters['Player'] = hitters.apply(get_name, axis=1)
                
                # Get UVS scores
                if 'uvs' in hitters.columns:
                    uvs_data = hitters[['Player', 'uvs']].copy()
                    uvs_data = uvs_data[uvs_data['uvs'].notna()].copy()
                    # Sort first, then remove duplicates - this keeps the highest UVS for each player
                    uvs_data = uvs_data.sort_values('uvs', ascending=False)
                    uvs_data = uvs_data.drop_duplicates(subset=['Player'], keep='first')
                    uvs_data['UVS'] = uvs_data['uvs'].round(4)
                    uvs_data = uvs_data[['Player', 'UVS']]
                    
                    uvs_table = dash_table.DataTable(
                        data=uvs_data.to_dict('records'),
                        columns=[{"name": col, "id": col} for col in uvs_data.columns],
                        style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '12px'},
                        style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
                        sort_action="native",
                        filter_action="native",
                        page_action="native",
                        page_size=25,
                        export_format="csv",
                        export_headers="display"
                    )
            except Exception as e:
                print(f"Error loading UVS data: {e}")
                uvs_table = dbc.Alert("Could not load UVS data.", color="warning")
        
        return dbc.Container([
            # UVS Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("UVS (Undervaluation Score)", className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Div([
                                html.P([
                                    html.Strong("UVS = "),
                                    "0.25(EPI) + 0.20(CQI) + 0.15(PDI) + 0.15(RPI) + 0.10(SE) + 0.10(LA)"
                                ], style={'fontFamily': 'monospace', 'fontSize': '14px', 'marginBottom': '20px'}),
                                html.Div([
                                    html.P([
                                        html.Strong("EPI (Expected Performance Index):"),
                                        " mean(z(xwOBA), z(xSLG), z(xBA), z(xISO))"
                                    ], style={'fontFamily': 'monospace', 'fontSize': '12px', 'marginBottom': '5px'}),
                                    html.P([
                                        html.Strong("CQI (Contact Quality Index):"),
                                        " mean(z(Barrel%), z(HardHit%), z(Exit Velo), z(Sweet Spot%))"
                                    ], style={'fontFamily': 'monospace', 'fontSize': '12px', 'marginBottom': '5px'}),
                                    html.P([
                                        html.Strong("PDI (Plate Discipline Index):"),
                                        " z(BB%) - z(K%) - z(O-Swing%) + z(Z-Contact%) + z(Contact%)"
                                    ], style={'fontFamily': 'monospace', 'fontSize': '12px', 'marginBottom': '5px'}),
                                    html.P([
                                        html.Strong("RPI (Run Production Index):"),
                                        " mean(z(wRC+), z(wOBA), z(OPS), z(ISO), z(R), z(RBI))"
                                    ], style={'fontFamily': 'monospace', 'fontSize': '12px', 'marginBottom': '5px'}),
                                    html.P([
                                        html.Strong("SE (Salary Efficiency):"),
                                        " z(WAR per $1M)"
                                    ], style={'fontFamily': 'monospace', 'fontSize': '12px', 'marginBottom': '5px'}),
                                    html.P([
                                        html.Strong("LA (Luck Adjustment):"),
                                        " mean(z(xwOBA - wOBA), z(xBA - BA), z(xSLG - SLG))"
                                    ], style={'fontFamily': 'monospace', 'fontSize': '12px', 'marginBottom': '5px'}),
                                ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px'}),
                            html.Div([
                                html.H6("The Formula", style={'marginTop': '20px', 'marginBottom': '15px'}),
                                html.P([
                                    html.Strong("z = (x - x̄) / σ", style={'fontFamily': 'monospace', 'fontSize': '16px'})
                                ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                                html.Ul([
                                    html.Li([
                                        "x → the individual player's value for a stat (e.g., their Barrel% = 12.5)"
                                    ]),
                                    html.Li([
                                        "x̄ (x-bar) → the ", html.Em("mean"), " (average) of that stat across all players (e.g., league average Barrel% = 8.0)"
                                    ]),
                                    html.Li([
                                        "σ (sigma) → the ", html.Strong("standard deviation", style={'color': '#ff6b35'}), " of that stat (how spread out the numbers are)"
                                    ])
                                ], style={'marginBottom': '15px'}),
                                html.P([
                                    "Then you subtract the average from each player's value, and divide by how spread out the data is."
                                ], style={'fontStyle': 'italic'})
                            ], style={'marginTop': '20px'})
                            ]),
                            # UVS table for all players
                            uvs_table if uvs_table is not None else dbc.Alert("UVS data not available.", color="warning")
                        ])
                    ], className="mb-4")
                ], width=12)
            ])
        ], fluid=True)
        
    except Exception as e:
        import traceback
        return dbc.Alert(
            f"Error loading undervalued players data: {str(e)}",
            color="danger"
        )


if __name__ == '__main__':
    import os
    
    # Get port from environment variable (for Render/Heroku) or default to 8050
    port = int(os.environ.get('PORT', 8050))
    
    print("\n" + "="*60)
    print("Undervalued MLB Players - Statistics Table")
    print("="*60)
    print("\n📊 Table-Only Dashboard")
    print(f"\nDashboard will be available at: http://localhost:{port}")
    print("="*60 + "\n")
    
    # Use host 0.0.0.0 to make it accessible
    # Set debug=False for more stable operation
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)

