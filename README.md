Expected goals vs traditional statistics: Premier League 2016-17 to 2025-26
Code for the out-of-sample comparison of xG- and goal-based predictors (half-season and next-match horizons), the persistence of finishing, the ablation of PFS, HTQI and FSI against controls, an Elo benchmark, and the score-state analysis of shot rates and shot quality.
Files
`common.py` shared functions: data loading, name harmonisation, validation and bootstrap helpers.
`analysis.py` runs the main analysis from one Excel workbook (sheets: season table, weekly team results). Outputs: persistence, bootstrap and ablation tables.
`robustness.py` repeats the primary comparisons with 10,000 bootstrap resamples and two seeds, adds a match-level bootstrap for Design B. Output: `robustness_results.csv`.
`extended.py` runs the extended ablation (goals baseline, attack-plus-defence inputs, previous-season points and xG controls, FSI), the Elo benchmark and the team-cluster bootstrap interval for persistence. Outputs: `ablation_extended.csv`, `attack_defence_comparison.csv`, `benchmark_elo.csv`, `persistence_team_cluster.csv`, `prior_gap_counts.csv`.
`shots.py` rebuilds the score state from the shot records, computes the exposure in each state, fits the Poisson model with an exposure offset and correlates SHDCI with the shot-level difference. It also runs the shot-based penalty check. Outputs: `penalty_proxy.csv`, `score_state_totals.csv`, `score_state_model.csv`, `shdci_correlation.csv`. It reads the shot and match files named by the `SHOTS` and `MATCHES` environment variables.
Run
    pip install -r requirements.txt
    python analysis.py
    python robustness.py
    python extended.py
    SHOTS=score_state_shots.json MATCHES=match_teams.json python shots.py
Set the `XLSX` environment variable to point to the workbook if it is not in the working folder.
Data
The raw data are not redistributed in this repository. Obtain them from the providers:
Understat (understat.com): season xG and xGA per team, per-match goals and xG per team, and shot-level records.
FBref (fbref.com): season totals of goals, non-penalty goals, penalties attempted, shots, shots on target and points per team.
Readers should obtain the data from the provider. The sheets expected by `analysis.py` are a season table (200 team-season rows) and a weekly team-results table (7,600 team-match rows, with goals and xG for and against and points). Thirteen team names differ between the sources and must be mapped to one name before merging. The shot file holds one record per shot with the fields minute, shot_side, xG, shooter_state, defender_state, match_id and season, and the match file holds match_id, season, home_team, away_team and datetime.
The paper identifies penalties by their fixed shot xG of about 0.761 and uses FBref penalty attempts for the primary correction. No script in this repository downloads data.
License
The code is released under the MIT License (see `LICENSE`). The licence covers the code only. It does not cover Understat or FBref data.
Citation
See `CITATION.cff`. A Zenodo DOI is minted for each tagged release.

