# Expected goals vs traditional statistics: Premier League 2016-17 to 2025-26

Code for the out-of-sample comparison of xG- and goal-based predictors (half-season and next-match horizons),
the persistence of finishing, and the PFS and HTQI ablation.

## Files
- `code/analysis.py` runs everything from one Excel workbook (sheets: season table, weekly team results). Outputs: persistence, bootstrap and ablation tables.
- `code/fetch_penalty_split.py` collects team-season penalty/non-penalty goals and xG from Understat's public JSON endpoints (requests spaced 1.5 s apart, non-commercial research use, no player names stored).

## Run
    pip install pandas numpy scikit-learn openpyxl
    python code/analysis.py     # expects the workbook as an .xlsx in the working folder or /content

## Data
The raw data are not redistributed in this repository. They come from two public sources, and each reader should obtain them directly from the source and follow its terms of use:
- Understat (understat.com): season xG and xGA per team, per-match goals and xG per team, and shot-level records, read from the public JSON endpoints that serve its web pages.
- FBref (fbref.com): season totals of goals, non-penalty goals, penalties attempted, shots, shots on target and points per team.

To rebuild the analysis workbook, collect the ten seasons 2016-17 to 2025-26 for all 20 clubs each season and merge the two sources on team and season. The sheets expected by `code/analysis.py` are a season table (200 team-season rows) and a weekly team-results table (7,600 team-match rows, with goals and xG for and against and points). Thirteen team names differ between the sources and must be mapped to one name before merging. If you collect Understat data, space requests at least 1.5 seconds apart and use them for non-commercial research only. The paper's data availability statement gives the status of Understat's permission.

`code/fetch_penalty_split.py` was written for a possible penalty split from Understat and was not used for any result in the paper. The paper identifies penalties by their fixed shot xG of about 0.761 and uses FBref penalty attempts for the primary correction.

## License
The code is released under the MIT License (see `LICENSE`). The licence covers the code only. It does not cover Understat or FBref data.

## Citation
See `CITATION.cff`. A Zenodo DOI is minted for each tagged release.

