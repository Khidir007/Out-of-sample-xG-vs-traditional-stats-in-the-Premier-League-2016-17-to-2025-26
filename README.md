# Expected goals vs traditional statistics: Premier League 2016-17 to 2025-26

Code for the out-of-sample comparison of xG- and goal-based predictors (half-season and next-match horizons),
the persistence of finishing, and the composite-metric ablation.

## Files
- `code/analysis.py` runs everything from one Excel workbook (sheets: season table, weekly team results). Outputs: persistence, bootstrap and ablation tables.
- `code/fetch_penalty_split.py` collects team-season penalty/non-penalty goals and xG from Understat's public JSON endpoints (requests spaced 1.5 s apart, non-commercial research use, no player names stored).

## Run
    pip install pandas numpy scikit-learn openpyxl
    python code/analysis.py     # expects the workbook as an .xlsx in the working folder or /content

## Data
The raw and match-level data are not redistributed here. The data availability statement of the paper gives the status of permission from Understat.

## Citation
See `CITATION.cff`. A Zenodo DOI is minted for each tagged release.
