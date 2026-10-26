# Expected goals vs traditional statistics: Premier League 2016-17 to 2025-26

Code for the out-of-sample comparison of xG- and goal-based predictors (half-season and next-match horizons),
the persistence of finishing, and the PFS and HTQI ablation.

## Files
- `analysis.py` runs everything from one Excel workbook (sheets: season table, weekly team results). Outputs: persistence, bootstrap and ablation tables.
- `robustness.py` repeats the primary comparisons with 10,000 bootstrap resamples and two seeds, and adds a match-level bootstrap for Design B. Output: `robustness_results.csv`.

## Run
    pip install pandas numpy scikit-learn openpyxl
    python analysis.py     # expects the workbook as an .xlsx in the working folder or /content
    python robustness.py   # same workbook; set the XLSX environment variable to point to it

## Data
The raw data are not redistributed in this repository. They come from two public sources, and each reader should obtain them directly from the source and follow its terms of use:
- Understat (understat.com): season xG and xGA per team, per-match goals and xG per team, and shot-level records, read from the public JSON endpoints that serve its web pages.
- FBref (fbref.com): season totals of goals, non-penalty goals, penalties attempted, shots, shots on target and points per team.

To rebuild the analysis workbook, collect the ten seasons 2016-17 to 2025-26 for all 20 clubs each season and merge the two sources on team and season. The sheets expected by `analysis.py` are a season table (200 team-season rows) and a weekly team-results table (7,600 team-match rows, with goals and xG for and against and points). Thirteen team names differ between the sources and must be mapped to one name before merging. If you collect Understat data, space requests at least 1.5 seconds apart and use them for non-commercial research only. The paper's data availability statement gives the status of Understat's permission.

The paper identifies penalties by their fixed shot xG of about 0.761 and uses FBref penalty attempts for the primary correction. No script in this repository downloads data.

## License
The code is released under the MIT License (see `LICENSE`). The licence covers the code only. It does not cover Understat or FBref data.

## Citation
See `CITATION.cff`. A Zenodo DOI is minted for each tagged release.


