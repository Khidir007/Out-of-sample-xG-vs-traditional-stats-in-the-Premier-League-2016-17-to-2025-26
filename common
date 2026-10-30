import glob, os, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)

_found = sorted(set(glob.glob("/content/**/*.xlsx", recursive=True) + glob.glob("/content/*.xlsx")))
_found = [f for f in _found if "Revised_results" not in f]
if os.environ.get("XLSX"): _found = [os.environ["XLSX"]]
if not _found:
    raise FileNotFoundError("No .xlsx found in /content. Open the Files panel (folder icon, left side), "
                            "drag 'NEW DATASET.xlsx' into it, wait for the upload to finish, then run this cell again.")
_pref = [f for f in _found if "NEW" in os.path.basename(f).upper()]
XLSX = (_pref or _found)[0]
print("Using:", XLSX)
sheets = pd.ExcelFile(XLSX).sheet_names
def pick(*names):
    for n in names:
        if n in sheets: return n
    raise KeyError(f"None of {names} found. Sheets: {sheets}")

PK_XG = 0.76
NBOOT = 10000
TEST_YEARS = list(range(2019, 2026))
NAME_MAP = {"Cardiff": "Cardiff City", "Hull": "Hull City", "Ipswich": "Ipswich Town",
            "Leeds": "Leeds United", "Leicester": "Leicester City", "Luton": "Luton Town",
            "Newcastle United": "Newcastle", "Norwich": "Norwich City",
            "Nottingham Forest": "Nottingham", "Stoke": "Stoke City", "Swansea": "Swansea City",
            "West Bromwich Albion": "West Brom", "Wolverhampton Wanderers": "Wolves",
            "Huddersfield Town": "Huddersfield"}

RAW = pd.read_excel(XLSX, sheet_name=pick("NEW DATASET", "NEW_DATASET", "Sheet1")).dropna(how="all")
RAW["SEASON"] = RAW["SEASON"].ffill()
RAW["TEAM"] = RAW["TEAM"].replace(NAME_MAP)
RAW = RAW.reset_index(drop=True)

def season_df(gapmode="np", pk_xg=PK_XG):
    df = RAW.copy()
    df["pk_att"] = pd.to_numeric(df["PENALTIES ATTEMPTED"], errors="coerce").fillna(0)
    df["npxG"] = df["xG"] - pk_xg * df["pk_att"]
    df["GAP_OLD"] = df["NON PENALTY GOALS"] - df["xG"]
    df["GAP_NP"] = df["NON PENALTY GOALS"] - df["npxG"]
    df["GAP_NP_DM"] = df["GAP_NP"] - df.groupby("SEASON")["GAP_NP"].transform("mean")
    df["GAP"] = df[{"old": "GAP_OLD", "np": "GAP_NP", "np_dm": "GAP_NP_DM"}[gapmode]]
    df["year"] = df["SEASON"].astype(str).str[:4].astype(int)
    return df

def weekly_games():
    w = pd.read_excel(XLSX, sheet_name=pick("WEEKLY TEAM RESULTS", "Weekly_team_results")).dropna(how="all")
    w = w.rename(columns={"SEASON": "year", "TEAM": "team", "OPPONENT": "opponent", "MATCHWEEK": "matchweek",
                          "DATE": "date", "GOALS FOR": "goals_for", "GOALS AGAINST": "goals_against",
                          "XG FOR": "xg_for", "XG AGAINST": "xg_against", "POINTS": "points"}).reset_index(drop=True)
    assert len(w) % 2 == 0
    a, b = w.iloc[::2], w.iloc[1::2]
    assert (a["team"].values == b["opponent"].values).all() and (a["date"].values == b["date"].values).all(), "pair order broken"
    w["home"] = (np.arange(len(w)) % 2 == 0).astype(int)
    w["TEAM"] = w["team"].replace(NAME_MAP); w["OPP"] = w["opponent"].replace(NAME_MAP)
    w["ts"] = pd.to_datetime(w["date"])
    w = w.sort_values(["year", "team", "ts"]).reset_index(drop=True)
    w["t"] = w.groupby(["year", "team"]).cumcount() + 1
    g = w.groupby(["year", "team"])
    cols = ["goals_for", "goals_against", "xg_for", "xg_against", "points"]
    for c in cols:
        w["P_" + c] = g[c].transform(lambda s: s.rolling(4, min_periods=4).mean().shift(1))
    for c in cols:
        w["Plag_" + c] = g["P_" + c].shift(1)
    oc = ["goals_for", "goals_against", "xg_for", "xg_against"]
    op = w[["year", "team", "date"] + ["P_" + c for c in oc]].rename(
        columns={"team": "opponent", **{"P_" + c: "oppP_" + c for c in oc}})
    w = w.merge(op, on=["year", "opponent", "date"], how="left")
    w["cluster"] = w["TEAM"] + "_" + w["year"].astype(str)
    return w

def fit_models(data, models, target, scheme, test_years=TEST_YEARS):
    ys, bms, cls, preds = [], [], [], {m: [] for m in models}
    for s in test_years:
        tr = data[data["year"] < s] if scheme == "WF" else data[data["year"] != s]
        te = data[data["year"] == s]
        ys.append(te[target].values); bms.append(np.full(len(te), tr[target].mean())); cls.append(te["cluster"].values)
        for m, cols in models.items():
            preds[m].append(LinearRegression().fit(tr[cols], tr[target]).predict(te[cols]))
    out = {"y": np.concatenate(ys), "bm": np.concatenate(bms), "cl": np.concatenate(cls)}
    out["pred"] = {m: np.concatenate(v) for m, v in preds.items()}
    return out

def r2(out, m):
    return 1 - ((out["y"] - out["pred"][m]) ** 2).sum() / ((out["y"] - out["bm"]) ** 2).sum()

def boot_delta(out, m1, m0, nboot=NBOOT, seed=7):
    y, bm, cl = out["y"], out["bm"], out["cl"]
    e1, e0, sst = (y - out["pred"][m1]) ** 2, (y - out["pred"][m0]) ** 2, (y - bm) ** 2
    ucl, inv = np.unique(cl, return_inverse=True); k = len(ucl)
    S1, S0, ST = (np.bincount(inv, weights=v, minlength=k) for v in (e1, e0, sst))
    idx = np.random.default_rng(seed).integers(0, k, size=(nboot, k))
    d = (S0[idx].sum(1) - S1[idx].sum(1)) / ST[idx].sum(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return (e0.sum() - e1.sum()) / sst.sum(), lo, hi

def persistence(df, col):
    x = df[["TEAM", "year", col]].copy(); n = x.copy(); n["year"] -= 1
    m = x.merge(n, on=["TEAM", "year"], suffixes=("_t", "_t1"))
    r = np.corrcoef(m[col + "_t"], m[col + "_t1"])[0, 1]
    z = np.arctanh(r); se = 1 / np.sqrt(len(m) - 3)
    return len(m), r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

