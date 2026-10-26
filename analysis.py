import glob, os, warnings
import numpy as np
import pandas as pd
from IPython.display import display
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

rows = []
for pk in (0.76, 0.79):
    df = season_df("np", pk)
    for lab, col in (("npG - xG (draft)", "GAP_OLD"), (f"npG - npxG (pk xG {pk})", "GAP_NP"),
                     (f"npG - npxG minus season mean ({pk})", "GAP_NP_DM")):
        n, r, lo, hi = persistence(df, col)
        pfs = df.groupby("TEAM")[col].mean()
        rows.append(dict(pk_xg=pk, definition=lab, mean_gap=df[col].mean(), n_pairs=n, r=r, r_lo=lo, r_hi=hi,
                         pfs_min=pfs.min(), pfs_max=pfs.max()))
persist = pd.DataFrame(rows)
print("\n=== Finishing gap and persistence ===")
display(persist.round(3))
df = season_df("np")
print("Mean gap by season:"); display(df.groupby("SEASON")[["GAP_OLD", "GAP_NP"]].mean().round(2))
pk = df[["TEAM", "year", "pk_att"]]; n_ = pk.copy(); n_["year"] -= 1
m_ = pk.merge(n_, on=["TEAM", "year"], suffixes=("_t", "_t1"))
print("Penalty attempts persistence r:", round(np.corrcoef(m_.pk_att_t, m_.pk_att_t1)[0, 1], 3))
persist.to_csv("persistence.csv", index=False)

W = weekly_games()
brow = []
def add(design, scheme, target, family, out):
    d, lo, hi = boot_delta(out, "xG", "Goals")
    brow.append(dict(design=design, scheme=scheme, target=target, family=family, n_test=len(out["y"]),
                     r2_goals=r2(out, "Goals"), r2_xg=r2(out, "xG"), delta_r2=d, ci_lo=lo, ci_hi=hi,
                     includes_zero=(lo <= 0 <= hi)))
for dname, pre in (("B aligned (window ends at week n)", "P_"), ("B as built (window ends at week n-1)", "Plag_")):
    d = W.dropna(subset=[pre + "goals_for", pre + "xg_for", "oppP_xg_for", "oppP_xg_against"]).copy()
    d["gd"] = d[pre + "goals_for"] - d[pre + "goals_against"]; d["xgd"] = d[pre + "xg_for"] - d[pre + "xg_against"]
    d["gf"], d["ga"], d["xf"], d["xa"] = (d[pre + c] for c in ("goals_for", "goals_against", "xg_for", "xg_against"))
    d["next_gd"] = d["goals_for"] - d["goals_against"]
    ctrl = ["home", "oppP_xg_for", "oppP_xg_against"]
    fams = {"attack only (as in the draft)": (["gf"], ["xf"]),
            "goal difference vs xG difference": (["gd"], ["xgd"]),
            "attack + defence": (["gf", "ga"], ["xf", "xa"]),
            "attack only + venue + opponent": (["gf"] + ctrl, ["xf"] + ctrl),
            "goal difference vs xG difference + venue + opponent": (["gd"] + ctrl, ["xgd"] + ctrl)}
    for scheme in ("WF", "LOSO"):
        for target, tcol in (("points", "points"), ("goals scored", "goals_for"), ("goal difference", "next_gd")):
            for fam, (g, x) in fams.items():
                if target == "goals scored" and fam.startswith("goal difference"): continue
                if target == "goal difference" and fam.startswith("attack only"): continue
                add(dname, scheme, target, fam, fit_models(d, {"Goals": g, "xG": x}, tcol, scheme))
W["half"] = np.where(W["t"] <= 19, 1, 2)
H = W[W.half == 1].groupby(["year", "TEAM"])[["goals_for", "goals_against", "xg_for", "xg_against", "points"]].sum()
S2 = W[W.half == 2].groupby(["year", "TEAM"])[["goals_for", "goals_against", "points"]].sum().add_prefix("sh_")
A = H.join(S2).reset_index(); A["cluster"] = np.arange(len(A))
A["gd"] = A.goals_for - A.goals_against; A["xgd"] = A.xg_for - A.xg_against; A["sh_gd"] = A.sh_goals_for - A.sh_goals_against
famsA = {"attack only (as in the draft)": (["goals_for"], ["xg_for"]),
         "goal difference vs xG difference": (["gd"], ["xgd"]),
         "attack + defence": (["goals_for", "goals_against"], ["xg_for", "xg_against"])}
for scheme in ("WF", "LOSO"):
    for target, tcol in (("points", "sh_points"), ("goals scored", "sh_goals_for"), ("goal difference", "sh_gd")):
        for fam, (g, x) in famsA.items():
            if target == "goals scored" and fam.startswith("goal difference"): continue
            if target == "goal difference" and fam.startswith("attack only"): continue
            add("A Understat split (matchweek 19/20)", scheme, target, fam, fit_models(A, {"Goals": g, "xG": x}, tcol, scheme))
D = season_df("np"); D["cluster"] = np.arange(len(D))
for scheme in ("WF", "LOSO"):
    for target, tcol in (("points", "SECOND HALF POINTS"), ("goals scored", "SECOND HALF GOALS")):
        add("A dataset columns (draft)", scheme, target, "attack only (as in the draft)",
            fit_models(D, {"Goals": ["FIRST HALF GOALS"], "xG": ["FIRST HALF XG"]}, tcol, scheme))
boot = pd.DataFrame(brow)
boot.to_csv("bootstrap_results.csv", index=False)
print("\n=== xG minus Goals: change in OOS R2, paired cluster bootstrap 95% interval ===")
display(boot.sort_values(["design", "scheme", "target", "family"]).round(3).reset_index(drop=True))

HT_COLS = ["xG", "SHOTS", "SHOTS ON TARGET", "GAP"]
METRICS = ["PFS", "HTQI"]; CONTROLS = ["PRIOR_PTS", "PRIOR_XG"]

def make_ablation(gapmode, pre="P_"):
    df = season_df(gapmode)
    wg = W.dropna(subset=[pre + "goals_for", pre + "xg_for"])
    w = pd.DataFrame({"TEAM": wg["TEAM"], "year": wg["year"], "cluster": wg["cluster"],
                      "roll_goals_for": wg[pre + "goals_for"], "roll_xg_for": wg[pre + "xg_for"],
                      "next_points": wg["points"], "next_goals_for": wg["goals_for"]}).reset_index(drop=True)
    assert set(w["TEAM"]) <= set(df["TEAM"]), set(w["TEAM"]) - set(df["TEAM"])
    rng = np.random.default_rng(42)

    def htqi_scores(train_rows):
        sc = StandardScaler().fit(train_rows[HT_COLS]); pca = PCA(1).fit(sc.transform(train_rows[HT_COLS]))
        z = pca.transform(sc.transform(df[HT_COLS]))[:, 0]
        ztr = pca.transform(sc.transform(train_rows[HT_COLS]))[:, 0]
        return -z if np.corrcoef(ztr, train_rows["xG"])[0, 1] < 0 else z

    def prior_features(exclude_year, train_rows):
        d = df.copy(); d["HTQI_season"] = htqi_scores(train_rows); out = []
        for team, g in d.groupby("TEAM"):
            g = g.sort_values("year")
            for _, r in g.iterrows():
                hist = g[(g["year"] < r["year"]) & (g["year"] != exclude_year)]
                row = {"TEAM": team, "year": r["year"], "PFS": np.nan, "HTQI": np.nan, "PRIOR_PTS": np.nan, "PRIOR_XG": np.nan}
                if len(hist):
                    row.update(PFS=hist["GAP"].mean(),
                               HTQI=hist.iloc[-1]["HTQI_season"], PRIOR_PTS=hist.iloc[-1]["POINTS"],
                               PRIOR_XG=hist.iloc[-1]["xG"])
                out.append(row)
        return pd.DataFrame(out)

    def specs(bg, bx):
        S = {"Baseline: Goals": [bg], "Baseline: xG": [bx]}
        for tag, b in (("Goals", bg), ("xG", bx)):
            for m in METRICS: S[f"{tag} + {m}"] = [b, m]
            S[f"{tag} + prior points (control)"] = [b, "PRIOR_PTS"]
            for m in METRICS: S[f"{tag} + prior points + {m}"] = [b, "PRIOR_PTS", m]
        return S

    def fit_predict(train, test, cols, target):
        tr, te = train[cols].copy(), test[cols].copy()
        for c in cols:
            if c in METRICS or c in CONTROLS:
                mu = tr[c].mean(); tr[c] = tr[c].fillna(mu); te[c] = te[c].fillna(mu)
        return LinearRegression().fit(tr, train[target]).predict(te)

    def run(setting, scheme):
        if setting == "FH":
            data = df.copy(); bg, bx = "FIRST HALF GOALS", "FIRST HALF XG"
            targets = {"Second-half points": "SECOND HALF POINTS", "Second-half goals": "SECOND HALF GOALS"}
            data["cluster"] = np.arange(len(data))
        else:
            data = w.copy(); bg, bx = "roll_goals_for", "roll_xg_for"
            targets = {"Next-week points": "next_points", "Next-week goals": "next_goals_for"}
        S = specs(bg, bx)
        store = {t: {m: [] for m in S} for t in targets}
        actual = {t: [] for t in targets}; bench = {t: [] for t in targets}; clus = {t: [] for t in targets}
        for s in [y for y in sorted(df["year"].unique()) if y >= 2019]:
            if scheme == "LOSO": tm, trd, excl = data["year"] != s, df[df["year"] != s], s
            else:                tm, trd, excl = data["year"] < s, df[df["year"] < s], None
            d = data.merge(prior_features(excl, trd), on=["TEAM", "year"], how="left")
            train, test = d[tm.values], d[d["year"] == s]
            for tname, tcol in targets.items():
                actual[tname].append(test[tcol].values); bench[tname].append(np.full(len(test), train[tcol].mean()))
                clus[tname].append(test["cluster"].values)
                for mname, cols in S.items(): store[tname][mname].append(fit_predict(train, test, cols, tcol))
        rows = []
        for tname in targets:
            y = np.concatenate(actual[tname]); cl = np.concatenate(clus[tname])
            sst = ((y - np.concatenate(bench[tname])) ** 2).sum()
            preds = {m: np.concatenate(store[tname][m]) for m in S}
            for m in S:
                e2 = (y - preds[m]) ** 2
                rows.append({"setting": setting, "scheme": scheme, "target": tname, "model": m, "n_test": len(y),
                             "oos_r2": 1 - e2.sum() / sst, "oos_rmse": np.sqrt(e2.mean())})
            ucl, inv = np.unique(cl, return_inverse=True); k = len(ucl)
            boots = rng.integers(0, k, size=(NBOOT, k))
            for m in S:
                if m.startswith("Baseline") or m.startswith("Metrics"): continue
                base = "Baseline: Goals" if m.startswith("Goals") else "Baseline: xG"
                dd = (y - preds[m]) ** 2 - (y - preds[base]) ** 2
                per = np.bincount(inv, weights=dd, minlength=k); cnt = np.bincount(inv, minlength=k)
                bs = per[boots].sum(1) / cnt[boots].sum(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                idx = [i for i, r in enumerate(rows) if r["target"] == tname and r["model"] == m][-1]
                rows[idx].update(vs_baseline=base, d_mse=dd.mean(), d_mse_lo=lo, d_mse_hi=hi)
        return rows

    all_rows = []
    for setting in ("FH", "WEEK"):
        for scheme in ("LOSO", "WF"):
            print("  running", gapmode, setting, scheme, flush=True); all_rows += run(setting, scheme)
    res = pd.DataFrame(all_rows)
    b = res[res["model"].str.startswith("Baseline")].set_index(["setting", "scheme", "target", "model"])["oos_r2"]
    res["d_r2"] = res.apply(lambda r: r["oos_r2"] - b[(r["setting"], r["scheme"], r["target"], r["vs_baseline"])]
                            if isinstance(r.get("vs_baseline"), str) else np.nan, axis=1)
    return res

print("\n=== ABLATION (gap = npG - npxG, aligned window, training-mean benchmark) ===")
res_np = make_ablation("np"); res_np.to_csv("ablation_np_aligned.csv", index=False)
print("\n=== ABLATION (gap = npG - npxG minus season mean) ===")
res_dm = make_ablation("np_dm"); res_dm.to_csv("ablation_npdm_aligned.csv", index=False)
with pd.ExcelWriter("Revised_results.xlsx") as xw:
    persist.to_excel(xw, sheet_name="Persistence", index=False)
    boot.to_excel(xw, sheet_name="Bootstrap xG vs Goals", index=False)
    res_np.to_excel(xw, sheet_name="Ablation gap npG-npxG", index=False)
    res_dm.to_excel(xw, sheet_name="Ablation gap minus mean", index=False)

SETTING, SCHEME, TARGET = "FH", "WF", "Second-half points"
view = res_dm[(res_dm.setting == SETTING) & (res_dm.scheme == SCHEME) & (res_dm.target == TARGET)]
display(view[["model", "n_test", "oos_r2", "oos_rmse", "vs_baseline", "d_r2", "d_mse", "d_mse_lo", "d_mse_hi"]].round(4).reset_index(drop=True))
try:
    from google.colab import files
    files.download("Revised_results.xlsx")
except Exception as e:
    print("Saved Revised_results.xlsx in the Files panel.", e)
