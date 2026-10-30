from common import *
import sys
W = weekly_games()
GAPMODE = "np_dm"
METRICS = ["PFS", "HTQI", "FSI"]; CONTROLS = ["PRIOR_PTS", "PRIOR_XG"]
HT_COLS = ["xG", "SHOTS", "SHOTS ON TARGET", "GAP"]

def elo_table(W):
    m = W[W.home == 1].sort_values("ts").reset_index(drop=True)
    R = {}; prev_year = None; rows = []; teams_prev = set()
    for _, r in m.iterrows():
        if r["year"] != prev_year:
            teams_now = set(m[m.year == r["year"]]["TEAM"]) | set(m[m.year == r["year"]]["OPP"])
            for t in teams_now:
                if t in R: R[t] = 0.75 * R[t] + 0.25 * 1500
                else: R[t] = 1400.0 if prev_year is not None else 1500.0
            prev_year = r["year"]
        h, a = r["TEAM"], r["OPP"]
        eh = 1 / (1 + 10 ** (-(R[h] - R[a] + 60) / 400))
        gd = r["goals_for"] - r["goals_against"]
        s = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        rows.append((r["year"], r["date"], h, a, R[h], R[a]))
        mult = 1.0 if abs(gd) <= 1 else 1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8
        delta = 20 * mult * (s - eh); R[h] += delta; R[a] -= delta
    e = pd.DataFrame(rows, columns=["year", "date", "h", "a", "eh", "ea"])
    home = e.rename(columns={"h": "TEAM", "a": "OPP", "eh": "elo", "ea": "elo_opp"})
    away = e.rename(columns={"a": "TEAM", "h": "OPP", "ea": "elo", "eh": "elo_opp"})
    return pd.concat([home, away])[["year", "date", "TEAM", "elo", "elo_opp"]]

ELO = elo_table(W)
W = W.merge(ELO, on=["year", "date", "TEAM"], how="left"); W["elo_diff"] = W["elo"] - W["elo_opp"]
assert W["elo_diff"].notna().all()
H1 = W[W.t <= 19].groupby(["year", "TEAM"])[["goals_against", "xg_against"]].sum().add_prefix("fh_").reset_index()

def make(gapmode, family):
    df = season_df(gapmode).merge(H1, on=["year", "TEAM"], how="left")
    wg = W.dropna(subset=["P_goals_for", "P_xg_for"])
    w = pd.DataFrame({"TEAM": wg["TEAM"], "year": wg["year"], "cluster": wg["cluster"],
                      "roll_goals_for": wg["P_goals_for"], "roll_xg_for": wg["P_xg_for"],
                      "roll_goals_against": wg["P_goals_against"], "roll_xg_against": wg["P_xg_against"],
                      "home": wg["home"], "elo_diff": wg["elo_diff"],
                      "next_points": wg["points"], "next_goals_for": wg["goals_for"]}).reset_index(drop=True)
    rng = np.random.default_rng(42)
    def htqi_scores(tr_rows):
        sc = StandardScaler().fit(tr_rows[HT_COLS]); pca = PCA(1).fit(sc.transform(tr_rows[HT_COLS]))
        z = pca.transform(sc.transform(df[HT_COLS]))[:, 0]; ztr = pca.transform(sc.transform(tr_rows[HT_COLS]))[:, 0]
        return -z if np.corrcoef(ztr, tr_rows["xG"])[0, 1] < 0 else z
    def prior_features(excl, tr_rows):
        d = df.copy(); d["HTQI_season"] = htqi_scores(tr_rows); out = []
        for team, g in d.groupby("TEAM"):
            g = g.sort_values("year")
            for _, r in g.iterrows():
                hist = g[(g["year"] < r["year"]) & (g["year"] != excl)]
                row = {"TEAM": team, "year": r["year"], "PFS": np.nan, "HTQI": np.nan, "FSI": np.nan, "PRIOR_PTS": np.nan, "PRIOR_XG": np.nan, "GAP_YEARS": np.nan}
                if len(hist):
                    sd = hist["GAP"].std(ddof=1) if len(hist) > 1 else 0.0
                    row.update(PFS=hist["GAP"].mean(), FSI=hist["GAP"].mean() / (1 + sd), HTQI=hist.iloc[-1]["HTQI_season"],
                               PRIOR_PTS=hist.iloc[-1]["POINTS"], PRIOR_XG=hist.iloc[-1]["xG"], GAP_YEARS=r["year"] - hist.iloc[-1]["year"])
                out.append(row)
        return pd.DataFrame(out)
    def specs(bg, bx):
        S = {"Baseline: Goals": bg, "Baseline: xG": bx}
        for tag, b in (("Goals", bg), ("xG", bx)):
            for m in METRICS: S[f"{tag} + {m}"] = b + [m]
            S[f"{tag} + prior points (control)"] = b + ["PRIOR_PTS"]
            S[f"{tag} + prior xG (control)"] = b + ["PRIOR_XG"]
            for m in METRICS: S[f"{tag} + prior points + {m}"] = b + ["PRIOR_PTS", m]
            for m in METRICS: S[f"{tag} + prior xG + {m}"] = b + ["PRIOR_XG", m]
        return S
    def fit_predict(train, test, cols, target):
        tr, te = train[cols].copy(), test[cols].copy()
        for c in cols:
            if c in METRICS or c in CONTROLS:
                mu = tr[c].mean(); tr[c] = tr[c].fillna(mu); te[c] = te[c].fillna(mu)
        return LinearRegression().fit(tr, train[target]).predict(te)
    gapinfo = []
    def run(setting, scheme):
        if setting == "FH":
            data = df.copy()
            bg, bx = ["FIRST HALF GOALS"], ["FIRST HALF XG"]
            if family == "attack+defence": bg, bx = bg + ["fh_goals_against"], bx + ["fh_xg_against"]
            targets = {"Second-half points": "SECOND HALF POINTS", "Second-half goals": "SECOND HALF GOALS"}
            data["cluster"] = np.arange(len(data))
        else:
            data = w.copy(); bg, bx = ["roll_goals_for"], ["roll_xg_for"]
            if family == "attack+defence": bg, bx = bg + ["roll_goals_against"], bx + ["roll_xg_against"]
            targets = {"Next-match points": "next_points", "Next-match goals": "next_goals_for"}
        S = specs(bg, bx)
        store = {t: {m: [] for m in S} for t in targets}
        actual = {t: [] for t in targets}; bench = {t: [] for t in targets}; clus = {t: [] for t in targets}
        for s in [y for y in sorted(df["year"].unique()) if y >= 2019]:
            if scheme == "LOSO": tm, trd, excl = data["year"] != s, df[df["year"] != s], s
            else:                tm, trd, excl = data["year"] < s, df[df["year"] < s], None
            d = data.merge(prior_features(excl, trd), on=["TEAM", "year"], how="left")
            train, test = d[tm.values], d[d["year"] == s]
            if setting == "FH" and scheme == "WF" and "Second-half points" in targets:
                gapinfo.append(dict(year=s, n=len(test), none=int(test.GAP_YEARS.isna().sum()), prev=int((test.GAP_YEARS == 1).sum()), gap=int((test.GAP_YEARS > 1).sum())))
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
                rows.append({"family": family, "setting": setting, "scheme": scheme, "target": tname, "model": m, "n_test": len(y), "oos_r2": 1 - e2.sum() / sst})
            ucl, inv = np.unique(cl, return_inverse=True); k = len(ucl)
            boots = rng.integers(0, k, size=(NBOOT, k))
            for m in S:
                if m.startswith("Baseline"): continue
                base = "Baseline: Goals" if m.startswith("Goals") else "Baseline: xG"
                dd = (y - preds[m]) ** 2 - (y - preds[base]) ** 2
                per = np.bincount(inv, weights=dd, minlength=k); cnt = np.bincount(inv, minlength=k)
                bs = per[boots].sum(1) / cnt[boots].sum(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                sstv = sst / len(y)
                idx = [i for i, r in enumerate(rows) if r["target"] == tname and r["model"] == m][-1]
                rows[idx].update(vs_baseline=base, d_mse=dd.mean(), d_mse_lo=lo, d_mse_hi=hi)
                ctrl = None
                for c in ("prior points", "prior xG"):
                    if f" + {c} + " in m: ctrl = m.split(" + ")[0] + f" + {c} (control)"
                if ctrl:
                    dd2 = (y - preds[m]) ** 2 - (y - preds[ctrl]) ** 2
                    per2 = np.bincount(inv, weights=dd2, minlength=k)
                    bs2 = per2[boots].sum(1) / cnt[boots].sum(1)
                    lo2, hi2 = np.percentile(bs2, [2.5, 97.5])
                    rows[idx].update(vs_control=ctrl, dc_mse=dd2.mean(), dc_lo=lo2, dc_hi=hi2)
        return rows
    allr = []
    for setting in ("FH", "WEEK"):
        for scheme in ("WF", "LOSO"):
            print("running", family, setting, scheme, flush=True); allr += run(setting, scheme)
    res = pd.DataFrame(allr)
    b = res[res["model"].str.startswith("Baseline")].set_index(["family", "setting", "scheme", "target", "model"])["oos_r2"]
    res["d_r2"] = res.apply(lambda r: r["oos_r2"] - b[(r["family"], r["setting"], r["scheme"], r["target"], r["vs_baseline"])] if isinstance(r.get("vs_baseline"), str) else np.nan, axis=1)
    return res, gapinfo

import json
def pairs(df, col):
    x = df[["TEAM", "year", col]].copy(); n = x.copy(); n["year"] -= 1
    return x.merge(n, on=["TEAM", "year"], suffixes=("_t", "_t1"))
def team_boot(m, col, nboot=10000, seed=7):
    teams = m["TEAM"].unique(); groups = {t: m[m.TEAM == t][[col + "_t", col + "_t1"]].values for t in teams}
    rng = np.random.default_rng(seed); rs = []
    for _ in range(nboot):
        pick = rng.choice(teams, size=len(teams), replace=True)
        a = np.vstack([groups[t] for t in pick])
        rs.append(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    return np.nanpercentile(rs, [2.5, 97.5])
df = season_df("np")
out = []
for lab, col in (("centred npG-npxG", "GAP_NP_DM"), ("uncentred npG-npxG", "GAP_NP"), ("uncentred npG-xG", "GAP_OLD")):
    m = pairs(df, col); r = np.corrcoef(m[col + "_t"], m[col + "_t1"])[0, 1]
    n, rr, flo, fhi = persistence(df, col)
    for seed in (7, 11):
        lo, hi = team_boot(m, col, 10000, seed)
        out.append(dict(gap=lab, n_pairs=n, n_teams=m.TEAM.nunique(), r=r, fisher_lo=flo, fisher_hi=fhi, seed=seed, team_boot_lo=lo, team_boot_hi=hi))
res = pd.DataFrame(out); print(res.round(3).to_string()); res.to_csv("persistence_team_cluster.csv", index=False)

d = W.dropna(subset=["P_goals_for", "P_xg_for"]).copy()
d["gd"] = d["P_goals_for"] - d["P_goals_against"]; d["xgd"] = d["P_xg_for"] - d["P_xg_against"]
d["gf"], d["xf"] = d["P_goals_for"], d["P_xg_for"]
rows = []
def boot(out, m1, m0, nboot, seed):
    y, bm = out["y"], out["bm"]
    e1, e0, sst = (y - out["pred"][m1]) ** 2, (y - out["pred"][m0]) ** 2, (y - bm) ** 2
    ucl, inv = np.unique(out["cl"], return_inverse=True); k = len(ucl)
    S1, S0, ST = (np.bincount(inv, weights=v, minlength=k) for v in (e1, e0, sst))
    idx = np.random.default_rng(seed).integers(0, k, size=(nboot, k))
    dd = (S0[idx].sum(1) - S1[idx].sum(1)) / ST[idx].sum(1)
    return (e0.sum() - e1.sum()) / sst.sum(), *np.percentile(dd, [2.5, 97.5])
for scheme in ("WF", "LOSO"):
    for target, tcol, g, x in (("points", "points", ["gd", "home"], ["xgd", "home"]), ("goals scored", "goals_for", ["gf", "home"], ["xf", "home"])):
        mods = {"Elo": ["elo_diff", "home"], "Goals": g, "xG": x, "Elo+Goals": ["elo_diff"] + g, "Elo+xG": ["elo_diff"] + x}
        out = fit_models(d, mods, tcol, scheme)
        for m in mods: rows.append(dict(scheme=scheme, target=target, comparison=f"R2 {m}", delta=r2(out, m), lo=np.nan, hi=np.nan, n=len(out["y"])))
        for m1, m0 in (("xG", "Goals"), ("Elo+xG", "Elo+Goals"), ("Elo+xG", "Elo"), ("Elo+Goals", "Elo"), ("Elo", "xG"), ("Elo", "Goals")):
            dl, lo, hi = boot(out, m1, m0, 10000, 7)
            rows.append(dict(scheme=scheme, target=target, comparison=f"{m1} minus {m0}", delta=dl, lo=lo, hi=hi, n=len(out["y"])))
res = pd.DataFrame(rows); res.to_csv("benchmark_elo.csv", index=False)
pd.set_option("display.width", 200); print(res.round(4).to_string())

frames = []
for fam in ("attack", "attack+defence"):
    r, gi = make(GAPMODE, fam); frames.append(r)
    if fam == "attack": pd.DataFrame(gi).to_csv("prior_gap_counts.csv", index=False)
ext = pd.concat(frames)
k = ext.d_r2 / (-ext.d_mse)
ext["r2_lo"] = -ext.d_mse_hi * k; ext["r2_hi"] = -ext.d_mse_lo * k
ext["rc"] = -ext.dc_mse * k; ext["rc_lo"] = -ext.dc_hi * k; ext["rc_hi"] = -ext.dc_lo * k
ext.to_csv("ablation_extended.csv", index=False)

d = W.dropna(subset=["P_goals_for", "P_xg_for"]).copy()
d["home"] = d["home"].astype(float)
def boot(out, m1, m0, nboot=10000, seed=7):
    y, bm = out["y"], out["bm"]
    e1, e0, sst = (y - out["pred"][m1]) ** 2, (y - out["pred"][m0]) ** 2, (y - bm) ** 2
    ucl, inv = np.unique(out["cl"], return_inverse=True); k = len(ucl)
    S1, S0, ST = (np.bincount(inv, weights=v, minlength=k) for v in (e1, e0, sst))
    idx = np.random.default_rng(seed).integers(0, k, size=(nboot, k))
    dd = (S0[idx].sum(1) - S1[idx].sum(1)) / ST[idx].sum(1)
    return (e0.sum() - e1.sum()) / sst.sum(), *np.percentile(dd, [2.5, 97.5])
AD_A = W[W.t <= 19].groupby(["year", "TEAM"])[["goals_for", "goals_against", "xg_for", "xg_against", "points"]].sum()
S2 = W[W.t > 19].groupby(["year", "TEAM"])[["goals_for", "points"]].sum().add_prefix("sh_")
AD_A = AD_A.join(S2).reset_index(); AD_A["cluster"] = np.arange(len(AD_A))
rows = []
for scheme in ("WF", "LOSO"):
    for target, tcol in (("points", "sh_points"), ("goals scored", "sh_goals_for")):
        out = fit_models(AD_A, {"Goals": ["goals_for", "goals_against"], "xG": ["xg_for", "xg_against"]}, tcol, scheme)
        dl, lo, hi = boot(out, "xG", "Goals")
        rows.append(dict(design="A", scheme=scheme, target=target, n=len(out["y"]), r2_goals=r2(out, "Goals"), r2_xg=r2(out, "xG"), delta=dl, lo=lo, hi=hi))
    for target, tcol in (("points", "points"), ("goals scored", "goals_for")):
        out = fit_models(d, {"Goals": ["P_goals_for", "P_goals_against", "home"], "xG": ["P_xg_for", "P_xg_against", "home"]}, tcol, scheme)
        dl, lo, hi = boot(out, "xG", "Goals")
        rows.append(dict(design="B", scheme=scheme, target=target, n=len(out["y"]), r2_goals=r2(out, "Goals"), r2_xg=r2(out, "xG"), delta=dl, lo=lo, hi=hi))
pd.DataFrame(rows).to_csv("attack_defence_comparison.csv", index=False)
