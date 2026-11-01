from common import *
W = weekly_games()
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
W = W.merge(ELO, on=["year", "date", "TEAM"], how="left")
assert W["elo"].notna().all()
W = W.sort_values(["year", "TEAM", "t"]).reset_index(drop=True)
W["xgd_m"] = W["xg_for"] - W["xg_against"]; W["gd_m"] = W["goals_for"] - W["goals_against"]
g = W.groupby(["year", "TEAM"])
for c in ("xg_for", "xg_against", "xgd_m", "goals_for", "goals_against", "gd_m"):
    W["S_" + c] = g[c].transform(lambda s: s.expanding().mean().shift(1))
op = W[["year", "TEAM", "date", "S_xg_for", "S_xg_against"]].rename(columns={"TEAM": "OPP", "S_xg_for": "oppS_xg_for", "S_xg_against": "oppS_xg_against"})
W = W.merge(op, on=["year", "OPP", "date"], how="left")
d = W.dropna(subset=["P_goals_for", "P_xg_for", "oppP_xg_for", "oppP_xg_against", "S_xg_for", "oppS_xg_for"]).copy()
d["home"] = d["home"].astype(float)
d["gd4"] = d["P_goals_for"] - d["P_goals_against"]; d["xgd4"] = d["P_xg_for"] - d["P_xg_against"]
d["gf4"], d["xf4"] = d["P_goals_for"], d["P_xg_for"]
c4 = ["home", "oppP_xg_for", "oppP_xg_against"]; cS = ["home", "oppS_xg_for", "oppS_xg_against"]
def boot(out, m1, m0, nboot=10000, seed=7):
    y, bm = out["y"], out["bm"]
    e1, e0, sst = (y - out["pred"][m1]) ** 2, (y - out["pred"][m0]) ** 2, (y - bm) ** 2
    ucl, inv = np.unique(out["cl"], return_inverse=True); k = len(ucl)
    S1, S0, ST = (np.bincount(inv, weights=v, minlength=k) for v in (e1, e0, sst))
    idx = np.random.default_rng(seed).integers(0, k, size=(nboot, k))
    dd = (S0[idx].sum(1) - S1[idx].sum(1)) / ST[idx].sum(1)
    return (e0.sum() - e1.sum()) / sst.sum(), *np.percentile(dd, [2.5, 97.5])
rows = []
for scheme in ("WF", "LOSO"):
    for target, tcol, gb, xb, gs, xs in (("points", "points", ["gd4"], ["xgd4"], ["S_gd_m"], ["S_xgd_m"]), ("goals scored", "goals_for", ["gf4"], ["xf4"], ["S_goals_for"], ["S_xg_for"])):
        mods = {"Goals, 4 matches": gb + c4, "xG, 4 matches": xb + c4, "Goals, season to date": gs + cS, "xG, season to date": xs + cS,
                "Elo": ["elo", "elo_opp", "home"], "Elo + xG, 4 matches": ["elo", "elo_opp"] + xb + c4, "Elo + xG, season to date": ["elo", "elo_opp"] + xs + cS,
                "Elo + goals, season to date": ["elo", "elo_opp"] + gs + cS}
        out = fit_models(d, mods, tcol, scheme)
        for m in mods: rows.append(dict(scheme=scheme, target=target, comparison="R2 " + m, delta=r2(out, m), lo=np.nan, hi=np.nan, n=len(out["y"])))
        for m1, m0 in (("xG, 4 matches", "Goals, 4 matches"), ("xG, season to date", "Goals, season to date"), ("xG, season to date", "xG, 4 matches"),
                       ("Elo", "xG, 4 matches"), ("Elo", "xG, season to date"), ("Elo + xG, 4 matches", "Elo"), ("Elo + xG, season to date", "Elo"),
                       ("Elo + xG, season to date", "Elo + goals, season to date")):
            dl, lo, hi = boot(out, m1, m0)
            rows.append(dict(scheme=scheme, target=target, comparison=m1 + " minus " + m0, delta=dl, lo=lo, hi=hi, n=len(out["y"])))
res = pd.DataFrame(rows); res.to_csv("benchmark_fair.csv", index=False)
pd.set_option("display.width", 220); print(res.round(4).to_string())
