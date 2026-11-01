import sys
from common import *
from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson
exec(open("fair_elo.py").read().split("\nrows = []")[0].split("from common import *")[1])
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005
M = W[W.home == 1].sort_values("ts").reset_index(drop=True)
teams = sorted(set(M.TEAM) | set(M.OPP)); ix = {t: i for i, t in enumerate(teams)}; nt = len(teams)
M["blk"] = M.year.astype(str) + "_" + pd.to_datetime(M.date).dt.isocalendar().week.astype(str).str.zfill(2)
def design(df):
    n = len(df); X = np.zeros((2 * n, 2 * nt + 1))
    for k, (h, a) in enumerate(zip(df.TEAM, df.OPP)):
        X[2 * k, ix[h]] = 1; X[2 * k, nt + ix[a]] = 1; X[2 * k, -1] = 1
        X[2 * k + 1, ix[a]] = 1; X[2 * k + 1, nt + ix[h]] = 1
    return X
def points_from(lh, la):
    g = np.arange(0, 11); ph, pa = poisson.pmf(g, lh), poisson.pmf(g, la); jt = np.outer(ph, pa)
    return 3 * np.tril(jt, -1).sum() + jt.diagonal().sum(), 3 * np.triu(jt, 1).sum() + jt.diagonal().sum(), np.tril(jt, -1).sum(), np.triu(jt, 1).sum()
out = []
for resp, (cf, ca) in (("goals", ("goals_for", "goals_against")), ("xg", ("xg_for", "xg_against"))):
    for blk, grp in M.groupby("blk", sort=True):
        yr = grp.year.iloc[0]; first = grp.ts.min()
        tr = M[(M.ts < first) & (M.year >= yr - 1)].copy()
        if len(tr) < 100: continue
        wt = np.where(tr.year == yr, 1.0, 0.5); wt2 = np.repeat(wt, 2)
        y = np.empty(2 * len(tr)); y[0::2] = tr[cf].values; y[1::2] = tr[ca].values
        mdl = PoissonRegressor(alpha=ALPHA, max_iter=300).fit(design(tr), y, sample_weight=wt2)
        lam = mdl.predict(design(grp))
        for k, (_, r) in enumerate(grp.iterrows()):
            ph, pa, _, _ = points_from(lam[2 * k], lam[2 * k + 1])
            out.append((resp, r.year, r.date, r.TEAM, ph, lam[2 * k]))
            out.append((resp, r.year, r.date, r.OPP, pa, lam[2 * k + 1]))
P = pd.DataFrame(out, columns=["resp", "year", "date", "TEAM", "ep", "eg"])
for resp in ("goals", "xg"):
    q = P[P.resp == resp].drop(columns="resp").rename(columns={"ep": "pois_ep_" + resp, "eg": "pois_eg_" + resp})
    d = d.merge(q, on=["year", "date", "TEAM"], how="left")
dd = d.dropna(subset=["pois_ep_goals", "pois_ep_xg"]).copy()
print("rows", len(dd), "of", len(d))
rows = []
for scheme in ("WF",):
    for target, tcol, pg, px, xs, gs, xb, gb in (("points", "points", "pois_ep_goals", "pois_ep_xg", ["S_xgd_m"], ["S_gd_m"], ["xgd4"], ["gd4"]), ("goals scored", "goals_for", "pois_eg_goals", "pois_eg_xg", ["S_xg_for"], ["S_goals_for"], ["xf4"], ["gf4"])):
        mods = {"Poisson, goals": ["home", pg], "Poisson, xG": ["home", px], "xG, season to date": xs + cS, "xG, 4 matches": xb + c4, "Elo": ["elo", "elo_opp", "home"], "Poisson xG + Elo": ["home", px, "elo", "elo_opp"]}
        out_ = fit_models(dd, mods, tcol, scheme)
        for m in mods: rows.append(dict(target=target, comparison="R2 " + m, delta=r2(out_, m), lo=np.nan, hi=np.nan, n=len(out_["y"])))
        for m1, m0 in (("Poisson, xG", "Poisson, goals"), ("Poisson, xG", "xG, season to date"), ("Poisson, xG", "xG, 4 matches"), ("Elo", "Poisson, xG"), ("Poisson xG + Elo", "Elo")):
            dl, lo, hi = boot(out_, m1, m0); rows.append(dict(target=target, comparison=m1 + " minus " + m0, delta=dl, lo=lo, hi=hi, n=len(out_["y"])))
res = pd.DataFrame(rows); res.to_csv("benchmark_poisson.csv", index=False); print(res.round(4).to_string())
