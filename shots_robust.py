from common import *
import json, scipy.sparse as sp
from scipy import stats
SHOTS = os.environ.get("SHOTS", "score_state_shots.json")
MATCHES = os.environ.get("MATCHES", "match_teams.json")
sh = pd.DataFrame(json.load(open(SHOTS)))
mt = pd.DataFrame(json.load(open(MATCHES)))
sh["ord"] = np.arange(len(sh))
teams = sh.merge(mt[["match_id", "home_team", "away_team"]], on="match_id")
teams["team"] = np.where(teams.shot_side == "h", teams.home_team, teams.away_team)
teams["team"] = teams["team"].replace(NAME_MAP)
df = season_df("np")
pk_rows = []
for lab, lo, hi, right in (("window 0.75 to 0.775", 0.75, 0.775, "right"), ("window 0.760 to 0.762", 0.760, 0.762, "both")):
    m = teams.xG.between(lo, hi, inclusive=right)
    c = teams[m].groupby(["team", "season"]).size().rename("pk_proxy").reset_index().rename(columns={"team": "TEAM", "season": "year"})
    d = df.merge(c, on=["TEAM", "year"], how="left"); d["pk_proxy"] = d.pk_proxy.fillna(0)
    d["g2"] = d["NON PENALTY GOALS"] - (d.xG - 0.76 * d.pk_proxy); d["g2c"] = d.g2 - d.groupby("year")["g2"].transform("mean")
    n, r, lo_, hi_ = persistence(d, "g2c")
    pk_rows.append(dict(window=lab, shots=int(m.sum()), fbref=int(d.pk_att.sum()), team_season_corr=np.corrcoef(d.pk_proxy, d.pk_att)[0, 1],
                        exact_matches=int((d.pk_proxy == d.pk_att).sum()), persistence_r=r, fisher_lo=lo_, fisher_hi=hi_))
pd.DataFrame(pk_rows).to_csv("penalty_proxy.csv", index=False)
print(pd.DataFrame(pk_rows).round(3).to_string())

sh["pen"] = sh.xG.between(0.7605, 0.7615)
sh["home_state"] = np.where(sh.shot_side == "h", sh.shooter_state, sh.defender_state)
SIGN = {"leading": 1, "level": 0, "trailing": -1}
BINEDGE = [0, 15, 30, 45, 60, 75, 90]
segs = []; recon = []
for mid, g in sh.groupby("match_id", sort=False):
    g = g.sort_values("ord")
    mins = g.minute.values.astype(float); side = g.shot_side.values; hs = g.home_state.map(SIGN).values
    T = max(94.0, mins[-1] + 1)
    trans = []; prev = 0; start = 0
    for i in range(len(g)):
        if hs[i] != prev:
            want = "h" if hs[i] > prev else "a"
            cand = [j for j in range(start, i) if side[j] == want]
            if cand: t = mins[cand[-1]] + 0.5
            else: t = ((mins[i - 1] if i > 0 else 0.0) + mins[i]) / 2
            trans.append((t, hs[i])); prev = hs[i]; start = i
    cuts = sorted(set([0.0, T] + BINEDGE[1:] + [t for t, _ in trans if 0 < t < T]))
    for a, b in zip(cuts[:-1], cuts[1:]):
        mid_t = (a + b) / 2; st = 0
        for t, s in trans:
            if t <= mid_t: st = s
        segs.append((mid, a, b, st, min(int(a // 15), 6)))
    recon.append((mid, len(trans), hs[-1]))
S = pd.DataFrame(segs, columns=["match_id", "a", "b", "hs", "bin"]); S["e"] = S.b - S.a
E_ = S.groupby(["match_id", "hs", "bin"], as_index=False).e.sum()
INV = {1: "leading", 0: "level", -1: "trailing"}
rows = []
for side, sgn in (("h", 1), ("a", -1)):
    t = E_.copy(); t["side"] = side; t["state"] = (t.hs * sgn).map(INV); rows.append(t[["match_id", "side", "state", "bin", "e"]])
X = pd.concat(rows, ignore_index=True)
sh["bin"] = np.minimum((sh.minute // 15).astype(int), 6)
sn = sh[~sh.pen]
own = sn.groupby(["match_id", "shot_side", "shooter_state", "bin"]).agg(shots=("xG", "size"), xg=("xG", "sum")).reset_index().rename(columns={"shot_side": "side", "shooter_state": "state"})
X = X.merge(own, on=["match_id", "side", "state", "bin"], how="outer")
X = X[X.e.notna()].copy(); X[["shots", "xg"]] = X[["shots", "xg"]].fillna(0)
opp = X.rename(columns={"side": "oside", "shots": "c_shots", "xg": "c_xg"})
opp["side"] = np.where(opp.oside == "h", "a", "h")
opp["state"] = opp.state.map({"leading": "trailing", "trailing": "leading", "level": "level"})
X = X.merge(opp[["match_id", "side", "state", "bin", "c_shots", "c_xg"]], on=["match_id", "side", "state", "bin"], how="left")
X[["c_shots", "c_xg"]] = X[["c_shots", "c_xg"]].fillna(0)
X = X.merge(mt.assign(year=mt.season)[["match_id", "home_team", "away_team", "year"]], on="match_id")
X["team"] = np.where(X.side == "h", X.home_team, X.away_team); X["opp"] = np.where(X.side == "h", X.away_team, X.home_team)
X["home"] = (X.side == "h").astype(int)
tot = X.groupby("state").agg(minutes=("e", "sum"), shots=("shots", "sum"), xg=("xg", "sum"), c_shots=("c_shots", "sum"), c_xg=("c_xg", "sum"))
tot["shots90"] = tot.shots / tot.minutes * 90; tot["xg90"] = tot.xg / tot.minutes * 90; tot["xg_per_shot"] = tot.xg / tot.shots
tot["c_shots90"] = tot.c_shots / tot.minutes * 90; tot["c_xg90"] = tot.c_xg / tot.minutes * 90
tot.to_csv("score_state_totals.csv"); print(tot.round(3).to_string())
Wg = weekly_games(); hm = Wg[Wg.home == 1][["year", "team", "opponent", "goals_for", "goals_against"]]
fin = pd.DataFrame(recon, columns=["match_id", "ntrans", "last_hs"]).merge(mt[["match_id", "season", "home_team", "away_team"]], on="match_id")
fin = fin.merge(hm, left_on=["season", "home_team", "away_team"], right_on=["year", "team", "opponent"])
agree = (np.sign(fin.goals_for - fin.goals_against) == fin.last_hs).mean()
print("last shot state agrees with final result:", round(agree, 4), "matches", len(fin))
_ok = np.sign(fin.goals_for - fin.goals_against) == fin.last_hs
_bad = fin[~_ok]
pd.DataFrame([dict(matches=len(fin), agree_share=float(_ok.mean()), disagree=int(len(_bad)), disagree_last_state_level=float((_bad.last_hs == 0).mean()),
                   disagree_margin_one=float(((_bad.goals_for - _bad.goals_against).abs() == 1).mean()))]).to_csv("score_state_reconstruction_check.csv", index=False)

X["ts"] = X.team + "_" + X.year.astype(str); X["ots"] = X.opp + "_" + X.year.astype(str)
grp = pd.factorize(X.match_id)[0]
def design(d, fe):
    cols = [np.ones(len(d)), (d.state == "leading").astype(float), (d.state == "trailing").astype(float), d.home.astype(float)]
    for b in range(1, 7): cols.append((d.bin == b).astype(float))
    M = np.column_stack(cols)
    if fe:
        M = np.hstack([M, pd.get_dummies(d.ts, drop_first=True).values.astype(float), pd.get_dummies(d.ots, drop_first=True).values.astype(float)])
    return M
def cluster_cov(M, resid_w, groups, bread_inv):
    G = groups.max() + 1
    Sm = np.zeros((G, M.shape[1])); np.add.at(Sm, groups, M * resid_w[:, None])
    return bread_inv @ (Sm.T @ Sm) @ bread_inv * (G / (G - 1))
def poisson_fit(M, y, off, groups, iters=60):
    beta = np.zeros(M.shape[1]); beta[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(iters):
        eta = M @ beta + off; mu = np.exp(eta); z = eta - off + (y - mu) / mu
        A = M.T @ (M * mu[:, None]) + 1e-8 * np.eye(M.shape[1]); new = np.linalg.solve(A, M.T @ (mu * z))
        if np.abs(new - beta).max() < 1e-9: beta = new; break
        beta = new
    mu = np.exp(M @ beta + off); A = M.T @ (M * mu[:, None]) + 1e-8 * np.eye(M.shape[1])
    return beta, cluster_cov(M, y - mu, groups, np.linalg.inv(A))
def re_design(d):
    F = np.column_stack([np.ones(len(d)), (d.state == "leading").astype(float), (d.state == "trailing").astype(float), d.home.astype(float)] + [(d.bin == b).astype(float) for b in range(1, 7)])
    T = pd.get_dummies(d.ts).values.astype(float); O = pd.get_dummies(d.ots).values.astype(float)
    return np.hstack([F, T, O]), F.shape[1], T.shape[1], O.shape[1]
def pen_diag(nf, nt, no, st, so):
    return np.concatenate([np.zeros(nf), np.full(nt, 1 / st), np.full(no, 1 / so)])
def poisson_re(M, nf, nt, no, y, off, groups, outer=25, inner=30):
    st = so = 0.05
    beta = np.zeros(M.shape[1]); beta[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(outer):
        for _ in range(inner):
            eta = M @ beta + off; mu = np.exp(eta); z = eta - off + (y - mu) / mu
            D = pen_diag(nf, nt, no, st, so)
            A = M.T @ (M * mu[:, None]) + np.diag(D) + 1e-8 * np.eye(M.shape[1]); new = np.linalg.solve(A, M.T @ (mu * z))
            if np.abs(new - beta).max() < 1e-8: beta = new; break
            beta = new
        C = np.linalg.inv(A); bt = beta[nf:nf + nt]; bo = beta[nf + nt:]
        st_new = (bt @ bt + np.trace(C[nf:nf + nt, nf:nf + nt])) / nt; so_new = (bo @ bo + np.trace(C[nf + nt:, nf + nt:])) / no
        if abs(st_new - st) + abs(so_new - so) < 1e-6: st, so = st_new, so_new; break
        st, so = st_new, so_new
    mu = np.exp(M @ beta + off); D = pen_diag(nf, nt, no, st, so)
    A = M.T @ (M * mu[:, None]) + np.diag(D) + 1e-8 * np.eye(M.shape[1])
    return beta, cluster_cov(M, y - mu, groups, np.linalg.inv(A)), np.sqrt(st), np.sqrt(so)
def gauss_re(M, nf, nt, no, y, w, groups, outer=40):
    st = so = 0.01; s2 = 0.01
    for _ in range(outer):
        D = pen_diag(nf, nt, no, st, so) * s2
        A = M.T @ (M * w[:, None]) + np.diag(D) + 1e-8 * np.eye(M.shape[1]); C = np.linalg.inv(A); beta = C @ (M.T @ (w * y))
        r = y - M @ beta; edf = np.trace(C @ (M.T @ (M * w[:, None]))); s2n = (w * r * r).sum() / (len(y) - edf)
        bt = beta[nf:nf + nt]; bo = beta[nf + nt:]
        st_n = (bt @ bt + s2n * np.trace(C[nf:nf + nt, nf:nf + nt])) / nt; so_n = (bo @ bo + s2n * np.trace(C[nf + nt:, nf + nt:])) / no
        done = abs(st_n - st) + abs(so_n - so) < 1e-9
        st, so, s2 = st_n, so_n, s2n
        if done: break
    D = pen_diag(nf, nt, no, st, so) * s2
    A = M.T @ (M * w[:, None]) + np.diag(D) + 1e-8 * np.eye(M.shape[1])
    return beta, cluster_cov(M, w * (y - M @ beta), groups, np.linalg.inv(A)), np.sqrt(st), np.sqrt(so)
import sys
VAR = sys.argv[1]
if VAR == "agree":
    _badids = set(fin[~_ok].match_id)
    X = X[~X.match_id.isin(_badids)].copy()
elif VAR == "no90":
    X = X[X.bin < 6].copy()
grp = pd.factorize(X.match_id)[0]
off = np.log(X.e.values)
print(VAR, "rows", len(X), "matches", X.match_id.nunique())
_sd = season_df("np"); _sd["rank"] = _sd.groupby("year")["POINTS"].rank(ascending=False, method="min")
_rk = {(t, y_): r_ for t, y_, r_ in zip(_sd.TEAM, _sd.year, _sd["rank"])}
X["rank"] = [_rk[(t, y_)] for t, y_ in zip(X.team.replace(NAME_MAP), X.year)]
_het = []
def het_fit(label, flag, names):
    M0 = design(X, True); Lf = ((X.state == "leading").values * flag).astype(float); Tf = ((X.state == "trailing").values * flag).astype(float)
    M2 = np.column_stack([M0, Lf, Tf]); ne = M2.shape[1]
    for y, lab in (("shots", "own shots per minute"), ("xg", "own xG per minute"), ("c_shots", "shots conceded per minute"), ("c_xg", "xG conceded per minute")):
        b, cov = poisson_fit(M2, X[y].values, off, grp)
        for g, gname in ((0, names[0]), (1, names[1])):
            for i, e, s_ in ((1, ne - 2, "leading"), (2, ne - 1, "trailing")):
                est = b[i] + g * b[e]; var = cov[i, i] + g * g * cov[e, e] + 2 * g * cov[i, e]; se = np.sqrt(var)
                _het.append(dict(split=label, group=gname, outcome=lab, state=s_, rate_ratio=np.exp(est), lo=np.exp(est - 1.96 * se), hi=np.exp(est + 1.96 * se)))
        _het.append(dict(split=label, group="difference in state effects", outcome=lab, state="leading", rate_ratio=np.exp(b[ne - 2]), lo=np.exp(b[ne - 2] - 1.96 * np.sqrt(cov[ne - 2, ne - 2])), hi=np.exp(b[ne - 2] + 1.96 * np.sqrt(cov[ne - 2, ne - 2]))))
het_fit("team finished in the top six", (X["rank"] <= 6).values, ("other teams", "top six"))
het_fit("minute of play", (X["bin"] >= 4).values, ("first 60 minutes", "after 60 minutes"))
pd.DataFrame(_het).to_csv("het_%s.csv" % VAR, index=False)
print(pd.DataFrame(_het).round(3).to_string())
