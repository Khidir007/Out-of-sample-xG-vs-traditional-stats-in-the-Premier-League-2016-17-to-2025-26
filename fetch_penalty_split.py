import time, json
import requests
import pandas as pd

SEASONS = list(range(2016, 2026))
LEAGUE = "EPL"
DELAY = 1.5
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

def get_json(url, referer):
    r = requests.get(url, headers={**HEADERS, "Referer": referer, "X-Requested-With": "XMLHttpRequest"}, timeout=20)
    r.raise_for_status()
    return r.json()

rows = []
for season in SEASONS:
    lg = get_json(f"https://understat.com/getLeagueData/{LEAGUE}/{season}", f"https://understat.com/league/{LEAGUE}/{season}")
    matches = [m for m in lg.get("dates", []) if m.get("isResult")]
    print(season, len(matches), "matches", flush=True)
    time.sleep(DELAY)
    for m in matches:
        mid = int(m["id"])
        teams = {"h": m["h"]["title"], "a": m["a"]["title"]}
        try:
            shots = get_json(f"https://understat.com/getMatchData/{mid}", f"https://understat.com/match/{mid}").get("shots") or {}
        except Exception as e:
            print("FAILED", mid, e, flush=True); time.sleep(DELAY); continue
        for side in ("h", "a"):
            for s in shots.get(side, []):
                rows.append({"season": season, "match_id": mid, "team": teams[side],
                             "situation": s["situation"], "result": s["result"], "xG": float(s["xG"])})
        time.sleep(DELAY)

shots = pd.DataFrame(rows)
shots["is_pen"] = shots["situation"].eq("Penalty")
shots["is_goal"] = shots["result"].eq("Goal")
shots["is_own"] = shots["result"].eq("OwnGoal")
out = (shots.groupby(["season", "team"])
       .apply(lambda g: pd.Series({
           "shots": len(g), "xG_all": g.xG.sum(),
           "pen_attempts": g.is_pen.sum(), "pen_xG": g.loc[g.is_pen, "xG"].sum(),
           "goals_all": g.is_goal.sum(), "pen_goals": (g.is_goal & g.is_pen).sum(),
           "np_goals": (g.is_goal & ~g.is_pen).sum(), "np_xG": g.loc[~g.is_pen, "xG"].sum()}))
       .reset_index())
out["npG_minus_npxG"] = out["np_goals"] - out["np_xG"]
out.to_csv("understat_team_season_penalty_split.csv", index=False)
print(out.groupby("season")[["pen_attempts", "pen_xG", "npG_minus_npxG"]].agg({"pen_attempts": "sum", "pen_xG": "sum", "npG_minus_npxG": "mean"}))
print("Mean npG - npxG per team-season:", out["npG_minus_npxG"].mean())
print("Penalty xG per attempt:", out.pen_xG.sum() / out.pen_attempts.sum())
try:
    from google.colab import files
    files.download("understat_team_season_penalty_split.csv")
except Exception as e:
    print("Saved understat_team_season_penalty_split.csv")
