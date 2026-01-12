# app.py — NHL Props Tester (manual teams, strict roster, robust)
# Run: streamlit run app.py
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="NHL Props — Beta", layout="wide")
st.title("NHL Props — Beta (manual teams, STRICT roster)")


# =========================================================
# Sidebar inputs
# =========================================================
DEFAULT_BASE = "."
BASE = Path(st.sidebar.text_input("Data folder", value=DEFAULT_BASE))
SEASON = int(st.sidebar.number_input("Season", value=2025, step=1))

st.sidebar.markdown("---")
st.sidebar.markdown("### Match assumptions (TEAM clock minutes)")
MINUTES_5V5_TOTAL = float(st.sidebar.slider("Team 5v5 minutes total", 44.0, 56.0, 50.0, 0.5))
MINUTES_PP_TOTAL = float(st.sidebar.slider("Team PP minutes total (5on4)", 2.0, 10.0, 6.0, 0.5))

st.sidebar.markdown("### Player projection")
POINTS_PER_GOAL = float(st.sidebar.slider("Points per goal (team)", 1.8, 2.6, 2.10, 0.05))
FLOOR_TOI_5V5_PG = float(st.sidebar.slider("Floor TOI 5v5 (min/game)", 0.0, 12.0, 6.0, 0.5))
FLOOR_TOI_PP_PG = float(st.sidebar.slider("Floor TOI PP (min/game)", 0.0, 6.0, 1.0, 0.25))

st.sidebar.markdown("### Reliability / shrinkage")
K_SECONDS_SHRINK = float(st.sidebar.slider("Shrink strength k (seconds)", 3600.0, 36000.0, 18000.0, 600.0))


# =========================================================
# Utilities
# =========================================================
def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def per60_from_seconds(df: pd.DataFrame, value_col: str, toi_sec_col: str) -> np.ndarray:
    toi_sec = pd.to_numeric(df[toi_sec_col], errors="coerce").fillna(0.0)
    hours = (toi_sec / 60.0) / 60.0
    val = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    return np.where(hours > 0, val / hours, 0.0)


def infer_icetime_unit_minutes_or_seconds(df: pd.DataFrame, icetime_col: str, games_col: str, label: str):
    tmp = df.copy()
    tmp[icetime_col] = pd.to_numeric(tmp.get(icetime_col, 0), errors="coerce").fillna(0.0)
    tmp[games_col] = pd.to_numeric(tmp.get(games_col, 0), errors="coerce").fillna(0.0)

    tmp = tmp[(tmp[games_col] >= 5) & (tmp[icetime_col] > 0)].copy()
    if tmp.empty:
        return "unknown", 1.0

    gp = tmp[games_col].replace(0, np.nan)

    toi_pg_if_minutes = (tmp[icetime_col] / gp)
    toi_pg_if_seconds = ((tmp[icetime_col] / 60.0) / gp)

    med_m = float(np.nanmedian(toi_pg_if_minutes))
    med_s = float(np.nanmedian(toi_pg_if_seconds))

    def in_band(x, lo, hi):
        return (x >= lo) and (x <= hi)

    if label.lower().startswith("goalie"):
        m_ok = in_band(med_m, 35.0, 75.0)
        s_ok = in_band(med_s, 35.0, 75.0)
    else:
        m_ok = in_band(med_m, 4.0, 35.0)
        s_ok = in_band(med_s, 4.0, 35.0)

    if m_ok and not s_ok:
        return "minutes", 60.0
    if s_ok and not m_ok:
        return "seconds", 1.0

    return f"ambiguous(med_if_minutes={med_m:.2f}, med_if_seconds={med_s:.2f})", None


def shrink_metric(value_sit, toi_sit_sec, value_all, toi_all_sec, k_seconds=18000.0):
    toi_sit_sec = safe_float(toi_sit_sec, 0.0)
    w = toi_sit_sec / (toi_sit_sec + k_seconds) if toi_sit_sec > 0 else 0.0

    if value_sit is None or (isinstance(value_sit, float) and np.isnan(value_sit)):
        w = 0.0

    if value_all is None or (isinstance(value_all, float) and np.isnan(value_all)):
        return value_sit

    return w * float(value_sit) + (1.0 - w) * float(value_all)


def poisson_home_win_prob(lam_home, lam_away, max_goals=15):
    p = 0.0
    for h in range(max_goals + 1):
        ph = math.exp(-lam_home) * (lam_home**h) / math.factorial(h)
        for a in range(max_goals + 1):
            pa = math.exp(-lam_away) * (lam_away**a) / math.factorial(a)
            if h > a:
                p += ph * pa
    return p


def p_ge_1(lam):
    return 1.0 - math.exp(-lam)


def p_ge_2(lam):
    return 1.0 - math.exp(-lam) * (1.0 + lam)


def get_team_row(team_stats, team, situation):
    r = team_stats[(team_stats["team"] == team) & (team_stats["situation"] == situation)]
    if r.empty:
        raise ValueError(f"team_stats missing: {team} {situation}")
    return r.iloc[0]


# =========================================================
# Load CSVs
# =========================================================
@st.cache_data(show_spinner=False)
def load_csvs(base: Path):
    players_df = pd.read_csv(base / "joueurs2025.csv")
    teams_df = pd.read_csv(base / "teams.csv")
    goalies_df = pd.read_csv(base / "goalies.csv")
    return players_df, teams_df, goalies_df


# =========================================================
# Build stats
# =========================================================
@st.cache_data(show_spinner=False)
def build_skater_stats(players_df: pd.DataFrame, season: int):
    df = players_df.copy()
    df = df[df["season"] == season].copy()
    df = df[df["position"].isin(["C", "L", "R", "D"])].copy()
    df = df[df["situation"].isin(["5on5", "5on4", "4on5", "all"])].copy()

    must = ["playerId", "name", "team", "position", "situation", "games_played", "icetime"]
    for c in must:
        if c not in df.columns:
            raise ValueError(f"joueurs2025.csv missing column: {c}")

    unit, mult = infer_icetime_unit_minutes_or_seconds(df[df["situation"] == "all"], "icetime", "games_played", "skater")
    if mult is None:
        raise ValueError(f"Skater icetime unit ambiguous: {unit}. Fix CSV.")

    df["icetime_raw"] = pd.to_numeric(df["icetime"], errors="coerce").fillna(0.0)
    df["icetime_sec"] = df["icetime_raw"] * mult
    df["games_played"] = pd.to_numeric(df["games_played"], errors="coerce").fillna(0.0)
    df["toi_pg_min"] = np.where(df["games_played"] > 0, (df["icetime_sec"] / 60.0) / df["games_played"], 0.0)

    want = [
        "I_F_points", "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
        "I_F_xGoals", "I_F_shotsOnGoal",
        "OnIce_F_xGoals", "OnIce_A_xGoals",
    ]
    for c in want:
        if c not in df.columns:
            df[c] = 0.0

    df["iP60"] = per60_from_seconds(df, "I_F_points", "icetime_sec")
    df["playerId"] = pd.to_numeric(df["playerId"], errors="coerce").astype("Int64")
    df = df.sort_values("icetime_sec", ascending=False).drop_duplicates(["playerId", "team", "situation"])
    df.attrs["icetime_unit_skater"] = unit
    return df.dropna(subset=["playerId"]).copy()


@st.cache_data(show_spinner=False)
def build_goalie_stats(goalies_df: pd.DataFrame, season: int):
    df = goalies_df.copy()
    df = df[df["season"] == season].copy()
    df = df[df["position"] == "G"].copy()
    df = df[df["situation"].isin(["5on5", "4on5", "5on4", "all", "other"])].copy()

    must = ["playerId", "name", "team", "situation", "games_played", "icetime", "xGoals", "goals"]
    for c in must:
        if c not in df.columns:
            raise ValueError(f"goalies.csv missing column: {c}")

    unit, mult = infer_icetime_unit_minutes_or_seconds(df[df["situation"] == "all"], "icetime", "games_played", "goalie")
    if mult is None:
        raise ValueError(f"Goalie icetime unit ambiguous: {unit}. Fix CSV.")

    df["icetime_raw"] = pd.to_numeric(df["icetime"], errors="coerce").fillna(0.0)
    df["icetime_sec"] = df["icetime_raw"] * mult
    df["playerId"] = pd.to_numeric(df["playerId"], errors="coerce").astype("Int64")

    tmp = df.copy()
    tmp["_diff"] = pd.to_numeric(tmp["xGoals"], errors="coerce").fillna(0.0) - pd.to_numeric(tmp["goals"], errors="coerce").fillna(0.0)
    df["GSAx60"] = per60_from_seconds(tmp, "_diff", "icetime_sec")
    df.attrs["icetime_unit_goalie"] = unit

    df = df.sort_values("icetime_sec", ascending=False).drop_duplicates(["playerId", "team", "situation"])
    return df.dropna(subset=["playerId"]).copy()


@st.cache_data(show_spinner=False)
def build_team_stats(teams_df: pd.DataFrame, season: int):
    df = teams_df.copy()
    df = df[df["season"] == season].copy()
    df = df[df["situation"].isin(["5on5", "5on4", "4on5", "all", "other"])].copy()

    must = ["team", "situation", "iceTime", "xGoalsFor", "xGoalsAgainst"]
    for c in must:
        if c not in df.columns:
            raise ValueError(f"teams.csv missing column: {c}")

    ice = pd.to_numeric(df["iceTime"], errors="coerce").fillna(0.0)
    med = float(np.nanmedian(ice[ice > 0])) if (ice > 0).any() else 0.0
    mult = 60.0 if (med > 0 and med < 1000) else 1.0  # minutes vs seconds heuristic

    df["iceTime_sec"] = ice * mult
    df["xGoalsFor"] = pd.to_numeric(df["xGoalsFor"], errors="coerce").fillna(0.0)
    df["xGoalsAgainst"] = pd.to_numeric(df["xGoalsAgainst"], errors="coerce").fillna(0.0)

    df["xGF60"] = per60_from_seconds(df.rename(columns={"iceTime_sec": "toi"}), "xGoalsFor", "toi")
    df["xGA60"] = per60_from_seconds(df.rename(columns={"iceTime_sec": "toi"}), "xGoalsAgainst", "toi")
    return df


# =========================================================
# Goalie + lambda
# =========================================================
def goalie_gsax60(goalie_stats, goalie_id, situation, k_seconds):
    g = goalie_stats[goalie_stats["playerId"] == goalie_id]
    if g.empty:
        return 0.0
    a = g[g["situation"] == "all"]
    s = g[g["situation"] == situation]
    all_val = float(a.iloc[0].get("GSAx60", 0.0)) if not a.empty else 0.0
    all_toi = float(a.iloc[0].get("icetime_sec", 0.0)) if not a.empty else 0.0
    if s.empty:
        return all_val
    sit_val = float(s.iloc[0].get("GSAx60", all_val))
    sit_toi = float(s.iloc[0].get("icetime_sec", 0.0))
    return float(shrink_metric(sit_val, sit_toi, all_val, all_toi, k_seconds=k_seconds))


def matchup_lambda(home_team, away_team, team_stats, goalie_stats, home_goalie_id, away_goalie_id,
                  minutes_5v5=50.0, minutes_pp=6.0, k_seconds=18000.0):
    h5 = get_team_row(team_stats, home_team, "5on5")
    a5 = get_team_row(team_stats, away_team, "5on5")
    hp = get_team_row(team_stats, home_team, "5on4")
    ap = get_team_row(team_stats, away_team, "5on4")
    hk = get_team_row(team_stats, home_team, "4on5")
    ak = get_team_row(team_stats, away_team, "4on5")

    home_xg60_5v5 = 0.5 * (float(h5["xGF60"]) + float(a5["xGA60"]))
    away_xg60_5v5 = 0.5 * (float(a5["xGF60"]) + float(h5["xGA60"]))

    home_xg60_pp = 0.5 * (float(hp["xGF60"]) + float(ak["xGA60"]))
    away_xg60_pp = 0.5 * (float(ap["xGF60"]) + float(hk["xGA60"]))

    lam_home_xg = (minutes_5v5/60.0) * home_xg60_5v5 + (minutes_pp/60.0) * home_xg60_pp
    lam_away_xg = (minutes_5v5/60.0) * away_xg60_5v5 + (minutes_pp/60.0) * away_xg60_pp

    away_gsax = goalie_gsax60(goalie_stats, away_goalie_id, "5on5", k_seconds)*(minutes_5v5/60.0) + \
                goalie_gsax60(goalie_stats, away_goalie_id, "4on5", k_seconds)*(minutes_pp/60.0)
    home_gsax = goalie_gsax60(goalie_stats, home_goalie_id, "5on5", k_seconds)*(minutes_5v5/60.0) + \
                goalie_gsax60(goalie_stats, home_goalie_id, "4on5", k_seconds)*(minutes_pp/60.0)

    lam_home = max(0.05, lam_home_xg - away_gsax)
    lam_away = max(0.05, lam_away_xg - home_gsax)

    return {
        "lam_home": lam_home,
        "lam_away": lam_away,
        "total": lam_home + lam_away,
        "p_home_win_reg": poisson_home_win_prob(lam_home, lam_away),
        "home_xg_raw": lam_home_xg,
        "away_xg_raw": lam_away_xg,
    }


# =========================================================
# Defaults + strict minutes
# =========================================================
def auto_default_lineups(skater_stats, goalie_stats, team):
    df5 = skater_stats[(skater_stats["team"] == team) & (skater_stats["situation"] == "5on5")].sort_values("icetime_sec", ascending=False)
    ids5 = df5["playerId"].astype(int).tolist()

    def s(lst, a, b):
        return lst[a:b] if len(lst) > a else []

    lineup_5v5 = {"L1": s(ids5, 0, 5), "L2": s(ids5, 5, 10), "L3": s(ids5, 10, 15), "L4": s(ids5, 15, 20)}

    dfpp = skater_stats[(skater_stats["team"] == team) & (skater_stats["situation"] == "5on4")].sort_values("icetime_sec", ascending=False)
    idsp = dfpp["playerId"].astype(int).tolist()
    lineup_pp = {"PP1": s(idsp, 0, 5), "PP2": s(idsp, 5, 10)}

    dfpk = skater_stats[(skater_stats["team"] == team) & (skater_stats["situation"] == "4on5")].sort_values("icetime_sec", ascending=False)
    idsk = dfpk["playerId"].astype(int).tolist()
    lineup_pk = {"PK1": s(idsk, 0, 5), "PK2": s(idsk, 5, 10)}

    g_all = goalie_stats[(goalie_stats["team"] == team) & (goalie_stats["situation"] == "all")].sort_values("icetime_sec", ascending=False)
    goalie_id = int(g_all.iloc[0]["playerId"]) if not g_all.empty else None
    return lineup_5v5, lineup_pp, lineup_pk, goalie_id


def _get_player_rows(skater_stats, team, pid, situation):
    sub = skater_stats[(skater_stats["team"] == team) & (skater_stats["playerId"] == pid)]
    if sub.empty:
        return None, None
    r_all = sub[sub["situation"] == "all"]
    r_sit = sub[sub["situation"] == situation]
    return (r_sit.iloc[0] if not r_sit.empty else None), (r_all.iloc[0] if not r_all.empty else None)


def projected_toi_pg_min(skater_stats, team, pid, situation, floor_min_pg, k_seconds):
    sit_row, all_row = _get_player_rows(skater_stats, team, pid, situation)

    if sit_row is None and all_row is None:
        return float(floor_min_pg)

    sit_val = safe_float(sit_row.get("toi_pg_min") if sit_row is not None else np.nan, np.nan)
    all_val = safe_float(all_row.get("toi_pg_min") if all_row is not None else np.nan, np.nan)

    sit_toi = safe_float(sit_row.get("icetime_sec") if sit_row is not None else 0.0, 0.0)
    all_toi = safe_float(all_row.get("icetime_sec") if all_row is not None else 0.0, 0.0)

    if np.isnan(all_val):
        base = sit_val
    else:
        base = shrink_metric(sit_val, sit_toi, all_val, all_toi, k_seconds=k_seconds)

    if base is None or (isinstance(base, float) and np.isnan(base)):
        base = floor_min_pg

    return max(float(base), float(floor_min_pg))


def allocate_minutes_in_unit(skater_stats, team, player_ids, situation, minutes_total, floor_min_pg, k_seconds):
    pids = [int(x) for x in player_ids if x is not None]
    if not pids:
        return {}
    weights = []
    for pid in pids:
        w = projected_toi_pg_min(skater_stats, team, pid, situation, floor_min_pg, k_seconds)
        weights.append(max(w, 0.0))
    s = sum(weights)
    if s <= 0:
        return {pid: float(minutes_total) / len(pids) for pid in pids}
    return {pid: float(minutes_total) * (w / s) for pid, w in zip(pids, weights)}


def _renorm_minutes(d, target):
    s = float(sum(d.values()))
    if s <= 0:
        return d
    k = float(target) / s
    return {pid: v * k for pid, v in d.items()}


def build_projected_minutes_strict(skater_stats, team, lineup_5v5, lineup_pp, out_ids,
                                   minutes_5v5_total, minutes_pp_total,
                                   floor_5v5_pg, floor_pp_pg, k_seconds):
    out_set = set(int(x) for x in (out_ids or []))

    roster = set()
    for ids in (lineup_5v5 or {}).values():
        roster |= set(int(x) for x in ids)
    for ids in (lineup_pp or {}).values():
        roster |= set(int(x) for x in ids)

    roster = [pid for pid in sorted(roster) if pid not in out_set]
    if not roster:
        return [], {}, {}

    target5 = float(minutes_5v5_total) * 5.0
    targetpp = float(minutes_pp_total) * 5.0

    base_lines = {"L1": 18.0, "L2": 14.0, "L3": 11.0, "L4": 7.0}
    scale = float(minutes_5v5_total) / sum(base_lines.values())
    per_player_line_mins = {k: v * scale for k, v in base_lines.items()}

    min5 = {}
    for L, per_player_mins in per_player_line_mins.items():
        ids = [int(x) for x in lineup_5v5.get(L, []) if int(x) in roster]
        if not ids:
            continue
        alloc = allocate_minutes_in_unit(skater_stats, team, ids, "5on5",
                                         per_player_mins * len(ids), floor_5v5_pg, k_seconds)
        for pid, m in alloc.items():
            min5[pid] = min5.get(pid, 0.0) + float(m)

    pp_split = {"PP1": 0.70, "PP2": 0.30}
    minpp = {}
    for U, frac in pp_split.items():
        team_clock = float(minutes_pp_total) * float(frac)
        ids = [int(x) for x in lineup_pp.get(U, []) if int(x) in roster]
        if not ids:
            continue
        alloc = allocate_minutes_in_unit(skater_stats, team, ids, "5on4",
                                         team_clock * len(ids), floor_pp_pg, k_seconds)
        for pid, m in alloc.items():
            minpp[pid] = minpp.get(pid, 0.0) + float(m)

    min5 = {pid: m for pid, m in min5.items() if pid in roster}
    minpp = {pid: m for pid, m in minpp.items() if pid in roster}

    min5 = _renorm_minutes(min5, target5)
    minpp = _renorm_minutes(minpp, targetpp)

    if abs(sum(min5.values()) - target5) > 0.5:
        raise ValueError(f"[{team}] 5v5 player-min failed: sum={sum(min5.values()):.2f} target={target5:.2f}")
    if abs(sum(minpp.values()) - targetpp) > 0.5:
        raise ValueError(f"[{team}] PP player-min failed: sum={sum(minpp.values()):.2f} target={targetpp:.2f}")

    return roster, min5, minpp


# =========================================================
# Player points model + value
# =========================================================
def player_expected_points(skater_stats, team, roster_ids, min5_by_pid, minpp_by_pid, k_seconds):
    rows = []
    for pid in roster_ids:
        pid = int(pid)
        s5, a5 = _get_player_rows(skater_stats, team, pid, "5on5")
        sp, ap = _get_player_rows(skater_stats, team, pid, "5on4")

        base_row = next((r for r in (a5, ap, s5, sp) if r is not None), None)
        name = base_row.get("name") if base_row is not None else f"#{pid}"
        pos = base_row.get("position") if base_row is not None else ""

        def ip60_shr(sit_row, all_row):
            sit_val = safe_float(sit_row.get("iP60") if sit_row is not None else np.nan, np.nan)
            all_val = safe_float(all_row.get("iP60") if all_row is not None else np.nan, np.nan)
            sit_toi = safe_float(sit_row.get("icetime_sec") if sit_row is not None else 0.0, 0.0)
            all_toi = safe_float(all_row.get("icetime_sec") if all_row is not None else 0.0, 0.0)
            if np.isnan(all_val):
                base = sit_val
            else:
                base = shrink_metric(sit_val, sit_toi, all_val, all_toi, k_seconds=k_seconds)
            if base is None or (isinstance(base, float) and np.isnan(base)):
                base = 0.0
            return max(0.0, float(base))

        ip60_5 = ip60_shr(s5, a5)
        ip60_p = ip60_shr(sp, ap)

        m5 = float(min5_by_pid.get(pid, 0.0))
        mp = float(minpp_by_pid.get(pid, 0.0))

        raw = (m5/60.0) * ip60_5 + (mp/60.0) * ip60_p

        rows.append({
            "playerId": pid,
            "name": name,
            "team": team,
            "position": pos,
            "min_proj_5v5": m5,
            "min_proj_PP": mp,
            "raw_E_points": raw,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["raw_E_points","min_proj_PP","min_proj_5v5"], ascending=False).reset_index(drop=True)
    return out


def scale_players_to_team(df_players, lam_team, points_per_goal):
    out = df_players.copy()
    target = float(lam_team) * float(points_per_goal)
    s = float(out["raw_E_points"].sum()) if not out.empty else 0.0
    out["E_points"] = 0.0 if s <= 0 else out["raw_E_points"] * (target / s)
    out["P1"] = out["E_points"].apply(p_ge_1)
    out["P2"] = out["E_points"].apply(p_ge_2)
    out["Over0.5_prob"] = out["P1"]
    out["Over1.5_prob"] = out["P2"]
    return out


def odds_key(season, home, away, market, side):
    return f"odds::{season}::{away}@{home}::{market}::{side}"


def init_odds_state(key, df_markets):
    base_cols = ["playerId","name","team","position","min_proj_5v5","min_proj_PP","E_points","P1","P2"]
    base = df_markets[base_cols].copy()

    if key not in st.session_state:
        base["odds_decimal"] = np.nan
        st.session_state[key] = base
        return

    old = st.session_state[key][["playerId","odds_decimal"]].drop_duplicates("playerId")
    st.session_state[key] = base.merge(old, on="playerId", how="left")


def compute_value(df, market):
    out = df.copy()
    model_p = out["P1"].astype(float) if market == "1+ point" else out["P2"].astype(float)
    odds = pd.to_numeric(out["odds_decimal"], errors="coerce")
    implied = np.where(odds > 0, 1.0 / odds, np.nan)
    out["model_prob"] = model_p
    out["implied_p"] = implied
    out["edge"] = out["model_prob"] - out["implied_p"]
    out["EV"] = odds * out["model_prob"] - 1.0
    cols = ["team","name","position","min_proj_5v5","min_proj_PP","E_points","model_prob","odds_decimal","implied_p","edge","EV"]
    return out[cols].sort_values(["EV","edge"], ascending=False).reset_index(drop=True)


# =========================================================
# UI helpers
# =========================================================
def team_player_options(skater_stats, team):
    df = skater_stats[(skater_stats["team"] == team) & (skater_stats["situation"] == "all")][
        ["playerId","name","position","icetime_sec","games_played","toi_pg_min"]
    ].drop_duplicates()
    df = df.sort_values(["position","icetime_sec"], ascending=[True, False])
    opts = []
    for _, r in df.iterrows():
        pid = int(r["playerId"])
        toi_pg = safe_float(r.get("toi_pg_min"), 0.0)
        gp = int(safe_float(r.get("games_played"), 0))
        opts.append((f"{r['name']} ({r['position']}) — {pid} | TOI/g={toi_pg:.1f} (gp={gp})", pid))
    return opts


def team_goalie_options(goalie_stats, team):
    df = goalie_stats[(goalie_stats["team"] == team) & (goalie_stats["situation"] == "all")][
        ["playerId","name","icetime_sec","games_played"]
    ].drop_duplicates()
    df = df.sort_values("icetime_sec", ascending=False)
    opts = []
    for _, r in df.iterrows():
        pid = int(r["playerId"])
        gp = int(safe_float(r.get("games_played"), 0))
        opts.append((f"{r['name']} — {pid} (gp={gp})", pid))
    return opts


def pick_ids(title, options, k, default_ids=None):
    label_to_id = {lab: pid for lab, pid in options}
    id_to_label = {pid: lab for lab, pid in options}
    defaults = []
    if default_ids:
        for pid in default_ids:
            if int(pid) in id_to_label:
                defaults.append(id_to_label[int(pid)])
    defaults = defaults[:k]
    chosen = st.multiselect(title, list(label_to_id.keys()), default=defaults, max_selections=k)
    return [label_to_id[x] for x in chosen]


# =========================================================
# Load & build
# =========================================================
try:
    players_df, teams_df, goalies_df = load_csvs(BASE)
except Exception as e:
    st.error(f"Could not load CSVs from {BASE}: {e}")
    st.stop()

try:
    skater_stats = build_skater_stats(players_df, SEASON)
    goalie_stats = build_goalie_stats(goalies_df, SEASON)
    team_stats = build_team_stats(teams_df, SEASON)
except Exception as e:
    st.error(f"Data preprocessing error: {e}")
    st.stop()

teams = sorted(skater_stats["team"].dropna().unique().tolist())
if not teams:
    st.error("No teams found in skater_stats for this season.")
    st.stop()

tmp_all = skater_stats[skater_stats["situation"] == "all"].copy()
if not tmp_all.empty:
    med_toi_pg = float(np.nanmedian(tmp_all["toi_pg_min"].values))
    if med_toi_pg < 2.0 or med_toi_pg > 35.0:
        st.error(f"TOI/game looks wrong (median={med_toi_pg:.2f} min). Fix icetime unit in joueurs2025.csv.")
        st.stop()


# =========================================================
# Match selection (manual)
# =========================================================
colA, colB, colC = st.columns([2, 2, 2])
with colA:
    home_team = st.selectbox("Home team", teams, index=0)
with colB:
    away_choices = [t for t in teams if t != home_team]
    away_team = st.selectbox("Away team", away_choices, index=0 if away_choices else 0)
with colC:
    edit_lineups = st.checkbox("Edit lineups (manual)", value=True)

home_auto = auto_default_lineups(skater_stats, goalie_stats, home_team)
away_auto = auto_default_lineups(skater_stats, goalie_stats, away_team)

home_lineup_5v5, home_lineup_pp, _, home_goalie_id = home_auto
away_lineup_5v5, away_lineup_pp, _, away_goalie_id = away_auto

home_out_ids, away_out_ids = [], []


# =========================================================
# Manual strict roster
# =========================================================
if edit_lineups:
    st.subheader("Manual lineups — STRICT roster")
    home_opts = team_player_options(skater_stats, home_team)
    away_opts = team_player_options(skater_stats, away_team)
    hg_opts = team_goalie_options(goalie_stats, home_team)
    ag_opts = team_goalie_options(goalie_stats, away_team)

    tab1, tab2 = st.tabs([f"Home ({home_team})", f"Away ({away_team})"])

    with tab1:
        st.markdown("### OUT / scratches")
        home_out_ids = pick_ids("Home OUT", home_opts, 25, default_ids=[])

        st.markdown("### 5v5 Lines (5 players each)")
        home_lineup_5v5 = {
            "L1": pick_ids("Home L1", home_opts, 5, home_auto[0]["L1"]),
            "L2": pick_ids("Home L2", home_opts, 5, home_auto[0]["L2"]),
            "L3": pick_ids("Home L3", home_opts, 5, home_auto[0]["L3"]),
            "L4": pick_ids("Home L4", home_opts, 5, home_auto[0]["L4"]),
        }
        st.markdown("### Power Play (5on4)")
        home_lineup_pp = {
            "PP1": pick_ids("Home PP1", home_opts, 5, home_auto[1]["PP1"]),
            "PP2": pick_ids("Home PP2", home_opts, 5, home_auto[1]["PP2"]),
        }
        if hg_opts:
            labels = [x[0] for x in hg_opts]
            mp = {lab: pid for lab, pid in hg_opts}
            selg = st.selectbox("Home goalie", labels, index=0, key="home_goalie_pick")
            home_goalie_id = mp[selg]

    with tab2:
        st.markdown("### OUT / scratches")
        away_out_ids = pick_ids("Away OUT", away_opts, 25, default_ids=[])

        st.markdown("### 5v5 Lines (5 players each)")
        away_lineup_5v5 = {
            "L1": pick_ids("Away L1", away_opts, 5, away_auto[0]["L1"]),
            "L2": pick_ids("Away L2", away_opts, 5, away_auto[0]["L2"]),
            "L3": pick_ids("Away L3", away_opts, 5, away_auto[0]["L3"]),
            "L4": pick_ids("Away L4", away_opts, 5, away_auto[0]["L4"]),
        }
        st.markdown("### Power Play (5on4)")
        away_lineup_pp = {
            "PP1": pick_ids("Away PP1", away_opts, 5, away_auto[1]["PP1"]),
            "PP2": pick_ids("Away PP2", away_opts, 5, away_auto[1]["PP2"]),
        }
        if ag_opts:
            labels = [x[0] for x in ag_opts]
            mp = {lab: pid for lab, pid in ag_opts}
            selg = st.selectbox("Away goalie", labels, index=0, key="away_goalie_pick")
            away_goalie_id = mp[selg]


# =========================================================
# Validate goalies
# =========================================================
if home_goalie_id is None or away_goalie_id is None:
    st.error("Missing goalieId for home or away. Check goalies.csv for season/team.")
    st.stop()


# =========================================================
# Compute lambdas + minutes + props
# =========================================================
lam = matchup_lambda(
    home_team, away_team, team_stats, goalie_stats,
    int(home_goalie_id), int(away_goalie_id),
    minutes_5v5=MINUTES_5V5_TOTAL,
    minutes_pp=MINUTES_PP_TOTAL,
    k_seconds=K_SECONDS_SHRINK
)

try:
    home_roster, home_min5, home_minpp = build_projected_minutes_strict(
        skater_stats, home_team, home_lineup_5v5, home_lineup_pp, home_out_ids,
        MINUTES_5V5_TOTAL, MINUTES_PP_TOTAL,
        FLOOR_TOI_5V5_PG, FLOOR_TOI_PP_PG,
        K_SECONDS_SHRINK
    )
    away_roster, away_min5, away_minpp = build_projected_minutes_strict(
        skater_stats, away_team, away_lineup_5v5, away_lineup_pp, away_out_ids,
        MINUTES_5V5_TOTAL, MINUTES_PP_TOTAL,
        FLOOR_TOI_5V5_PG, FLOOR_TOI_PP_PG,
        K_SECONDS_SHRINK
    )
except Exception as e:
    st.error(f"Minutes projection error: {e}")
    st.stop()

home_raw = player_expected_points(skater_stats, home_team, home_roster, home_min5, home_minpp, K_SECONDS_SHRINK)
away_raw = player_expected_points(skater_stats, away_team, away_roster, away_min5, away_minpp, K_SECONDS_SHRINK)

home_markets = scale_players_to_team(home_raw, lam["lam_home"], POINTS_PER_GOAL)
away_markets = scale_players_to_team(away_raw, lam["lam_away"], POINTS_PER_GOAL)


# =========================================================
# Header
# =========================================================
st.subheader(f"{away_team} @ {home_team}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("λ_home", f"{lam['lam_home']:.2f}")
c2.metric("λ_away", f"{lam['lam_away']:.2f}")
c3.metric("Total λ", f"{lam['total']:.2f}")
c4.metric("P(home win) reg", f"{lam['p_home_win_reg']*100:.1f}%")
c5.metric("xG raw H/A", f"{lam['home_xg_raw']:.2f} / {lam['away_xg_raw']:.2f}")

st.caption(
    f"Minutes check (player-min): "
    f"Home 5v5={sum(home_min5.values()):.1f}/{MINUTES_5V5_TOTAL*5:.1f}, PP={sum(home_minpp.values()):.1f}/{MINUTES_PP_TOTAL*5:.1f} | "
    f"Away 5v5={sum(away_min5.values()):.1f}/{MINUTES_5V5_TOTAL*5:.1f}, PP={sum(away_minpp.values()):.1f}/{MINUTES_PP_TOTAL*5:.1f}"
)


# =========================================================
# Value section
# =========================================================
st.markdown("## Value (tu rentres les cotes ici)")
market = st.selectbox("Marché", ["1+ point", "2+ points"], index=0)

k_home = odds_key(SEASON, home_team, away_team, market, "HOME")
k_away = odds_key(SEASON, home_team, away_team, market, "AWAY")

init_odds_state(k_home, home_markets)
init_odds_state(k_away, away_markets)

with st.form("odds_form", clear_on_submit=False):
    left, right = st.columns(2)

    with left:
        st.markdown(f"### {home_team} — cotes")
        edited_home = st.data_editor(
            st.session_state[k_home],
            key=f"editor::{k_home}",
            use_container_width=True,
            height=420,
            column_config={
                "odds_decimal": st.column_config.NumberColumn("odds_decimal (EU)", min_value=1.01, step=0.01, format="%.2f"),
                "E_points": st.column_config.NumberColumn("E_points", format="%.4f"),
                "P1": st.column_config.NumberColumn("P(≥1)", format="%.4f"),
                "P2": st.column_config.NumberColumn("P(≥2)", format="%.4f"),
            },
            disabled=["playerId","name","team","position","min_proj_5v5","min_proj_PP","E_points","P1","P2"],
        )

    with right:
        st.markdown(f"### {away_team} — cotes")
        edited_away = st.data_editor(
            st.session_state[k_away],
            key=f"editor::{k_away}",
            use_container_width=True,
            height=420,
            column_config={
                "odds_decimal": st.column_config.NumberColumn("odds_decimal (EU)", min_value=1.01, step=0.01, format="%.2f"),
                "E_points": st.column_config.NumberColumn("E_points", format="%.4f"),
                "P1": st.column_config.NumberColumn("P(≥1)", format="%.4f"),
                "P2": st.column_config.NumberColumn("P(≥2)", format="%.4f"),
            },
            disabled=["playerId","name","team","position","min_proj_5v5","min_proj_PP","E_points","P1","P2"],
        )

    submitted = st.form_submit_button("✅ Appliquer les cotes et recalculer")

if submitted:
    st.session_state[k_home] = edited_home
    st.session_state[k_away] = edited_away

vhome = compute_value(st.session_state[k_home], market)
vaway = compute_value(st.session_state[k_away], market)

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Best edges (Home)")
    st.dataframe(vhome.head(25), use_container_width=True, height=380)
with c2:
    st.markdown("#### Best edges (Away)")
    st.dataframe(vaway.head(25), use_container_width=True, height=380)


# =========================================================
# Props display
# =========================================================
st.markdown("## Player point props (model) — STRICT roster")

a, b = st.columns(2)
with a:
    st.markdown(f"### {home_team}")
    show = home_markets[["playerId","name","position","min_proj_5v5","min_proj_PP","E_points","Over0.5_prob","Over1.5_prob"]].copy()
    show = show.sort_values("Over0.5_prob", ascending=False)
    st.dataframe(show, use_container_width=True, height=520)

with b:
    st.markdown(f"### {away_team}")
    show = away_markets[["playerId","name","position","min_proj_5v5","min_proj_PP","E_points","Over0.5_prob","Over1.5_prob"]].copy()
    show = show.sort_values("Over0.5_prob", ascending=False)
    st.dataframe(show, use_container_width=True, height=520)

with st.expander("Debug"):
    st.write({
        "skater_icetime_unit_detected": skater_stats.attrs.get("icetime_unit_skater", "?"),
        "goalie_icetime_unit_detected": goalie_stats.attrs.get("icetime_unit_goalie", "?"),
        "home_out_ids": home_out_ids,
        "away_out_ids": away_out_ids,
        "home_roster_count": len(home_roster),
        "away_roster_count": len(away_roster),
        "data_folder": str(BASE.resolve()),
    })
