"""
data.py
Pulls StatsBomb open data (from local clone) and builds a match-outcome dataset.
"""

import json
from pathlib import Path

import pandas as pd

# Adjust this if your clone lives somewhere else
OPEN_DATA_DIR = Path("data/raw/open-data/data")


def get_available_competitions() -> pd.DataFrame:
    """See what competitions/seasons are actually available before picking a scope."""
    with open(OPEN_DATA_DIR / "competitions.json") as f:
        comps = pd.DataFrame(json.load(f))
    return comps


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Pull all matches for a competition/season from the local clone."""
    path = OPEN_DATA_DIR / "matches" / str(competition_id) / f"{season_id}.json"
    with open(path) as f:
        raw = json.load(f)

    matches = pd.json_normalize(raw)

    # Normalise the nested team name fields to flat columns matching
    # what the rest of the pipeline expects
    matches = matches.rename(columns={
        "home_team.home_team_name": "home_team",
        "away_team.away_team_name": "away_team",
    })

    matches["match_date"] = pd.to_datetime(matches["match_date"])

    keep_cols = [
        "match_id", "match_date", "home_team", "away_team",
        "home_score", "away_score", "competition_stage.name",
    ]
    matches = matches[[c for c in keep_cols if c in matches.columns]]

    return matches


def add_result_column(matches: pd.DataFrame) -> pd.DataFrame:
    """Add a target column: H (home win), A (away win), D (draw)."""
    def result(row):
        if row["home_score"] > row["away_score"]:
            return "H"
        elif row["home_score"] < row["away_score"]:
            return "A"
        else:
            return "D"

    matches["result"] = matches.apply(result, axis=1)
    return matches


def _team_matches(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """All matches involving a team, with a normalised team/opponent/GF/GA view."""
    home = matches[matches["home_team"] == team].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["gf"] = home["home_score"]
    home["ga"] = home["away_score"]
    home["venue"] = "H"

    away = matches[matches["away_team"] == team].copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["gf"] = away["away_score"]
    away["ga"] = away["home_score"]
    away["venue"] = "A"

    combined = pd.concat([home, away]).sort_values("match_date")
    return combined


def _points(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    elif gf == ga:
        return 1
    return 0


def _form_features(team_history: pd.DataFrame, before_date, n_games: int) -> dict:
    """Compute rolling form from a team's history, using only matches before `before_date`."""
    past = team_history[team_history["match_date"] < before_date].tail(n_games)

    if len(past) == 0:
        return {
            "form_points_avg": None,
            "form_goals_for_avg": None,
            "form_goals_against_avg": None,
            "form_goal_diff_avg": None,
            "form_win_rate": None,
            "form_sample_size": 0,
        }

    points = [_points(row.gf, row.ga) for row in past.itertuples()]

    return {
        "form_points_avg": sum(points) / len(past),
        "form_goals_for_avg": past["gf"].mean(),
        "form_goals_against_avg": past["ga"].mean(),
        "form_goal_diff_avg": (past["gf"] - past["ga"]).mean(),
        "form_win_rate": sum(1 for p in points if p == 3) / len(past),
        "form_sample_size": len(past),
    }


def build_team_form(matches: pd.DataFrame, n_games: int = 5) -> pd.DataFrame:
    """
    For every match, and for each team in it (home + away):
      1. Find that team's previous matches (home or away), sorted by date,
         limited to matches strictly BEFORE the current match date.
      2. Take the last n_games of those.
      3. Compute: points won, goals scored, goals conceded, goal difference,
         win rate — all from that window only.
    If a team has fewer than n_games of history (e.g. early season), use
    whatever is available and flag it with a `form_sample_size` column so
    you can decide later whether to drop or keep those rows.
    """
    matches = matches.sort_values("match_date").reset_index(drop=True)

    all_teams = pd.concat([matches["home_team"], matches["away_team"]]).unique()
    history_by_team = {team: _team_matches(matches, team) for team in all_teams}

    home_rows = []
    away_rows = []

    for row in matches.itertuples():
        home_form = _form_features(history_by_team[row.home_team], row.match_date, n_games)
        away_form = _form_features(history_by_team[row.away_team], row.match_date, n_games)

        home_rows.append({f"home_{k}": v for k, v in home_form.items()})
        away_rows.append({f"away_{k}": v for k, v in away_form.items()})

    home_df = pd.DataFrame(home_rows)
    away_df = pd.DataFrame(away_rows)

    result = pd.concat([matches.reset_index(drop=True), home_df, away_df], axis=1)
    return result


def save_dataset(df: pd.DataFrame, path: str = "data/processed/matches.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


if __name__ == "__main__":
    comps = get_available_competitions()
    print(comps[["competition_id", "season_id", "competition_name", "season_name"]])

    # Once you've picked one from the printed list, plug the IDs in here:
    # matches = get_matches(competition_id=..., season_id=...)
    # matches = add_result_column(matches)
    # matches = build_team_form(matches)
    # save_dataset(matches)