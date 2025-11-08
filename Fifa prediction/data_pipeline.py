"""
data_pipeline.py (final version using Kaggle WorldCupMatches.csv)
----------------------------------------------------------------
This script builds a clean, team-level dataset from:
1. Kaggle WorldCupMatches.csv  (match-level data)
2. WorldCupTeams.csv           (team metadata: confederation, appearances)
3. worldcup_scraped_placeholder.csv  (scraped features like avg_age, fifa_rank, etc.)

Output: data/clean/cleaned_worldcup_dataset.csv
"""

import pandas as pd
import numpy as np
import os

# ===============================
# Paths (update as needed)
# ===============================
RAW_MATCH_PATH = "D:/aiml/assignment2/v3/data/raw/WorldCupMatches.csv"
SCRAPED_PATH = "D:/aiml/assignment2/v3/data/raw/wiki_worldcup_squads_2022.csv"
TEAMS_PATH = "D:/aiml/assignment2/v3/data/raw/WorldCupTeams.csv"
OUTPUT_PATH = "D:/aiml/assignment2/v3/data/clean/cleaned_worldcup_dataset.csv"


# ===============================
# Functions
# ===============================
def load_data():
    """Load match, scraped, and team datasets."""
    matches = pd.read_csv(RAW_MATCH_PATH)
    scraped = pd.read_csv(SCRAPED_PATH)
    teams = pd.read_csv(TEAMS_PATH)
    return matches, scraped, teams


def clean_matches(matches):
    """Basic cleaning of the Kaggle WorldCupMatches.csv file."""
    # Remove duplicates and clean missing team names
    matches = matches.drop_duplicates()
    matches = matches.dropna(subset=["Home Team Name", "Away Team Name"])
    
    # Rename important columns for clarity
    matches = matches.rename(
        columns={
            "Year": "year",
            "Home Team Name": "home_team",
            "Away Team Name": "away_team",
            "Home Team Goals": "home_goals",
            "Away Team Goals": "away_goals",
            "Win conditions": "win_conditions"
        }
    )
    return matches


def engineer_features(matches, scraped, teams):
    """
    Aggregate team-level features:
    - goals_for, goals_against, matches_played, goal_diff, win_rate
    Merge scraped data and team metadata.
    """
    # Aggregate stats for home teams
    home_stats = (
        matches.groupby(["year", "home_team"])
        .agg(
            goals_for=("home_goals", "sum"),
            goals_against=("away_goals", "sum"),
            matches_played=("home_team", "count")
        )
        .reset_index()
        .rename(columns={"home_team": "team"})
    )

    # Aggregate stats for away teams
    away_stats = (
        matches.groupby(["year", "away_team"])
        .agg(
            goals_for=("away_goals", "sum"),
            goals_against=("home_goals", "sum"),
            matches_played=("away_team", "count")
        )
        .reset_index()
        .rename(columns={"away_team": "team"})
    )

    # Combine home + away stats
    combined = pd.concat([home_stats, away_stats])
    team_stats = (
        combined.groupby(["year", "team"])
        .agg(
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            matches_played=("matches_played", "sum")
        )
        .reset_index()
    )

    # Add goal difference
    team_stats["goal_diff"] = team_stats["goals_for"] - team_stats["goals_against"]

    # Estimate win rate
    if "win_conditions" in matches.columns:
        winners = matches[matches["win_conditions"].notnull()]
        winners["winner_team"] = winners["win_conditions"].str.split("after").str[0].str.strip()
        win_count = winners.groupby(["year", "winner_team"]).size().reset_index(name="wins")
        win_count = win_count.rename(columns={"winner_team": "team"})
        team_stats = team_stats.merge(win_count, how="left", on=["year", "team"])
        team_stats["wins"] = team_stats["wins"].fillna(0)
        team_stats["win_rate"] = team_stats["wins"] / team_stats["matches_played"]
    else:
        team_stats["win_rate"] = np.nan

    # Merge scraped (avg_age, fifa_rank, avg_caps)
    if not scraped.empty:
        scraped = scraped.rename(columns={"tournament_year": "year"})
        team_stats = team_stats.merge(scraped, how="left", on=["year", "team"])

    # Merge team metadata (confederation, appearances)
    if not teams.empty:
        team_stats = team_stats.merge(
            teams[["team", "confederation", "appearances"]],
            how="left",
            on="team"
        )

    return team_stats


def main():
    os.makedirs("data/clean", exist_ok=True)
    matches, scraped, teams = load_data()
    matches = clean_matches(matches)
    final_df = engineer_features(matches, scraped, teams)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Cleaned dataset saved to {OUTPUT_PATH}")
    print(f"Rows: {len(final_df)}, Columns: {len(final_df.columns)}")
    print(final_df.head(10))


if __name__ == "__main__":
    main()
