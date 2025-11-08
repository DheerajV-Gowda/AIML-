import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Helper Function for Knockout Prediction
# ---------------------------------------------------------------------
def predict_match_winner(model, teamA, teamB, df_features):
    """
    Predicts the winner between two teams based on the higher predicted finalist probability.
    """
    probA = model.predict_proba(df_features[df_features["team"] == teamA].drop(columns=["team"]))[:, 1][0]
    probB = model.predict_proba(df_features[df_features["team"] == teamB].drop(columns=["team"]))[:, 1][0]
    return teamA if probA > probB else teamB
# ---------------------------------------------------------------------
# Unified Knockout Simulation: Prints & Saves Results
# ---------------------------------------------------------------------
def simulate_knockout(model, df_features, teams, output_dir):
    """
    Simulate knockout stage (Round of 16 → Final).
    Prints results and saves all matches + final summary to CSV.
    """
    rounds = ["Round of 16", "Quarterfinals", "Semifinals", "Final"]
    stage_teams = teams.copy()
    round_no = 0
    all_results = []

    print("\n🏆 Simulating Knockout Stage...\n")

    while len(stage_teams) > 1:
        round_name = rounds[round_no]
        next_round = []
        print(f"🔹 {round_name}")
        for i in range(0, len(stage_teams), 2):
            t1, t2 = stage_teams[i], stage_teams[i + 1]
            prob1 = model.predict_proba(df_features[df_features["team"] == t1].drop(columns=["team"]))[:, 1][0]
            prob2 = model.predict_proba(df_features[df_features["team"] == t2].drop(columns=["team"]))[:, 1][0]
            winner = t1 if prob1 > prob2 else t2
            print(f"{t1} vs {t2} → Winner: {winner}")
            all_results.append({
                "Round": round_name,
                "Match": f"{t1} vs {t2}",
                "Winner": winner,
                "Prob_Team1": round(prob1, 3),
                "Prob_Team2": round(prob2, 3)
            })
            next_round.append(winner)
        stage_teams = next_round
        round_no += 1
        print("")

    # Final winner
    final_winner = stage_teams[0]
    print(f"🏆 Predicted Champion: {final_winner}\n")

    # Save detailed knockout predictions
    knockout_df = pd.DataFrame(all_results)
    knockout_path = output_dir / "knockout_predictions.csv"
    knockout_df.to_csv(knockout_path, index=False)
    print(f"💾 Knockout match predictions saved → {knockout_path}")

    # Save summary with champion flag for the app
    final_results = df_features[["team", "probability_finalist"]].copy()
    final_results["predicted_champion"] = (final_results["team"] == final_winner).astype(int)
    final_results.to_csv(output_dir / "predictions.csv", index=False)
    print(f"✅ Final summary saved → {output_dir / 'predictions.csv'}")

    return final_winner

# 1️⃣  Load and clean match data
# ---------------------------------------------------------------------
RAW_MATCHES = Path("D:/aiml/assignment2/v3/data/raw/WorldCupMatches.csv")
RAW_TEAMS   = Path("D:/aiml/assignment2/v3/data/raw/WorldCupTeams.csv")
RAW_WIKI    = Path("D:/aiml/assignment2/v3/data/raw/wiki_worldcup_squads_2022.csv")
OUT_DIR     = Path("D:/aiml/assignment2/v3/data/clean/task2_outputs/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

matches = pd.read_csv(RAW_MATCHES)
matches = matches[matches["Year"].between(1998, 2022)]
matches = matches.rename(columns={
    "Year": "year",
    "Home Team Name": "home_team",
    "Away Team Name": "away_team",
    "Home Team Goals": "home_goals",
    "Away Team Goals": "away_goals",
    "Stage": "stage"
})

# Home / Away aggregates
home = (
    matches.groupby(["year", "home_team"])
    .agg(goals_for_home=("home_goals", "sum"),
         goals_against_home=("away_goals", "sum"),
         matches_home=("home_team", "count"))
    .reset_index()
    .rename(columns={"home_team": "team"})
)

away = (
    matches.groupby(["year", "away_team"])
    .agg(goals_for_away=("away_goals", "sum"),
         goals_against_away=("home_goals", "sum"),
         matches_away=("away_team", "count"))
    .reset_index()
    .rename(columns={"away_team": "team"})
)

combined = pd.concat([home, away], ignore_index=True)
team_stats = (
    combined.groupby(["year", "team"])
    .agg(goals_for=("goals_for_home", "sum"),
         goals_against=("goals_against_home", "sum"),
         matches_played=("matches_home", "sum"))
    .reset_index()
)
team_stats["goal_diff"] = team_stats["goals_for"] - team_stats["goals_against"]

# Compute wins
def match_winner(row):
    if row["home_goals"] > row["away_goals"]:
        return row["home_team"]
    elif row["home_goals"] < row["away_goals"]:
        return row["away_team"]
    else:
        return None

matches["winner"] = matches.apply(match_winner, axis=1)
wins = (
    matches.dropna(subset=["winner"])
    .groupby(["year", "winner"]).size()
    .reset_index(name="wins")
    .rename(columns={"winner": "team"})
)
team_stats = team_stats.merge(wins, how="left", on=["year", "team"])
team_stats["wins"] = team_stats["wins"].fillna(0)
team_stats["win_rate"] = team_stats["wins"] / team_stats["matches_played"]

# ---------------------------------------------------------------------
# 2️⃣  Merge metadata (teams, wiki squads)
# ---------------------------------------------------------------------
teams_meta = pd.read_csv(RAW_TEAMS)
team_stats = team_stats.merge(
    teams_meta[["team", "confederation", "appearances"]],
    how="left", on="team"
)

if RAW_WIKI.exists():
    wiki = pd.read_csv(RAW_WIKI)
    # Clean up potential spaces or case mismatches
    wiki.columns = wiki.columns.str.strip().str.lower()
    wiki["team"] = wiki["team"].str.strip()
    wiki["year"] = wiki["year"].astype(int)

    print("Loaded squad data:", wiki.shape)
    print("Years in squad file:", wiki["year"].unique())

    # Merge safely
    team_stats = team_stats.merge(
        wiki[["team", "players_listed", "year"]],
        how="left", on=["team", "year"]
    )

    print("Matched teams with players_listed:",
          team_stats["players_listed"].notna().sum())
else:
    team_stats["players_listed"] = np.nan


# ---------------------------------------------------------------------
# 3️⃣  Label creation: finalists
# ---------------------------------------------------------------------
finals = matches[matches["stage"].str.contains("Final", case=False, na=False)]
final_pairs = (
    finals.groupby("year")
    .apply(lambda d: list(pd.unique(d[["home_team", "away_team"]].values.ravel())))
    .to_dict()
)
team_stats["is_finalist"] = team_stats.apply(
    lambda r: 1 if r["team"] in final_pairs.get(r["year"], []) else 0,
    axis=1
)

# Fill missing values
team_stats["players_listed"] = (
    team_stats.groupby("year")["players_listed"].transform(lambda x: x.fillna(x.median()))
)
team_stats["players_listed"] = team_stats["players_listed"].fillna(team_stats["players_listed"].median())
team_stats["confederation"] = team_stats["confederation"].fillna("OTHER")
team_stats = team_stats.replace([np.inf, -np.inf], np.nan)
team_stats = team_stats.fillna({"goal_diff": 0, "win_rate": 0, "appearances": 0})

# Save cleaned dataset
cleaned_path = OUT_DIR / "cleaned_worldcup_dataset_for_modeling.csv"
team_stats.to_csv(cleaned_path, index=False)
print(f"✅ Cleaned dataset saved → {cleaned_path}")

# ---------------------------------------------------------------------
# 4️⃣  Model training
# ---------------------------------------------------------------------
features = ["goal_diff", "win_rate", "appearances", "players_listed", "confederation"]
X = team_stats[features]
y = team_stats["is_finalist"]

numeric_feats = ["goal_diff", "win_rate", "appearances", "players_listed"]
cat_feats = ["confederation"]

numeric_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transform, numeric_feats),
    ("cat", cat_transform, cat_feats)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe_lr = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])
pipe_rf = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
])

pipe_lr.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 5️⃣  Evaluation
# ---------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba),
        y_pred=y_pred,
        y_proba=y_proba
    )

res_lr = evaluate(pipe_lr, X_test, y_test)
res_rf = evaluate(pipe_rf, X_test, y_test)

metrics = pd.DataFrame([
    {"model": "LogisticRegression", **{k: res_lr[k] for k in ["accuracy","precision","recall","f1","roc_auc"]}},
    {"model": "RandomForest", **{k: res_rf[k] for k in ["accuracy","precision","recall","f1","roc_auc"]}},
])
metrics_path = OUT_DIR / "model_metrics.csv"
metrics.to_csv(metrics_path, index=False)
print(metrics)

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (res, title) in zip(axes, [(res_lr, "Logistic Regression"), (res_rf, "Random Forest")]):
    cm = confusion_matrix(y_test, res["y_pred"])
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrices.png")

# ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, res_lr["y_proba"])
fpr_rf, tpr_rf, _ = roc_curve(y_test, res_rf["y_proba"])
plt.figure(figsize=(6, 5))
plt.plot(fpr_lr, tpr_lr, label=f"LogReg AUC={res_lr['roc_auc']:.2f}")
plt.plot(fpr_rf, tpr_rf, label=f"RF AUC={res_rf['roc_auc']:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend()
plt.savefig(OUT_DIR / "roc_curves.png")

# ---------------------------------------------------------------------
# 6️⃣  Save models & predictions
# ---------------------------------------------------------------------
joblib.dump(pipe_lr, OUT_DIR / "model_logreg.joblib")
joblib.dump(pipe_rf, OUT_DIR / "model_rf.joblib")

preds = X_test.copy()
preds["y_true"] = y_test.values
preds["pred_logreg"] = res_lr["y_pred"]
preds["prob_logreg"] = res_lr["y_proba"]
preds["pred_rf"] = res_rf["y_pred"]
preds["prob_rf"] = res_rf["y_proba"]
preds.to_csv(OUT_DIR / "test_predictions.csv", index=False)

# ---------------------------------------------------------------------
# 7️⃣  Feature importances (Random Forest)
# ---------------------------------------------------------------------
pre = pipe_rf.named_steps["pre"]
num_feats = numeric_feats
ohe = pre.named_transformers_["cat"].named_steps["ohe"]
ohe_feats = list(ohe.get_feature_names_out(["confederation"]))
feature_names = num_feats + ohe_feats
importances = pipe_rf.named_steps["clf"].feature_importances_
feat_imp = pd.DataFrame({"feature": feature_names[:len(importances)], "importance": importances})
feat_imp.sort_values("importance", ascending=False).to_csv(OUT_DIR / "feature_importances.csv", index=False)

print(f"\n✅ All outputs saved in: {OUT_DIR.resolve()}")


# ---------------------------------------------------------------------
# 8️⃣  Predict FIFA World Cup 2026 Finalists using trained Random Forest
# ---------------------------------------------------------------------
print("\n🔮 Predicting potential 2026 finalists using trained Random Forest...")

teams_2026 = [
    # UEFA (8)
    "France", "Germany", "Spain", "Portugal", "England", "Netherlands", "Croatia", "Italy",
    # CONMEBOL (5)
    "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador",
    # CONCACAF (4)
    "Mexico", "United States", "Canada", "Costa Rica",
    # CAF (4)
    "Morocco", "Senegal", "Nigeria", "Egypt",
    # AFC (4)
    "Japan", "South Korea", "Iran", "Australia"
]  # ✅ total 25 teams

goal_diff = [20, 18, 17, 15, 16, 14, 12, 13, 22, 19, 11, 10, 8, 9, 7, 6, 5, 4, 6, 4, 7, 5, 5, 4, 4]
win_rate = [0.80, 0.75, 0.78, 0.76, 0.72, 0.70, 0.68, 0.70, 0.82, 0.81, 0.73, 0.70, 0.68,
            0.70, 0.68, 0.66, 0.65, 0.63, 0.64, 0.60, 0.70, 0.68, 0.66, 0.64, 0.63]
appearances = [16, 20, 16, 8, 16, 11, 6, 19, 18, 22, 14, 6, 17, 11, 2, 6, 6, 3, 7, 3, 7, 11, 6, 6, 6]
players_listed = [26] * 25
confederations = [
    "UEFA", "UEFA", "UEFA", "UEFA", "UEFA", "UEFA", "UEFA", "UEFA",
    "CONMEBOL", "CONMEBOL", "CONMEBOL", "CONMEBOL", "CONMEBOL",
    "CONCACAF", "CONCACAF", "CONCACAF", "CONCACAF",
    "CAF", "CAF", "CAF", "CAF",
    "AFC", "AFC", "AFC", "AFC"
]  # ✅ 25 entries total

# Check counts for safety
print("List lengths ->", len(teams_2026), len(goal_diff), len(win_rate), len(appearances), len(players_listed), len(confederations))

# Create DataFrame
df_2026 = pd.DataFrame({
    "team": teams_2026,
    "goal_diff": goal_diff,
    "win_rate": win_rate,
    "appearances": appearances,
    "players_listed": players_listed,
    "confederation": confederations
})

# Predict using the trained Random Forest model
df_2026["probability_finalist"] = pipe_rf.predict_proba(df_2026)[:, 1]
df_2026_sorted = df_2026.sort_values("probability_finalist", ascending=False)

print("\n🏆 Predicted probability of reaching the 2026 World Cup Final:")
print(df_2026_sorted[["team", "probability_finalist"]].head(10))

pred_2026_path = OUT_DIR / "predicted_finalists_2026.csv"
df_2026_sorted.to_csv(pred_2026_path, index=False)
print(f"\n✅ 2026 finalist predictions saved to: {pred_2026_path}")

# ---------------------------------------------------------------------
# Knockout Stage Simulation
# ---------------------------------------------------------------------
top16_teams = list(df_2026_sorted["team"].head(16))
print(f"\n⚽ Selected Top 16 Teams for Knockout Simulation: {top16_teams}")

champion = simulate_knockout(pipe_rf, df_2026_sorted, top16_teams, OUT_DIR)
print(f"\n🏆 Predicted 2026 FIFA World Cup Champion: {champion}")
