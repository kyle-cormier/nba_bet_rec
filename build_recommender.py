"""
NBA Betting Recommender with team affinity + recency features:
- Creates labels via negative sampling
- Engineers user, event, team-affinity, and recency features (no leakage)
- Trains a fast baseline model
- Exports per-user scores and Top-3 recommendations (validation period)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib


# ---------------------------
# 0) Paths / Config
# ---------------------------
TRAIN_PATH = "/Users/Kyle/Documents/nba_bet_rec/df_train.xlsx"
VAL_PATH = "/Users/Kyle/Documents/nba_bet_rec/df_validation.xlsx"
VENDOR_SCHEDULE_PATH = "/Users/Kyle/Documents/nba_bet_rec/playoff_schedule.csv"

OUT_DIR = Path("/Users/Kyle/Documents/nba_bet_rec/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "bet_recommender_model.pkl"
METRICS_PATH = OUT_DIR / "model_classification_report.txt"
VAL_SCORES_PATH = OUT_DIR / "validation_all_scores.csv"
TOP3_PATH = OUT_DIR / "validation_top3_recommendations.csv"

NEGATIVE_RATIO_TRAIN = 2
NEGATIVE_RATIO_VAL = 2
MAX_NEGS_PER_USER = 60
RANDOM_STATE_TRAIN = 42
RANDOM_STATE_VAL = 777

# ✅ Aggregation method for duplicate user/day/event scores ("mean" or "max")
AGG_METHOD = "max"


# ---------------------------
# 1) Load data
# ---------------------------
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["betdate"] = pd.to_datetime(df["betdate"], errors="coerce")
    df["wager_amount"] = pd.to_numeric(df.get("wager_amount", 0), errors="coerce").fillna(0.0)
    return df

df_train = load_df(TRAIN_PATH)
df_val = load_df(VAL_PATH)

def parse_event_description(event_str):
    try:
        if isinstance(event_str, str) and "@" in event_str:
            away, home = event_str.split("@", 1)
            return away.strip(), home.strip()
    except Exception:
        pass
    return None, None

for df in (df_train, df_val):
    teams = df["event_description"].apply(lambda x: pd.Series(parse_event_description(x)))
    teams.columns = ["away_team", "home_team"]
    df[["away_team", "home_team"]] = teams


# ---------------------------
# 2) Build candidate datasets
# ---------------------------
def build_candidate_dataset(df_bets, schedule_df,
                            negative_ratio=2,
                            max_negs_per_user=60,
                            random_state=42):
    rng = np.random.default_rng(seed=random_state)

    df_pos = df_bets[["mask_id", "event_description", "betdate", "wager_amount"]].copy()
    df_pos["label"] = 1

    sch = schedule_df.dropna(subset=["event_description"]).copy()
    sch["betdate"] = pd.to_datetime(sch["betdate"], errors="coerce")
    sch = sch.dropna(subset=["betdate"]).sort_values("betdate")
    sch = sch.groupby("event_description", as_index=False).first()

    all_games = sch["event_description"].unique().tolist()
    user_list = df_bets["mask_id"].dropna().unique().tolist()

    neg_rows = []
    for user in user_list:
        user_events = set(
            df_bets.loc[df_bets["mask_id"] == user, "event_description"].dropna().unique().tolist()
        )
        n_pos = max(1, len(user_events))
        pool = [g for g in all_games if g not in user_events]
        if not pool:
            continue
        n_neg = min(len(pool), negative_ratio * n_pos, max_negs_per_user)
        sampled = rng.choice(pool, size=n_neg, replace=False)
        tmp = sch.loc[sch["event_description"].isin(sampled), ["event_description", "betdate"]].copy()
        tmp["mask_id"] = user
        tmp["wager_amount"] = 0.0
        tmp["label"] = 0
        neg_rows.append(tmp[["mask_id", "event_description", "betdate", "wager_amount", "label"]])

    df_neg = pd.concat(neg_rows, ignore_index=True) if neg_rows else pd.DataFrame(
        columns=["mask_id", "event_description", "betdate", "wager_amount", "label"]
    )

    df_full = pd.concat([df_pos, df_neg], ignore_index=True).dropna(subset=["mask_id", "event_description"])
    df_full["betdate"] = pd.to_datetime(df_full["betdate"], errors="coerce")
    df_full["wager_amount"] = pd.to_numeric(df_full["wager_amount"], errors="coerce").fillna(0.0)
    df_full["label"] = df_full["label"].astype(int)

    # 🔑 Parse home/away teams again so they exist in candidate dataset
    teams = df_full["event_description"].apply(lambda x: pd.Series(parse_event_description(x)))
    teams.columns = ["away_team", "home_team"]
    df_full[["away_team", "home_team"]] = teams

    return df_full

schedule_train = df_train[["event_description", "betdate"]].drop_duplicates()
schedule_val = df_val[["event_description", "betdate"]].drop_duplicates()

df_train_labeled = build_candidate_dataset(df_train, schedule_train,
    negative_ratio=NEGATIVE_RATIO_TRAIN, max_negs_per_user=MAX_NEGS_PER_USER,
    random_state=RANDOM_STATE_TRAIN)
df_val_labeled = build_candidate_dataset(df_val, schedule_val,
    negative_ratio=NEGATIVE_RATIO_VAL, max_negs_per_user=MAX_NEGS_PER_USER,
    random_state=RANDOM_STATE_VAL)


# ---------------------------
# 3) Feature engineering
# ---------------------------
# User + event aggregates (from TRAIN only)
user_stats = df_train.groupby("mask_id")["wager_amount"].agg(
    user_mean_wager="mean",
    user_std_wager="std",
    user_max_wager="max",
    user_total_bets="count"
).reset_index()

event_stats = df_train.groupby("event_description")["wager_amount"].agg(
    event_total_wager="sum",
    event_mean_wager="mean",
    event_num_bets="count"
).reset_index()

# --- Team affinity features ---
df_team_long = df_train.melt(
    id_vars=["mask_id", "wager_amount", "betdate"],
    value_vars=["home_team", "away_team"],
    value_name="team"
).drop(columns="variable")

team_counts = df_team_long.groupby(["mask_id", "team"]).size().reset_index(name="user_team_bet_count")
team_avg_wager = df_team_long.groupby(["mask_id", "team"])["wager_amount"].mean().reset_index(name="user_team_avg_wager")
user_team_stats = team_counts.merge(team_avg_wager, on=["mask_id","team"], how="outer")

# --- Recency features (corrected) ---
df_train_sorted = df_train.sort_values(["mask_id", "betdate"])
df_train_sorted["days_since_last_bet"] = (
    df_train_sorted.groupby("mask_id")["betdate"].diff().dt.days.fillna(9999)
)

def count_recent(series, window_days):
    counts = []
    for i, d in enumerate(series):
        start = d - pd.Timedelta(days=window_days)
        counts.append(((series[:i] >= start) & (series[:i] <= d)).sum())
    return pd.Series(counts, index=series.index)

df_train_sorted["user_recent_bets_7d"] = (
    df_train_sorted.groupby("mask_id")["betdate"].transform(lambda x: count_recent(x, 7))
)
df_train_sorted["user_recent_bets_30d"] = (
    df_train_sorted.groupby("mask_id")["betdate"].transform(lambda x: count_recent(x, 30))
)

user_last_seen = df_train_sorted.groupby("mask_id").agg(
    last_bet_date=("betdate", "max"),
    avg_days_between=("days_since_last_bet","mean"),
    recent_bets_7d=("user_recent_bets_7d","max"),
    recent_bets_30d=("user_recent_bets_30d","max")
).reset_index()


# ---------------------------
# Feature joiner
# ---------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = (df
        .merge(user_stats, on="mask_id", how="left")
        .merge(event_stats, on="event_description", how="left")
        .copy()
    )
    out["day_of_week"] = out["betdate"].dt.dayofweek
    out["month"] = out["betdate"].dt.month

    # Merge team affinity
    out = out.merge(user_team_stats.add_prefix("home_"),
                    left_on=["mask_id","home_team"],
                    right_on=["home_mask_id","home_team"],
                    how="left")
    out = out.merge(user_team_stats.add_prefix("away_"),
                    left_on=["mask_id","away_team"],
                    right_on=["away_mask_id","away_team"],
                    how="left")

    # Merge recency
    out = out.merge(user_last_seen, on="mask_id", how="left")
    out["days_since_last_bet"] = (out["betdate"] - out["last_bet_date"]).dt.days.fillna(9999)

    # Fill missing
    for col in [
        "user_mean_wager","user_std_wager","user_max_wager","user_total_bets",
        "event_total_wager","event_mean_wager","event_num_bets",
        "home_user_team_bet_count","home_user_team_avg_wager",
        "away_user_team_bet_count","away_user_team_avg_wager",
        "days_since_last_bet","avg_days_between","recent_bets_7d","recent_bets_30d"
    ]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)

    return out

df_train_fe = add_features(df_train_labeled)
df_val_fe = add_features(df_val_labeled)

FEATURE_COLS = [
    "user_mean_wager","user_std_wager","user_max_wager","user_total_bets",
    "event_total_wager","event_mean_wager","event_num_bets",
    "day_of_week","month",
    "home_user_team_bet_count","home_user_team_avg_wager",
    "away_user_team_bet_count","away_user_team_avg_wager",
    "days_since_last_bet","avg_days_between","recent_bets_7d","recent_bets_30d"
]

X_train = df_train_fe[FEATURE_COLS].values
y_train = df_train_fe["label"].astype(int).values
X_val = df_val_fe[FEATURE_COLS].values
y_val = df_val_fe["label"].astype(int).values


# ---------------------------
# 4) Train baseline
# ---------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=300, class_weight="balanced", solver="liblinear", random_state=42
    ))
])
pipe.fit(X_train, y_train)


# ---------------------------
# 5) Evaluate + Export
# ---------------------------
val_prob = pipe.predict_proba(X_val)[:, 1]
val_pred = (val_prob >= 0.5).astype(int)

auc = roc_auc_score(y_val, val_prob)
report_text = classification_report(y_val, val_pred)

joblib.dump(pipe, MODEL_PATH)
with open(METRICS_PATH, "w") as f:
    f.write(f"Validation AUC: {auc:.4f}\n\n")
    f.write(report_text)

# Base scored validation set
val_scored = df_val_fe[["mask_id","event_description","betdate","label"]].copy()
val_scored["betdate"] = pd.to_datetime(val_scored["betdate"], errors="coerce").dt.date
val_scored["score"] = val_prob.round(8)

# ✅ Collapse duplicates on user/day/event with configurable agg method
collapsed = (
    val_scored.groupby(["mask_id","betdate","event_description"], as_index=False)
    .agg({
        "score": AGG_METHOD,
        "label": "max"
    })
)

# ✅ Assign ranks per user/day
collapsed = collapsed.sort_values(["mask_id","betdate","score"], ascending=[True, True, False])
collapsed["rank"] = collapsed.groupby(["mask_id","betdate"]).cumcount() + 1

# Save full scored validation set (with ranks)
collapsed.to_csv(VAL_SCORES_PATH, index=False)

# ✅ Top-3 per user-day only
top3 = collapsed[collapsed["rank"] <= 3].copy()
top3.to_csv(TOP3_PATH, index=False)

print("---- SUMMARY ----")
print("Train rows:", df_train_fe.shape[0], "| Val rows:", df_val_fe.shape[0])
print("Users (train):", df_train["mask_id"].nunique(), "| Users (val):", df_val["mask_id"].nunique())
print("Events (train):", df_train["event_description"].nunique(), "| Events (val):", df_val["event_description"].nunique())
print(f"Validation AUC: {auc:.4f}")
print("\nArtifacts:")
print("Model:", MODEL_PATH)
print("Metrics:", METRICS_PATH)
print("All scores CSV:", VAL_SCORES_PATH)
print("Top-3 CSV:", TOP3_PATH)


# ---------------------------
# 6) Precision@3 / Recall@3
# ---------------------------
def precision_recall_at_k(df, k=3):
    results = []
    for user, group in df.groupby("mask_id"):
        group = group.sort_values("score", ascending=False)
        topk = group.head(k)
        prec = topk["label"].sum() / k
        actual_pos = group["label"].sum()
        rec = topk["label"].sum() / actual_pos if actual_pos > 0 else np.nan
        results.append((prec, rec))
    results = pd.DataFrame(results, columns=["precision", "recall"])
    return results["precision"].mean(), results["recall"].mean()

p_at3, r_at3 = precision_recall_at_k(collapsed, k=3)

print("---- RECOMMENDER METRICS ----")
print(f"Precision@3: {p_at3:.3f}")
print(f"Recall@3:    {r_at3:.3f}")