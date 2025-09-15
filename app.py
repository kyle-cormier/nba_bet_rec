import streamlit as st
import pandas as pd
import joblib
import openai
from datetime import date, timedelta
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="🏀 NBA Betting Recommender", layout="wide")
st.title("🏀 NBA Betting Recommender Dashboard")

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---------------------------
# Paths (relative to this file, works in Docker & locally)
# ---------------------------
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "bet_recommender_model.pkl"
TOP3_PATH = BASE_DIR / "validation_top3_recommendations.csv"
ALL_SCORES_PATH = BASE_DIR / "validation_all_scores.csv"
VAL_PATH = BASE_DIR / "df_validation.xlsx"
SCHEDULE_PATH = BASE_DIR / "playoff_schedule.csv"

# ---------------------------
# Loaders (cached)
# ---------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Model not loaded: {e}")
        return None

@st.cache_data
def load_top3():
    df = pd.read_csv(TOP3_PATH)
    df["betdate"] = pd.to_datetime(df["betdate"], errors="coerce").dt.date
    return df

@st.cache_data
def load_all_scores():
    df = pd.read_csv(ALL_SCORES_PATH)
    df["betdate"] = pd.to_datetime(df["betdate"], errors="coerce").dt.date
    return df

@st.cache_data
def load_val_actuals():
    df = pd.read_excel(VAL_PATH)
    if "betdate" in df.columns:
        df["betdate"] = pd.to_datetime(df["betdate"], errors="coerce").dt.date
    return df

@st.cache_data
def load_schedule():
    try:
        df = pd.read_csv(SCHEDULE_PATH, sep="\t", engine="python")
        if df.shape[1] == 1:
            df = pd.read_csv(SCHEDULE_PATH)
    except Exception:
        df = pd.read_csv(SCHEDULE_PATH)

    df.columns = [c.lower() for c in df.columns]

    if "date" in df.columns:
        df["betdate"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.date
    if "game" in df.columns:
        df["event_description"] = df["game"]

    return df[["betdate", "event_description"]]

# ---------------------------
# Helpers
# ---------------------------
def normalize_event(event: str):
    """Normalize game strings from vendor (&) and model (@) into sorted tuples of team names."""
    event = str(event).lower().strip()
    if "&" in event:
        teams = [t.strip() for t in event.split("&")]
    elif "@" in event:
        teams = [t.strip() for t in event.split("@")]
    else:
        teams = [event]
    return tuple(sorted(teams))

# ---------------------------
# Data
# ---------------------------
model = load_model()
top3_recs = load_top3()
all_scores = load_all_scores()
val_actuals = load_val_actuals()
schedule = load_schedule()

# ---------------------------
# Sidebar Navigation
# ---------------------------
page = st.sidebar.radio("📑 Navigate", ["Top-3 by User", "Workflow & Simulation"])

# ---------------------------
# PAGE 1: Top-3 by User
# ---------------------------
if page == "Top-3 by User":
    st.markdown("""
    This page shows **top-3 recommended games per user** from validation.
    Use the sidebar to filter by `mask_id` or explore all recommendations.
    """)

    mask_ids = sorted(top3_recs["mask_id"].unique())
    mask_id = st.sidebar.selectbox("Select mask_id", ["All users"] + [str(m) for m in mask_ids])

    if mask_id != "All users":
        filtered = top3_recs[top3_recs["mask_id"].astype(str) == mask_id].copy()
        st.subheader(f"Recommendations for user {mask_id}")
        st.dataframe(
            filtered[["mask_id", "betdate", "event_description", "score", "rank"]],
            hide_index=True
        )
    else:
        st.subheader("All Top-3 Recommendations")
        st.dataframe(
            top3_recs[["mask_id", "betdate", "event_description", "score", "rank"]],
            hide_index=True
        )

# ---------------------------
# PAGE 2: Workflow & Simulation
# ---------------------------
elif page == "Workflow & Simulation":
    tab1, tab2 = st.tabs(["📅 Daily Workflow", "📊 Simulation (2 Weeks)"])

    # --- Daily Workflow ---
    with tab1:
        st.header("⚙️ Daily Workflow (Simulated 9AM EST Run)")

        # --- Filters ---
        mask_ids = sorted(all_scores["mask_id"].unique())
        mask_id_input = st.selectbox("Select mask_id (optional)", ["All users"] + [str(m) for m in mask_ids])
        mask_id_filter = None if mask_id_input == "All users" else int(mask_id_input)

        available_dates = sorted(schedule["betdate"].unique())
        selected_date = st.selectbox("Select a date:", available_dates)

        # --- Vendor Schedule ---
        todays_games = schedule[schedule["betdate"] == selected_date]
        if todays_games.empty:
            st.warning("No games scheduled for this date.")
        else:
            st.subheader("📅 Vendor Schedule (Games Today)")
            todays_games = todays_games.reset_index(drop=True).copy()
            todays_games.insert(0, "scheduled_game", range(1, len(todays_games) + 1))
            st.dataframe(
                todays_games[["scheduled_game", "betdate", "event_description"]],
                hide_index=True
            )

            # --- Scores merged with schedule ---
            today_scores = all_scores[all_scores["betdate"] == selected_date].copy()
            today_scores["event_norm"] = today_scores["event_description"].apply(normalize_event)
            todays_games["event_norm"] = todays_games["event_description"].apply(normalize_event)

            merged = todays_games.merge(
                today_scores,
                on="event_norm",
                how="left",
                suffixes=("_schedule", "_score")
            )

            if mask_id_filter is not None:
                merged = merged[merged["mask_id"] == mask_id_filter]

            # --- Top-3 Display ---
            if merged.empty or merged["score"].dropna().empty:
                st.warning("⚠️ No scores available for this date after filtering.")
            else:
                top3_today = None  # ensure variable exists
                agg = None

                if mask_id_filter is not None:
                    # Per-user top 3 (rank from CSV)
                    top3_today = merged[merged["rank"] <= 3].copy()
                    st.subheader(f"🎯 Top-3 Recommendations for User {mask_id_filter}")
                    st.dataframe(
                        top3_today[["mask_id", "betdate_score", "event_description_score", "score", "rank"]],
                        hide_index=True
                    )
                else:
                    # Global top 3 (computed on the fly)
                    agg = (
                        merged.groupby("event_description_schedule", as_index=False)
                        .agg(mean_score=("score", "mean"),
                            n_users=("mask_id", "nunique"))
                        .sort_values("mean_score", ascending=False)
                        .head(3)
                    )
                    agg["rank"] = range(1, len(agg) + 1)
                    st.subheader("🌎 Top-3 Betting Games (Global OLG View)")
                    st.dataframe(
                        agg[["rank", "event_description_schedule", "mean_score", "n_users"]],
                        hide_index=True
                    )

                # ✅ GPT Messaging Checkbox (safe for both branches)
                if st.checkbox("Generate personalized messaging (GPT-4o-mini)", key="daily_msg"):
                    try:
                        if mask_id_filter is not None and top3_today is not None:
                            context = "\n".join(
                                f"{r.event_description_score} ({r.score:.2f})"
                                for _, r in top3_today.iterrows()
                            )
                            who = f"for mask_id {mask_id_filter}"
                        elif agg is not None:
                            context = "\n".join(
                                f"{r.event_description_schedule} ({r.mean_score:.2f})"
                                for _, r in agg.iterrows()
                            )
                            who = "(global OLG view)"
                        else:
                            context = "No data available"
                            who = ""

                        prompt = f"""
                        You are a sports betting assistant.

                        IMPORTANT: The scores provided do NOT represent a team's chance of winning.
                        They represent the model's predicted likelihood that a user (or users on average) will place a bet on that game.

                        Based on model predictions {who} for {selected_date}, here are the top games with their bet-likelihood scores:

                        {context}

                        Write 2–3 sentences of clear messaging for an OLG analyst, explaining these picks in terms of BETTING LIKELIHOOD, 
                        not game outcomes or winning odds.
                        """
                        resp = openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=250
                        )
                        st.success(resp.choices[0].message.content)
                    except openai.RateLimitError:
                        st.error("⚠️ GPT messaging unavailable: quota exceeded on this API key.")

    # --- Simulation (2 Weeks) ---
    with tab2:
        st.header("📊 Simulation & Validation (2-Week Backtest)")

        available_dates = sorted(all_scores["betdate"].unique())
        start_date = st.selectbox("Start date:", available_dates, index=0)
        end_date = st.selectbox("End date:", available_dates, index=len(available_dates)-1)

        sim_scores = all_scores[
            (all_scores["betdate"] >= start_date) &
            (all_scores["betdate"] <= end_date)
        ].copy()

        if sim_scores.empty:
            st.warning("No scores in this period.")
        else:
            actuals = val_actuals[
                (val_actuals["betdate"] >= start_date) &
                (val_actuals["betdate"] <= end_date)
            ].copy()

            def precision_recall_at_k(pred_df, act_df, k=3):
                rows = []
                for (uid, d), grp in pred_df.groupby(["mask_id", "betdate"]):
                    topk = grp[grp["rank"] <= k]
                    pred_set = set(topk["event_description"].astype(str))
                    a = act_df[(act_df["mask_id"] == uid) & (act_df["betdate"] == d)]
                    act_set = set(a["event_description"].astype(str))
                    if len(pred_set) == 0:
                        continue
                    hits = len(pred_set & act_set)
                    prec = hits / k
                    rec = (hits / len(act_set)) if len(act_set) > 0 else None
                    rows.append({"mask_id": uid, "betdate": d, "precision": prec, "recall": rec})
                return pd.DataFrame(rows)

            pr = precision_recall_at_k(sim_scores, actuals, k=3)
            if pr.empty:
                st.info("No overlapping user/day pairs to evaluate.")
            else:
                st.write("### Simulation Metrics (per user-day average)")
                st.metric("Precision@3", f"{pr['precision'].mean():.3f}")
                st.metric("Recall@3", f"{pr['recall'].dropna().mean():.3f}")

                st.write("### Sample Top-3 per User-Day")
                sample_top3 = sim_scores[sim_scores["rank"] <= 3].copy()
                st.dataframe(
                    sample_top3[["mask_id", "betdate", "event_description", "score", "rank"]],
                    hide_index=True
                )