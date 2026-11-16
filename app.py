"""Study Habits, Sleep and Marks - a 200-student implementation study (simulated data).

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Study Habits, Sleep and Marks", page_icon="🎓", layout="wide")


# ------------------------------------------------------------------ Module 1
@st.cache_data
def make_data(n: int = 200, seed: int = 2026) -> pd.DataFrame:
    """Generate simulated students from a stated model (no real data used)."""
    rng = np.random.default_rng(seed)

    # Hidden trait: discipline (never saved in the table)
    discipline = rng.normal(0, 1, n)

    # Habits, each influenced a little by discipline
    study = np.clip(rng.normal(3, 1, n) + 0.8 * discipline, 0, 8)
    sleep = np.clip(rng.normal(6.5, 1, n) + 0.3 * discipline, 4, 9)
    attendance = np.clip(rng.normal(75, 8, n) + 6 * discipline, 40, 100)
    screen = np.clip(rng.normal(5, 1.5, n) - 0.5 * discipline, 1, 10)

    # The stated rule of the model
    marks = (30 + 4 * study + 0.25 * attendance + 1.5 * sleep
             - 0.8 * screen + rng.normal(0, 6, n))
    marks = np.round(np.clip(marks, 0, 100), 1)

    # Habit Score (0-100): weights add up to 1.0
    w = {"study": 0.4, "attendance": 0.3, "sleep": 0.3}
    s_study = np.clip(study / 6, 0, 1) * 100
    s_sleep = np.clip((sleep - 4) / 4, 0, 1) * 100
    habit = w["study"] * s_study + w["attendance"] * attendance + w["sleep"] * s_sleep

    return pd.DataFrame({
        "student_id": [f"S{100 + i}" for i in range(n)],
        "study_hours": np.round(study, 1),
        "sleep_hours": np.round(sleep, 1),
        "attendance_pct": np.round(attendance, 1),
        "screen_hours": np.round(screen, 1),
        "habit_score": np.round(habit, 1),
        "marks": marks,
    })


# ------------------------------------------------------------------ Sidebar
st.sidebar.title("Settings")
n_students = st.sidebar.slider("Number of students", 50, 1000, 200, step=50)
seed = st.sidebar.number_input("Random seed", value=2026, step=1)
st.sidebar.caption("The same seed always gives the same data.")

df = make_data(n_students, int(seed))

st.title("🎓 Study Habits, Sleep and Marks")
st.caption("An implementation study on simulated students. The data is simulated on purpose: "
           "real student data needs consent, and nothing here is evidence about real students.")

tab_over, tab_data, tab_an, tab_fig = st.tabs(
    ["Overview", "Data", "Analysis", "Graphs"])

# ------------------------------------------------------------------ Overview
with tab_over:
    st.subheader("Problem")
    st.write("Teachers see marks, attendance and study or sleep habits separately, so they cannot say "
             "which habits go with better marks. This project joins them in one pipeline using simulated data.")
    st.subheader("Five stages")
    st.table(pd.DataFrame({
        "Stage": ["1. Specify", "2. Generate", "3. Index", "4. Analyse", "5. Visualise"],
        "What happens": [
            "Write the rules for how habits affect marks",
            "Create the students and save a table",
            "Combine 3 habits into one 0-100 Habit Score",
            "Correlation, regression, groups, sleep bands",
            "Draw the graphs"],
    }))
    st.subheader("The marks rule")
    st.code("marks = 30 + 4*study + 0.25*attendance + 1.5*sleep - 0.8*screen + noise", language="python")
    st.subheader("Habit Score weights (sum = 1.0)")
    st.write("Study 0.4 (full marks at 6 h/day), Attendance 0.3, Sleep 0.3 (4 h = 0, 8 h = 100).")

# ------------------------------------------------------------------ Data
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Students", len(df))
    c2.metric("Mean marks", f"{df.marks.mean():.1f}")
    c3.metric("Sleeping under 6 h", f"{(df.sleep_hours < 6).mean() * 100:.1f}%")
    st.dataframe(df, use_container_width=True, height=360)
    st.download_button("Download students.csv", df.to_csv(index=False).encode("utf-8"),
                       file_name="students.csv", mime="text/csv")
    st.subheader("Summary")
    st.dataframe(df.describe().loc[["mean", "std", "min", "max"]].round(2), use_container_width=True)

# ------------------------------------------------------------------ Module 2
cols = ["study_hours", "sleep_hours", "attendance_pct", "screen_hours", "habit_score"]
df["group"] = pd.cut(df.marks, [0, 50, 70, 100], labels=["Low (<50)", "Medium (50-70)", "High (70+)"])
df["sleep_band"] = pd.cut(df.sleep_hours, [0, 6, 7, 12], labels=["<6 h", "6-7 h", ">7 h"])

with tab_an:
    st.subheader("1. Correlation with marks")
    corr = df[cols + ["marks"]].corr()["marks"].drop("marks").round(3)
    st.dataframe(corr.rename("r with marks"), use_container_width=True)
    st.warning("Habit Score is built from study, sleep and attendance, "
               "so its link with marks is partly by construction.")

    st.subheader("2. Regression: marks ~ study + sleep + attendance + screen")
    X = np.column_stack([np.ones(len(df)), df.study_hours, df.sleep_hours,
                         df.attendance_pct, df.screen_hours])
    beta, *_ = np.linalg.lstsq(X, df.marks.to_numpy(), rcond=None)
    reg = pd.DataFrame({"term": ["intercept", "study", "sleep", "attendance", "screen"],
                        "coefficient": np.round(beta, 3),
                        "rule in model": ["30", "+4", "+1.5", "+0.25", "-0.8"]})
    st.dataframe(reg, use_container_width=True, hide_index=True)
    st.caption("Each coefficient is the change in marks per unit of that habit, others held fixed. "
               "They should be close to the rule that generated the data.")

    st.subheader("3. Marks groups")
    grp = df.groupby("group", observed=True).agg(
        students=("marks", "size"), study=("study_hours", "mean"),
        sleep=("sleep_hours", "mean"), attendance=("attendance_pct", "mean"),
        habit=("habit_score", "mean")).round(2)
    st.dataframe(grp, use_container_width=True)
    if (grp.students < 15).any():
        st.info("A group with very few students gives unreliable averages.")

    st.subheader("4. Mean marks by sleep band")
    sl = df.groupby("sleep_band", observed=True).marks.mean().round(1)
    st.dataframe(sl.rename("mean marks"), use_container_width=True)

# ------------------------------------------------------------------ Module 3
with tab_fig:
    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df.marks, bins=20, color="#2563eb", edgecolor="white")
        ax.axvline(df.marks.mean(), color="black", linestyle="--")
        ax.set(title="Distribution of marks", xlabel="Marks", ylabel="Students")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(df.study_hours, df.marks, s=18, alpha=0.7, color="#0d9488")
        m, c = np.polyfit(df.study_hours, df.marks, 1)
        xs = np.linspace(df.study_hours.min(), df.study_hours.max(), 50)
        ax.plot(xs, m * xs + c, color="#f97316", linewidth=2)
        ax.set(title="Study hours vs marks", xlabel="Study hours per day", ylabel="Marks")
        st.pyplot(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(sl.index.astype(str), sl.values, color=["#ef4444", "#f59e0b", "#22c55e"])
    for i, v in enumerate(sl.values):
        ax.text(i, v + 1, f"{v:.1f}", ha="center")
    ax.set(ylim=(0, 100), title="Average marks by sleep", xlabel="Sleep per night", ylabel="Mean marks")
    st.pyplot(fig)
    plt.close(fig)
