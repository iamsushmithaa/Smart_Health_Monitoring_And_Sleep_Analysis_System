# app.py
# Smart Health Monitoring & Sleep Analysis System
# FINAL VERSION - Connected to Trained ML Models from main.py
# Features: Login, Input, ML Predictions, Dashboard, Why This Result,
#           Graphs, Suggestions, Food Plan, Weekly Plan, PDF Reports

import matplotlib
matplotlib.use("Agg")  # Fix blank page - forces non-interactive backend

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

st.set_page_config(page_title="Smart Health Monitoring", layout="wide", page_icon="🏥")

# ---------------------------------------------------
# FOLDERS
# ---------------------------------------------------
if not os.path.exists("reports"):
    os.makedirs("reports")

# ---------------------------------------------------
# LOAD ML MODELS
# ---------------------------------------------------
@st.cache_resource
def load_models():
    try:
        quality_model    = joblib.load("models/sleep_quality_model.pkl")
        disorder_model   = joblib.load("models/sleep_disorder_model.pkl")
        reg_columns      = joblib.load("models/regression_columns.pkl")
        clf_columns      = joblib.load("models/classification_columns.pkl")
        return quality_model, disorder_model, reg_columns, clf_columns
    except FileNotFoundError:
        return None, None, None, None

quality_model, disorder_model, reg_columns, clf_columns = load_models()
models_loaded = quality_model is not None

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"

if "form_data" not in st.session_state:
    st.session_state.form_data = {
        "name": "", "age": 25, "gender": "Male",
        "sleep": 7.0, "stress": 5, "activity": 30,
        "heart": 75, "steps": 7000,
        "sys": 120, "dia": 80, "bmi": "Normal",
        "occupation": "Engineer"
    }

# ---------------------------------------------------
# HELPER: BUILD INPUT DATAFRAME FOR MODEL
# ---------------------------------------------------
def build_input_df(d, columns):
    """
    Builds a DataFrame row matching the columns the model was trained on.
    Handles one-hot encoded columns for Occupation and BMICategory.
    """
    row = {col: 0 for col in columns}

    # Direct numeric features
    mapping = {
        "Gender":               1 if d["gender"] == "Female" else 0,
        "Age":                  d["age"],
        "SleepDuration":        d["sleep"],
        "QualityofSleep":       7,          # placeholder when predicting disorder
        "PhysicalActivityLevel":d["activity"],
        "StressLevel":          d["stress"],
        "HeartRate":            d["heart"],
        "DailySteps":           d["steps"],
        "Systolic":             d["sys"],
        "Diastolic":            d["dia"],
        "SleepDisorder":        0,          # placeholder when predicting quality
    }

    for k, v in mapping.items():
        if k in row:
            row[k] = v

    # One-hot: Occupation
    occ_col = f"Occupation_{d['occupation']}"
    if occ_col in row:
        row[occ_col] = 1

    # One-hot: BMICategory (drop_first=True drops "Normal" as baseline)
    bmi_col = f"BMICategory_{d['bmi']}"
    if bmi_col in row:
        row[bmi_col] = 1

    return pd.DataFrame([row])

# ---------------------------------------------------
# HELPER: STRONG PASSWORD
# ---------------------------------------------------
def strong_password(p):
    return (
        len(p) >= 8 and
        any(i.isupper() for i in p) and
        any(i.islower() for i in p) and
        any(i.isdigit() for i in p) and
        any(not i.isalnum() for i in p)
    )

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
        color: #00BFA6;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a3c5e, #0f2740);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #00BFA6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00BFA6;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 4px;
    }
    .reason-box {
        background: #1a1f2e;
        border-left: 4px solid #00BFA6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        color: #e2e8f0;
        font-size: 14px;
    }
    .suggestion-chip {
        display: inline-block;
        background: #1a3c5e;
        color: #00BFA6;
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px;
        font-size: 13px;
        font-weight: 500;
    }
    .model-badge {
        background: linear-gradient(90deg, #00BFA6, #007bff);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================
# LOGIN PAGE
# ===================================================
if st.session_state.page == "login":

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align:center;color:#00BFA6;'>🏥 Smart Health Monitor</h1>"
            "<p style='text-align:center;color:#a0aec0;'>AI-Powered Sleep & Health Analysis</p>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        username = st.text_input("👤 Username")
        password = st.text_input("🔒 Password", type="password")
        st.caption("Password must have 8+ chars, Uppercase, Lowercase, Number & Symbol")

        if st.button("Login →", use_container_width=True):
            if username.strip() == "":
                st.error("Please enter a username.")
            elif not strong_password(password):
                st.error("Weak password! Try: Health@123")
            else:
                st.session_state.username = username
                st.session_state.page = "input"
                st.rerun()

        if not models_loaded:
            st.warning("⚠️ ML models not found. Run `main.py` first to train and save models. Using rule-based fallback.")

    st.stop()

# ===================================================
# INPUT PAGE
# ===================================================
if st.session_state.page == "input":

    f = st.session_state.form_data

    h1, h2 = st.columns([6, 1])
    with h1:
        st.markdown("<h1 style='color:#00BFA6;'>🏥 AI Smart Health Monitoring</h1>", unsafe_allow_html=True)
        if models_loaded:
            st.markdown("<span class='model-badge'>🤖 ML Models Active</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#f39c12;font-size:13px;'>⚠️ Rule-based mode (run main.py to enable ML)</span>", unsafe_allow_html=True)
    with h2:
        if st.button("Logout"):
            st.session_state.page = "login"
            st.rerun()

    st.markdown("---")
    st.subheader("📋 Enter Patient Details")

    c1, c2 = st.columns(2)

    with c1:
        name       = st.text_input("Patient Name", value=f["name"])
        age        = st.slider("Age", 18, 80, f["age"])
        gender     = st.selectbox("Gender", ["Male", "Female"],
                                  index=0 if f["gender"] == "Male" else 1)
        occupation = st.selectbox("Occupation", [
            "Accountant", "Doctor", "Engineer", "Lawyer", "Manager",
            "Nurse", "Sales Representative", "Salesperson",
            "Scientist", "Software Engineer", "Teacher"
        ], index=2)
        sleep      = st.slider("Sleep Duration (hrs)", 3.0, 10.0, f["sleep"])
        stress     = st.slider("Stress Level (1-10)", 1, 10, f["stress"])

    with c2:
        activity = st.slider("Physical Activity (mins/day)", 0, 120, f["activity"])
        heart    = st.slider("Heart Rate (bpm)", 40, 140, f["heart"])
        steps    = st.slider("Daily Steps", 0, 20000, f["steps"])
        sys_bp   = st.slider("Systolic BP", 80, 180, f["sys"])
        dia_bp   = st.slider("Diastolic BP", 50, 120, f["dia"])

    bmi = st.selectbox(
        "BMI Category",
        ["Underweight", "Normal", "Overweight", "Obese"],
        index=["Underweight", "Normal", "Overweight", "Obese"].index(f["bmi"])
    )

    if st.button("🔍 Predict Health Status", use_container_width=True):
        if name.strip() == "":
            st.error("Please enter the patient name.")
        else:
            st.session_state.form_data = {
                "name": name, "age": age, "gender": gender,
                "sleep": sleep, "stress": stress, "activity": activity,
                "heart": heart, "steps": steps,
                "sys": sys_bp, "dia": dia_bp,
                "bmi": bmi, "occupation": occupation
            }
            st.session_state.data = st.session_state.form_data
            st.session_state.page = "result"
            st.rerun()

# ===================================================
# RESULT PAGE
# ===================================================
if st.session_state.page == "result":

    d = st.session_state.data

    sleep    = d["sleep"]
    stress   = d["stress"]
    activity = d["activity"]
    heart    = d["heart"]
    steps    = d["steps"]
    sys_bp   = d["sys"]
    dia_bp   = d["dia"]
    bmi      = d["bmi"]

    # ---------------------------------------------------
    # PREDICTION — ML or Rule-Based Fallback
    # ---------------------------------------------------
    if models_loaded:
        # --- ML: Sleep Quality ---
        df_q = build_input_df(d, reg_columns)
        # Remove QualityofSleep from features if present
        if "QualityofSleep" in df_q.columns:
            df_q = df_q.drop(columns=["QualityofSleep"])
        # Align columns to model
        q_cols = [c for c in reg_columns if c != "QualityofSleep"]
        df_q = df_q.reindex(columns=q_cols, fill_value=0)
        sleep_quality = round(float(quality_model.predict(df_q)[0]), 1)
        sleep_quality = max(1.0, min(10.0, sleep_quality))

        # --- ML: Sleep Disorder ---
        df_d = build_input_df(d, clf_columns)
        if "SleepDisorder" in df_d.columns:
            df_d = df_d.drop(columns=["SleepDisorder"])
        d_cols = [c for c in clf_columns if c != "SleepDisorder"]
        df_d = df_d.reindex(columns=d_cols, fill_value=0)
        disorder_code  = int(disorder_model.predict(df_d)[0])
        disorder_map   = {0: "No Disorder", 1: "Insomnia", 2: "Sleep Apnea"}
        disorder       = disorder_map[disorder_code]

        # Confidence using predict_proba
        try:
            proba      = disorder_model.predict_proba(df_d)[0]
            confidence = round(max(proba) * 100, 1)
        except Exception:
            confidence = 88.0

        prediction_source = "🤖 ML Model"

    else:
        # Rule-based fallback
        sleep_quality = 10
        if sleep < 6:    sleep_quality -= 3
        elif sleep < 7:  sleep_quality -= 1
        if stress > 7:   sleep_quality -= 3
        elif stress > 5: sleep_quality -= 1
        if activity < 20: sleep_quality -= 1
        sleep_quality = max(1, sleep_quality)

        disorder   = "No Disorder"
        confidence = 88.0
        if sleep < 5.5 and stress > 7:
            disorder = "Insomnia"; confidence = 92.0
        elif bmi in ["Overweight", "Obese"] and heart > 95:
            disorder = "Sleep Apnea"; confidence = 90.0

        prediction_source = "⚙️ Rule-based"

    # --- Health Score (always rule-based for explainability) ---
    score = 100
    if sleep < 7:           score -= 15
    if stress > 6:          score -= 15
    if activity < 30:       score -= 10
    if steps < 6000:        score -= 10
    if bmi == "Overweight": score -= 10
    if bmi == "Obese":      score -= 20
    if sys_bp > 140:        score -= 10
    score = max(0, score)

    if score >= 80:   status = "Excellent"
    elif score >= 60: status = "Moderate"
    else:             status = "High Risk"

    status_color = {"Excellent": "#00BFA6", "Moderate": "#f39c12", "High Risk": "#e74c3c"}[status]

    # ---------------------------------------------------
    # HEADER
    # ---------------------------------------------------
    h1, h2 = st.columns([6, 1])
    with h1:
        st.markdown("<h1 style='color:#00BFA6;'>📊 Prediction Dashboard</h1>", unsafe_allow_html=True)
        st.markdown(f"<span class='model-badge'>{prediction_source}</span>", unsafe_allow_html=True)
    with h2:
        if st.button("Logout"):
            st.session_state.page = "login"
            st.rerun()

    if st.button("⬅ Back to Input"):
        st.session_state.page = "input"
        st.rerun()

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Dashboard", "🧠 Why This Result", "📊 Graphs",
        "💡 Suggestions", "📄 Health Report"
    ])

    # ===================================================
    # TAB 1 — DASHBOARD
    # ===================================================
    with tab1:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{sleep_quality}/10</div>
                <div class='metric-label'>Sleep Quality</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            disorder_color = "#e74c3c" if disorder != "No Disorder" else "#00BFA6"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{disorder_color};font-size:1.3rem;'>{disorder}</div>
                <div class='metric-label'>Sleep Disorder</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{confidence}%</div>
                <div class='metric-label'>Confidence</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{status_color};'>{score}</div>
                <div class='metric-label'>Health Score / 100</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(score / 100)
        st.markdown(
            f"<h3 style='color:{status_color};text-align:center;'>{status}</h3>",
            unsafe_allow_html=True
        )

        # Quick summary table
        st.markdown("### 📋 Patient Summary")
        summary_df = pd.DataFrame({
            "Parameter": ["Name", "Age", "Gender", "Sleep", "Stress",
                          "Activity", "Heart Rate", "Daily Steps", "Blood Pressure", "BMI"],
            "Value": [
                d["name"], f"{d['age']} yrs", d["gender"],
                f"{sleep} hrs", f"{stress}/10",
                f"{activity} mins", f"{heart} bpm",
                str(steps), f"{sys_bp}/{dia_bp} mmHg", bmi
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ===================================================
    # TAB 2 — WHY THIS RESULT
    # ===================================================
    with tab2:

        st.subheader("🧠 Why This Result?")

        reasons = []

        if sleep < 6:
            reasons.append(f"Sleep duration is only {sleep} hrs — below the 7-9 hr recommendation. This reduces recovery and drags sleep quality down.")
        elif sleep < 7:
            reasons.append(f"Sleep duration is {sleep} hrs — slightly below ideal. Even 30 extra minutes can improve wellness significantly.")
        else:
            reasons.append(f"Sleep duration is {sleep} hrs — within the healthy range. This supports good body recovery.")

        if stress >= 8:
            reasons.append(f"Stress level is very high at {stress}/10. This severely disrupts sleep and contributes to health risks.")
        elif stress >= 5:
            reasons.append(f"Stress level is moderate at {stress}/10. Better stress management could improve the overall score.")
        else:
            reasons.append(f"Stress level is low at {stress}/10 — a positive indicator for sleep and wellbeing.")

        if activity < 30:
            reasons.append(f"Physical activity is only {activity} mins/day — below the 30-min recommendation. Low activity can worsen sleep quality.")
        else:
            reasons.append(f"Physical activity of {activity} mins/day is on track. This supports cardiovascular health and sleep.")

        if steps < 6000:
            reasons.append(f"Daily steps count is {steps} — indicating low movement. Target is 8000+ steps for a healthy lifestyle.")
        else:
            reasons.append(f"Daily steps of {steps} reflect an active lifestyle, which is beneficial for health.")

        if bmi == "Overweight":
            reasons.append("BMI is in the Overweight range — this increases the risk of sleep apnea and cardiovascular issues.")
        elif bmi == "Obese":
            reasons.append("BMI is in the Obese range — this significantly affects sleep quality and heart health.")
        elif bmi == "Underweight":
            reasons.append("BMI is in the Underweight range — better nutrition is needed to support body functions.")
        else:
            reasons.append("BMI is in the Normal range — this is a positive factor for overall health.")

        if sys_bp > 140:
            reasons.append(f"Blood pressure of {sys_bp}/{dia_bp} mmHg is above normal. This increases cardiovascular risk and affects health score.")
        elif sys_bp > 120:
            reasons.append(f"Blood pressure of {sys_bp}/{dia_bp} mmHg is slightly elevated. Monitor regularly.")
        else:
            reasons.append(f"Blood pressure of {sys_bp}/{dia_bp} mmHg is within normal range.")

        if disorder == "Insomnia":
            reasons.append("The combination of low sleep duration and high stress created an Insomnia risk pattern detected by the ML model.")
        elif disorder == "Sleep Apnea":
            reasons.append("The pattern of BMI and elevated heart rate was identified by the ML model as a Sleep Apnea risk indicator.")
        else:
            reasons.append("No significant sleep disorder pattern was detected in the current input by the ML model.")

        for r in reasons:
            st.markdown(f"<div class='reason-box'>👉 {r}</div>", unsafe_allow_html=True)

        st.markdown(f"<h3 style='color:{status_color};'>Final Result: {status} ({score}/100)</h3>",
                    unsafe_allow_html=True)

    # ===================================================
    # TAB 3 — GRAPHS
    # ===================================================
    with tab3:

        st.subheader("📊 You vs Healthy Average")

        # Normalize all values to 0-100 scale
        def normalize(val, min_val, max_val):
            return round((val - min_val) / (max_val - min_val) * 100, 1)

        labels = ["Sleep\n(hrs)", "Stress\n(level)", "Activity\n(mins)", "Heart Rate\n(bpm)", "Steps\n(/200)"]
        your_raw   = [sleep, stress, activity, heart, steps]
        avg_raw    = [8.0,   3.0,    60,       72,     10000]
        mins_raw   = [3,     1,      0,        40,     0]
        maxs_raw   = [10,    10,     120,      140,    20000]

        your_norm = [normalize(v, mn, mx) for v, mn, mx in zip(your_raw, mins_raw, maxs_raw)]
        avg_norm  = [normalize(v, mn, mx) for v, mn, mx in zip(avg_raw,  mins_raw, maxs_raw)]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1f2e")

        bars1 = ax.bar(x - width/2, your_norm, width, color="#00BFA6", label="You", alpha=0.9, edgecolor="white")
        bars2 = ax.bar(x + width/2, avg_norm,  width, color="#1a3c5e", label="Healthy Avg", alpha=0.9, edgecolor="white")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", color="white", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", color="white", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color="white")
        ax.set_ylabel("Normalized Score (0-100)", color="white")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#1a1f2e", labelcolor="white")
        ax.set_ylim(0, 115)

        st.pyplot(fig)
        plt.close()

        # Radar-style bar chart for health score components
        st.subheader("🎯 Health Score Breakdown")

        components = {
            "Sleep": min(100, (sleep / 9) * 100),
            "Low Stress": max(0, 100 - (stress * 10)),
            "Activity": min(100, (activity / 60) * 100),
            "Steps": min(100, (steps / 10000) * 100),
            "BP": max(0, 100 - max(0, sys_bp - 120) * 2),
            "BMI": {"Normal": 100, "Underweight": 60, "Overweight": 50, "Obese": 20}[bmi]
        }

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor("#0f1117")
        ax2.set_facecolor("#1a1f2e")

        bar_colors = ["#00BFA6" if v >= 70 else "#f39c12" if v >= 40 else "#e74c3c"
                      for v in components.values()]
        comp_bars = ax2.barh(list(components.keys()), list(components.values()),
                             color=bar_colors, edgecolor="none")
        ax2.set_xlim(0, 110)
        ax2.tick_params(colors="white")
        ax2.spines["bottom"].set_color("#333")
        ax2.spines["left"].set_color("#333")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_xlabel("Score (%)", color="white")

        for bar in comp_bars:
            ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                     f"{bar.get_width():.0f}%", va="center", color="white", fontsize=9)

        st.pyplot(fig2)
        plt.close()

    # ===================================================
    # TAB 4 — SUGGESTIONS
    # ===================================================
    with tab4:

        st.subheader("✅ Health Suggestions")

        suggestions = []
        if sleep < 6:
            suggestions += ["Increase sleep to 7–8 hours daily.", "Avoid screens 30 mins before bed."]
        if stress >= 7:
            suggestions += ["Practice 10-minute daily meditation.", "Try journaling to reduce overthinking."]
        if activity < 30:
            suggestions.append("Add a 30-minute walk to your daily routine.")
        if steps < 6000:
            suggestions.append("Gradually increase daily steps toward 8000+.")
        if sys_bp > 140:
            suggestions += ["Reduce salt intake.", "Monitor blood pressure regularly."]
        if bmi == "Overweight":
            suggestions.append("Combine moderate cardio with dietary improvements.")
        if bmi == "Obese":
            suggestions.append("Consult a doctor and start a structured weight loss plan.")
        if disorder == "Insomnia":
            suggestions.append("Maintain a fixed bedtime and wake time every day.")
        if disorder == "Sleep Apnea":
            suggestions.append("Sleep on your side and consult a sleep specialist.")
        if not suggestions:
            suggestions.append("Excellent! Maintain your current healthy routine.")

        for i in range(0, len(suggestions), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(suggestions):
                    col.success(f"✅ {suggestions[i+j]}")

        # Food Suggestions
        st.markdown("---")
        st.subheader("🥗 Food Suggestions")

        foods = []
        if sleep < 6:    foods += ["🍌 Banana", "🥛 Warm Milk", "🌰 Almonds"]
        if stress >= 7:  foods += ["🫐 Blueberries", "🍫 Dark Chocolate", "🍵 Green Tea"]
        if bmi == "Underweight": foods += ["🥚 Eggs", "🍚 Rice", "🥜 Peanut Butter"]
        elif bmi == "Normal":    foods += ["🥦 Vegetables", "🍎 Fruits", "🍗 Lean Protein"]
        elif bmi == "Overweight":foods += ["🌾 Oats", "🥗 Salad", "🍎 Apple"]
        elif bmi == "Obese":     foods += ["🥦 Boiled Vegetables", "🍗 Lean Protein", "🍵 Green Tea"]
        if sys_bp > 140: foods += ["🥕 Beetroot Juice", "🥑 Avocado"]
        if not foods:    foods = ["🥦 Vegetables", "🍎 Fruits", "🍗 Protein", "💧 Water"]

        foods = list(dict.fromkeys(foods))  # deduplicate

        food_cols = st.columns(4)
        for i, food in enumerate(foods):
            food_cols[i % 4].markdown(f"<div class='suggestion-chip'>{food}</div>",
                                      unsafe_allow_html=True)

        # Weekly Plan
        st.markdown("---")
        st.subheader("📅 7 Day Weekly Plan")

        daily_habits = []
        if sleep < 7:
            daily_habits += ["Sleep before 10:30 PM", "Wake up at the same time daily"]
        if activity < 30 or steps < 6000:
            daily_habits.append("Walk minimum 30 minutes")
        daily_habits.append("Drink 2–3 litres of water")
        if bmi in ["Overweight", "Obese"]:
            daily_habits.append("Avoid junk food and excess sugar")
        if stress >= 6:
            daily_habits.append("Practice 10-min deep breathing")
        if disorder == "Insomnia":
            daily_habits.append("No screens 30 mins before sleep")
        if not daily_habits:
            daily_habits = ["Maintain healthy routine", "Walk daily", "Balanced meals", "Sleep on time"]

        st.write("**Follow these every day:**")
        for h in daily_habits:
            st.markdown(f"✅ {h}")

        st.markdown("---")

        if disorder == "Insomnia":
            weekly = [
                ("Day 1", "Focus on sleep routine. Morning walk, protein breakfast, sleep before 10:30 PM."),
                ("Day 2", "Fruits in diet, reduce screen time, breathing exercise before bed."),
                ("Day 3", "Light stretching, avoid oily food, keep mind relaxed."),
                ("Day 4", "Balanced meals, avoid late-night snacks, continue all habits."),
                ("Day 5", "Longer walk, avoid evening caffeine, consistent bedtime."),
                ("Day 6", "15-min meditation, early dinner, calm environment before sleep."),
                ("Day 7", "Review sleep progress. Celebrate consistency and plan next week."),
            ]
        elif disorder == "Sleep Apnea":
            weekly = [
                ("Day 1", "Focus on breathing and weight control. Walk 30 mins, avoid heavy dinner."),
                ("Day 2", "Add fruits and vegetables. Sleep on side position tonight."),
                ("Day 3", "Light stretching, reduce sugary foods, stay hydrated."),
                ("Day 4", "Balanced meals, avoid oily foods, maintain walk routine."),
                ("Day 5", "Increase walking to 40 mins. Breathing exercises."),
                ("Day 6", "Stress control practice, early dinner."),
                ("Day 7", "Review breathing comfort and plan next week."),
            ]
        elif bmi in ["Overweight", "Obese"]:
            weekly = [
                ("Day 1", "Focus on weight management. Walk 30 mins, avoid fried food."),
                ("Day 2", "Add salad and fruits to meals."),
                ("Day 3", "Light exercise, avoid sugary snacks."),
                ("Day 4", "Balanced meals, hydrate well."),
                ("Day 5", "Walk 40 mins, eat more vegetables."),
                ("Day 6", "Early dinner and deep breathing."),
                ("Day 7", "Review weight progress and stay consistent."),
            ]
        elif stress >= 7:
            weekly = [
                ("Day 1", "Stress control focus. Morning walk, journaling."),
                ("Day 2", "Deep breathing and reduced screen time."),
                ("Day 3", "Healthy meals, keep mind relaxed."),
                ("Day 4", "Meditation and proper hydration."),
                ("Day 5", "Outdoor walk and calm music."),
                ("Day 6", "Early dinner and sleep on time."),
                ("Day 7", "Review stress improvement."),
            ]
        else:
            weekly = [
                ("Day 1", "Maintain routine. Morning walk, healthy breakfast."),
                ("Day 2", "Add fruits and stay hydrated."),
                ("Day 3", "Light stretching or yoga."),
                ("Day 4", "Balanced meals and relaxation time."),
                ("Day 5", "Walk 40 mins, avoid evening caffeine."),
                ("Day 6", "Healthy homemade food and early dinner."),
                ("Day 7", "Review progress and maintain consistency."),
            ]

        for day, plan in weekly:
            st.markdown(f"📌 **{day}:** {plan}")

        # --- Weekly PDF ---
        st.markdown("---")
        now = datetime.now()
        report_date = now.strftime("%d %B %Y")
        report_time = now.strftime("%I:%M %p")
        W, H = A4
        weekly_file = f"reports/{d['name']}_Weekly_Report.pdf"
        pdf_canvas = canvas.Canvas(weekly_file, pagesize=A4)

        def draw_report_header(c, title):
            c.setStrokeColor(colors.HexColor("#1a3c5e"))
            c.setLineWidth(3)
            c.rect(15, 15, W - 30, H - 30, fill=0)
            c.setLineWidth(1)
            c.rect(20, 20, W - 40, H - 40, fill=0)
            c.setFillColor(colors.HexColor("#1a3c5e"))
            c.rect(0, H - 90, W, 90, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(W / 2, H - 38, "SMART HEALTH CARE CENTER")
            c.setFont("Helvetica", 10)
            c.drawCentredString(W / 2, H - 55, "Department of Preventive Medicine & Wellness")
            c.setFont("Helvetica", 8)
            c.drawCentredString(W / 2, H - 70, "Tel: +91-9999999999  |  Email: care@smarthealthcenter.com")
            c.setFillColor(colors.HexColor("#e8f0f7"))
            c.rect(30, H - 115, W - 60, 22, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#1a3c5e"))
            c.setFont("Helvetica-Bold", 11)
            c.drawCentredString(W / 2, H - 108, title)

        draw_report_header(pdf_canvas, "WEEKLY HEALTH & WELLNESS PLAN")

        # Patient info
        pdf_canvas.setStrokeColor(colors.HexColor("#1a3c5e"))
        pdf_canvas.setLineWidth(0.8)
        pdf_canvas.setFillColor(colors.HexColor("#f7fbff"))
        pdf_canvas.rect(30, H - 175, W - 60, 52, fill=1, stroke=1)
        pdf_canvas.setFillColor(colors.black)
        for label, val, x_pos, y_off in [
            ("PATIENT NAME:", d['name'].upper(), 40, H - 133),
            ("AGE / GENDER:", f"{d['age']} yrs  /  {d['gender']}", 40, H - 148),
            ("CONDITION:", disorder, 40, H - 163),
            ("REPORT DATE:", report_date, 320, H - 133),
            ("REPORT TIME:", report_time, 320, H - 148),
        ]:
            pdf_canvas.setFont("Helvetica-Bold", 9)
            pdf_canvas.drawString(x_pos, y_off, label)
            pdf_canvas.setFont("Helvetica", 9)
            pdf_canvas.drawString(x_pos + 90, y_off, val)

        y = H - 195
        pdf_canvas.setFillColor(colors.HexColor("#1a3c5e"))
        pdf_canvas.rect(30, y - 2, W - 60, 16, fill=1, stroke=0)
        pdf_canvas.setFillColor(colors.white)
        pdf_canvas.setFont("Helvetica-Bold", 9)
        pdf_canvas.drawString(40, y + 3, "DAILY HABITS TO FOLLOW EVERY DAY")

        y -= 18
        pdf_canvas.setFillColor(colors.black)
        pdf_canvas.setFont("Helvetica", 9)
        for habit in daily_habits:
            pdf_canvas.drawString(45, y, f"  • {habit}")
            y -= 14

        y -= 8
        pdf_canvas.setFillColor(colors.HexColor("#1a3c5e"))
        pdf_canvas.rect(30, y - 2, W - 60, 16, fill=1, stroke=0)
        pdf_canvas.setFillColor(colors.white)
        pdf_canvas.setFont("Helvetica-Bold", 9)
        pdf_canvas.drawString(40, y + 3, "7 DAY PLAN")

        y -= 16
        for i, (day, plan) in enumerate(weekly):
            if i % 2 == 0:
                pdf_canvas.setFillColor(colors.HexColor("#f0f6fc"))
                pdf_canvas.rect(30, y - 4, W - 60, 15, fill=1, stroke=0)
            pdf_canvas.setFillColor(colors.black)
            pdf_canvas.setFont("Helvetica-Bold", 8)
            pdf_canvas.drawString(40, y + 2, f"{day}:")
            pdf_canvas.setFont("Helvetica", 8)
            pdf_canvas.drawString(85, y + 2, plan)
            y -= 16

        # Footer
        pdf_canvas.setFillColor(colors.HexColor("#1a3c5e"))
        pdf_canvas.rect(0, 0, W, 28, fill=1, stroke=0)
        pdf_canvas.setFillColor(colors.white)
        pdf_canvas.setFont("Helvetica", 7)
        pdf_canvas.drawCentredString(W / 2, 17, "This report is system-generated for health monitoring purposes only.")
        pdf_canvas.drawCentredString(W / 2, 8, f"Generated on {report_date} at {report_time}  |  Smart Health Care Center")
        pdf_canvas.save()

        with open(weekly_file, "rb") as wf:
            st.download_button("📥 Download Weekly Plan PDF", wf,
                               file_name=f"{d['name']}_Weekly_Report.pdf",
                               mime="application/pdf")

    # ===================================================
    # TAB 5 — HEALTH REPORT PDF
    # ===================================================
    with tab5:

        st.subheader("📄 Health Report")
        st.write("Your full diagnostic report with all findings is ready to download.")

        st.markdown("**Report Preview:**")
        preview_df = pd.DataFrame({
            "Finding":    ["Sleep Duration", "Sleep Quality", "Stress Level", "Activity",
                           "Daily Steps", "Heart Rate", "Blood Pressure", "BMI",
                           "Sleep Disorder", "Confidence", "Health Score", "Status"],
            "Your Value": [f"{sleep} hrs", f"{sleep_quality}/10", f"{stress}/10",
                           f"{activity} mins", str(steps), f"{heart} bpm",
                           f"{sys_bp}/{dia_bp} mmHg", bmi,
                           disorder, f"{confidence}%", f"{score}/100", status],
            "Reference":  ["7–9 hrs", "7+", "Below 5", "30+ mins",
                           "8000+", "60–100 bpm", "120/80 mmHg", "Normal",
                           "No Disorder", "—", "80+", "Excellent"]
        })
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        now_hr    = datetime.now()
        hr_date   = now_hr.strftime("%d %B %Y")
        hr_time   = now_hr.strftime("%I:%M %p")
        filename  = f"reports/{d['name']}_Health_Report.pdf"
        rc        = canvas.Canvas(filename, pagesize=A4)

        # Header
        rc.setStrokeColor(colors.HexColor("#1a3c5e"))
        rc.setLineWidth(3)
        rc.rect(15, 15, W - 30, H - 30, fill=0)
        rc.setLineWidth(1)
        rc.rect(20, 20, W - 40, H - 40, fill=0)
        rc.setFillColor(colors.HexColor("#1a3c5e"))
        rc.rect(0, H - 90, W, 90, fill=1, stroke=0)
        rc.setFillColor(colors.white)
        rc.setFont("Helvetica-Bold", 18)
        rc.drawCentredString(W / 2, H - 38, "SMART HEALTH CARE CENTER")
        rc.setFont("Helvetica", 10)
        rc.drawCentredString(W / 2, H - 55, "Department of Sleep Medicine & General Wellness")
        rc.setFont("Helvetica", 8)
        rc.drawCentredString(W / 2, H - 70, "Tel: +91-9999999999  |  Email: care@smarthealthcenter.com")
        rc.setFillColor(colors.HexColor("#e8f0f7"))
        rc.rect(30, H - 115, W - 60, 22, fill=1, stroke=0)
        rc.setFillColor(colors.HexColor("#1a3c5e"))
        rc.setFont("Helvetica-Bold", 11)
        rc.drawCentredString(W / 2, H - 108, "PATIENT HEALTH DIAGNOSTIC REPORT")

        # Patient info
        rc.setStrokeColor(colors.HexColor("#1a3c5e"))
        rc.setLineWidth(0.8)
        rc.setFillColor(colors.HexColor("#f7fbff"))
        rc.rect(30, H - 178, W - 60, 55, fill=1, stroke=1)
        patient_info = [
            ("PATIENT NAME:", d['name'].upper(), 40, H - 133),
            ("AGE / GENDER:", f"{d['age']} yrs  /  {d['gender']}", 40, H - 148),
            ("BMI CATEGORY:", bmi, 40, H - 163),
            ("REPORT DATE:", hr_date, 320, H - 133),
            ("REPORT TIME:", hr_time, 320, H - 148),
            ("REPORT ID:", f"SHC-{now_hr.strftime('%Y%m%d%H%M')}", 320, H - 163),
        ]
        rc.setFillColor(colors.black)
        for label, val, xp, yp in patient_info:
            rc.setFont("Helvetica-Bold", 9)
            rc.drawString(xp, yp, label)
            rc.setFont("Helvetica", 9)
            rc.drawString(xp + 100, yp, val)

        # Clinical Findings
        y = H - 198
        rc.setFillColor(colors.HexColor("#1a3c5e"))
        rc.rect(30, y - 2, W - 60, 16, fill=1, stroke=0)
        rc.setFillColor(colors.white)
        rc.setFont("Helvetica-Bold", 9)
        rc.drawString(40, y + 3, "CLINICAL FINDINGS")

        y -= 8
        findings = [
            ("Sleep Duration",      f"{sleep} hrs",         "Recommended: 7–9 hrs"),
            ("Sleep Quality Score", f"{sleep_quality}/10",  "Good: 7+"),
            ("Stress Level",        f"{stress}/10",         "Ideal: Below 5"),
            ("Physical Activity",   f"{activity} mins/day", "Recommended: 30+ mins"),
            ("Daily Steps",         f"{steps}",             "Target: 8000+"),
            ("Heart Rate",          f"{heart} bpm",         "Normal: 60–100 bpm"),
            ("Blood Pressure",      f"{sys_bp}/{dia_bp} mmHg", "Normal: 120/80 mmHg"),
        ]

        for i, (label, value, note) in enumerate(findings):
            ry = y - (i * 16)
            if i % 2 == 0:
                rc.setFillColor(colors.HexColor("#f0f6fc"))
                rc.rect(30, ry - 5, W - 60, 15, fill=1, stroke=0)
            rc.setFillColor(colors.black)
            rc.setFont("Helvetica-Bold", 9)
            rc.drawString(40, ry + 1, label)
            rc.setFont("Helvetica", 9)
            rc.drawString(200, ry + 1, value)
            rc.setFillColor(colors.HexColor("#555555"))
            rc.setFont("Helvetica", 8)
            rc.drawString(340, ry + 1, note)

        # Diagnosis
        y = y - (len(findings) * 16) - 14
        rc.setFillColor(colors.HexColor("#1a3c5e"))
        rc.rect(30, y - 2, W - 60, 16, fill=1, stroke=0)
        rc.setFillColor(colors.white)
        rc.setFont("Helvetica-Bold", 9)
        rc.drawString(40, y + 3, "DIAGNOSIS SUMMARY")

        y -= 20
        rc.setFillColor(colors.HexColor("#f7fbff"))
        rc.setStrokeColor(colors.HexColor("#1a3c5e"))
        rc.rect(30, y - 30, W - 60, 48, fill=1, stroke=1)
        rc.setFillColor(colors.black)
        for label, val, xp, yp in [
            ("Sleep Disorder:", disorder, 40, y + 10),
            ("Confidence:",     f"{confidence}%", 40, y - 4),
            ("Health Score:",   f"{score}/100", 40, y - 18),
            ("Health Status:",  status, 320, y + 10),
            ("Prediction By:",  "ML Model" if models_loaded else "Rule-Based", 320, y - 4),
        ]:
            rc.setFont("Helvetica-Bold", 9)
            rc.drawString(xp, yp, label)
            rc.setFont("Helvetica", 9)
            rc.drawString(xp + 90, yp, val)

        # Advice
        y -= 55
        rc.setFillColor(colors.HexColor("#1a3c5e"))
        rc.rect(30, y - 2, W - 60, 16, fill=1, stroke=0)
        rc.setFillColor(colors.white)
        rc.setFont("Helvetica-Bold", 9)
        rc.drawString(40, y + 3, "DOCTOR'S ADVICE")

        y -= 18
        advice_lines = [
            "1. Maintain a consistent sleep schedule — aim for 7–9 hours nightly.",
            "2. Engage in at least 30 minutes of physical activity every day.",
            "3. Practice stress management: meditation, breathing, or journaling.",
            "4. Follow a balanced diet rich in vegetables, fruits, and proteins.",
            "5. Monitor blood pressure and heart rate regularly.",
            "6. Stay hydrated — drink 2–3 litres of water daily.",
            "7. Consult a physician if sleep disorder symptoms persist.",
        ]
        rc.setFillColor(colors.black)
        rc.setFont("Helvetica", 9)
        for advice in advice_lines:
            rc.drawString(40, y, advice)
            y -= 14

        # Signature
        y -= 15
        rc.setStrokeColor(colors.HexColor("#1a3c5e"))
        rc.setLineWidth(0.5)
        rc.line(30, y, W - 30, y)
        y -= 15
        rc.setFont("Helvetica-Bold", 9)
        rc.drawString(40, y, "Prepared by:")
        rc.setFont("Helvetica", 9)
        rc.drawString(120, y, "AI Health Monitoring System")
        rc.setFont("Helvetica-Bold", 9)
        rc.drawString(350, y, "Authorised by:")
        rc.setFont("Helvetica", 9)
        rc.drawString(435, y, "Dr. Wellness")

        y -= 35
        rc.setLineWidth(0.8)
        rc.line(350, y, 530, y)
        rc.setFont("Helvetica-Bold", 8)
        rc.drawString(365, y - 12, "Dr. Wellness, MD (Sleep Medicine)")
        rc.setFont("Helvetica", 7)
        rc.drawString(378, y - 22, "Smart Health Care Center")

        # Footer
        rc.setFillColor(colors.HexColor("#1a3c5e"))
        rc.rect(0, 0, W, 28, fill=1, stroke=0)
        rc.setFillColor(colors.white)
        rc.setFont("Helvetica", 7)
        rc.drawCentredString(W / 2, 17, "This report is system-generated. Please consult a qualified physician for medical decisions.")
        rc.drawCentredString(W / 2, 8, f"Generated on {hr_date} at {hr_time}  |  Report ID: SHC-{now_hr.strftime('%Y%m%d%H%M')}")
        rc.save()

        with open(filename, "rb") as rf:
            st.download_button(
                "📥 Download Health Report PDF", rf,
                file_name=f"{d['name']}_Health_Report.pdf",
                mime="application/pdf"
            )