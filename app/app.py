"""
CareerCompass - Interactive Web Application & Agent Demo
CSE395 Supervised Learning Project
"""

import sys
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommend import (
    recommend_careers,
    INTEREST_COLUMNS,
)
from src.roadmap import (
    load_profiles,
    find_occupation,
    generate_roadmap,
)

st.set_page_config(
    page_title="CareerCompass — Career & Education Advisor",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PRESET_PROFILES = {
    "Custom Student Profile": {
        "interests": {
            "practical": 2.5,
            "analytical": 2.5,
            "creative": 2.5,
            "social": 2.5,
            "business": 2.5,
            "organized": 2.5,
        },
        "skills": {
            "Mathematics": 2.5,
            "Critical Thinking": 2.5,
            "Active Learning": 2.5,
            "Science": 2.0,
            "Writing": 2.0,
            "Speaking": 2.0,
        },
    },
    "Profile 1: Computer Science, Data Systems & AI": {
        "interests": {
            "practical": 1.5,
            "analytical": 5.0,
            "creative": 1.0,
            "social": 0.5,
            "business": 2.0,
            "organized": 4.0,
        },
        "skills": {
            "Mathematics": 2.5,
            "Critical Thinking": 3.0,
            "Active Learning": 3.0,
            "Writing": 2.0,
            "Speaking": 1.5,
        },
    },
    "Profile 2: Visual Arts & Creative Design": {
        "interests": {
            "practical": 3.0,
            "analytical": 2.0,
            "creative": 5.0,
            "social": 1.5,
            "business": 2.0,
            "organized": 2.0,
        },
        "skills": {
            "Writing": 3.0,
            "Critical Thinking": 2.5,
            "Active Listening": 3.0,
            "Speaking": 2.0,
            "Mathematics": 1.5,
        },
    },
    "Profile 3: Healthcare, Medicine & Biomedical Research": {
        "interests": {
            "practical": 1.5,
            "analytical": 5.0,
            "creative": 0.5,
            "social": 4.5,
            "business": 1.5,
            "organized": 2.5,
        },
        "skills": {
            "Science": 3.5,
            "Critical Thinking": 3.0,
            "Reading Comprehension": 3.5,
            "Mathematics": 2.5,
            "Active Listening": 3.5,
        },
    },
}

# SIDEBAR CONTROLS
st.sidebar.title("🧭 CareerCompass")
st.sidebar.markdown("**Education-to-Career Planning Agent**")

preset_choice = st.sidebar.selectbox(
    "Choose Student Profile:",
    list(PRESET_PROFILES.keys()),
    index=1,
)

current_preset = PRESET_PROFILES[preset_choice]

st.sidebar.markdown("---")
st.sidebar.subheader("1. Interest Areas (0–5)")

interest_display_labels = {
    "practical": "Practical & Hands-on",
    "analytical": "Analytical & Scientific",
    "creative": "Creative & Design",
    "social": "Social & Community",
    "business": "Business & Leadership",
    "organized": "Structured & Detail-Oriented",
}

interests = {}
for key, label in interest_display_labels.items():
    default_val = current_preset["interests"].get(key, 2.5)
    interests[key] = st.sidebar.slider(
        label,
        min_value=0.0,
        max_value=5.0,
        value=float(default_val),
        step=0.5,
    )

st.sidebar.markdown("---")
st.sidebar.subheader("2. Self-Reported Skills (0–5)")
all_skills = [
    "Mathematics",
    "Critical Thinking",
    "Active Learning",
    "Active Listening",
    "Reading Comprehension",
    "Science",
    "Speaking",
    "Writing",
    "Monitoring",
    "Learning Strategies",
]

skills = {}
for skill in all_skills:
    default_val = current_preset["skills"].get(skill, 2.0)
    skills[skill] = st.sidebar.slider(
        skill,
        min_value=0.0,
        max_value=5.0,
        value=float(default_val),
        step=0.5,
    )

top_n = st.sidebar.slider("Number of Recommendations", min_value=3, max_value=10, value=5)

# HEADER
st.markdown('<div class="main-header">🧭 CareerCompass — Career & Education Advisor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Step Educational Roadmap: <b>Courses → Skills → Projects → Career</b> | CSE395 Supervised Learning Project</div>',
    unsafe_allow_html=True,
)

# TABS
tab_recs, tab_roadmap, tab_ml = st.tabs([
    "🎯 Career Matches & Recommendations",
    "🗺️ Multi-Step Educational Roadmap",
    "📊 Supervised Learning (Random Forest) Model",
])

# Run recommendation pipeline
try:
    with st.spinner("Analyzing profile using Scikit-Learn KNN & Random Forest Classifier..."):
        recommendations = recommend_careers(interests, skills, top_n=top_n)
        profiles = load_profiles()
except Exception as e:
    st.error(f"Error loading recommendations: {e}")
    st.stop()

# TAB 1: RECOMMENDATIONS
with tab_recs:
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Total Occupations", "910 Profiles")
    with col_kpi2:
        st.metric("Top Career Match", recommendations.iloc[0]["Title"])
    with col_kpi3:
        st.metric("Match Score", f"{recommendations.iloc[0]['Match Score (%)']:.1f}%")
    with col_kpi4:
        pred_z = recommendations.iloc[0].get("Predicted Job Zone", "N/A")
        st.metric("ML Predicted Zone", f"Zone {pred_z}")

    st.markdown("### Top Career Recommendations")
    st.markdown("Occupations matched via Scikit-Learn Nearest Neighbors and evaluated with Random Forest Job Zone classification:")

    display_df = recommendations[[
        "O*NET-SOC Code", "Title", "Job Zone", "Predicted Job Zone",
        "Interest Match (%)", "Skill Match (%)", "Match Score (%)"
    ]].copy()

    display_df["Match Score (%)"] = display_df["Match Score (%)"].map("{:.1f}%".format)
    display_df["Interest Match (%)"] = display_df["Interest Match (%)"].map("{:.1f}%".format)
    display_df["Skill Match (%)"] = display_df["Skill Match (%)"].map("{:.1f}%".format)

    st.dataframe(display_df, use_container_width=True)
    st.info("💡 Switch to the **'Multi-Step Educational Roadmap'** tab to generate your complete Courses → Skills → Career pathway.")

# TAB 2: ROADMAP
with tab_roadmap:
    career_titles = recommendations["Title"].tolist()
    selected_title = st.selectbox("Select Target Career for Personalized Roadmap:", career_titles, index=0)

    selected_rec = recommendations[recommendations["Title"] == selected_title].iloc[0]
    occupation_data = find_occupation(profiles, occupation_code=selected_rec["O*NET-SOC Code"])

    roadmap = generate_roadmap(
        occupation_data,
        skills,
        predicted_job_zone=selected_rec.get("Predicted Job Zone"),
    )

    st.markdown(f"## 🗺️ Roadmap for: **{roadmap['title']}**")
    st.markdown(f"**Code:** `{roadmap['code']}` | **Domain:** `{roadmap['domain']}` | **Job Zone:** `{roadmap['job_zone']}` | **Random Forest Predicted Zone:** `{roadmap.get('predicted_job_zone')}`")

    if roadmap.get("description"):
        st.markdown(f"> *{roadmap['description']}*")

    st.markdown("---")

    # STEP 1: COURSES
    st.markdown("### 🎓 Step 1: Domain-Specific Coursework Roadmap (Courses)")
    st.markdown("Curated curriculum tailored to this specific occupational domain across foundational, intermediate, and advanced stages:")

    cr = roadmap["course_roadmap"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Stage 1: Foundational Principles**")
        for c in cr["stage_1_foundational"]:
            st.markdown(f"- 📘 {c}")
    with c2:
        st.markdown("**Stage 2: Core & Major Discipline**")
        for c in cr["stage_2_core"]:
            st.markdown(f"- 📙 {c}")
    with c3:
        st.markdown("**Stage 3: Advanced Specialization**")
        if cr.get("stage_3_advanced"):
            for c in cr["stage_3_advanced"]:
                st.markdown(f"- 🔬 {c}")
        else:
            st.markdown("*Focus on practical studio/apprenticeship mastery.*")

    if cr.get("certifications"):
        st.markdown("#### 🏆 Recommended Professional Credentials")
        cert_cols = st.columns(len(cr["certifications"]))
        for idx, cert in enumerate(cr["certifications"]):
            with cert_cols[idx]:
                st.info(f"**{cert}**")

    st.markdown("---")

    # STEP 2: SKILLS
    st.markdown("### ⚡ Step 2: Skill Acquisition & Gap Remediation (Skills)")
    gaps = roadmap["gaps"]
    if gaps:
        st.markdown("Skill gaps calculated against benchmark requirements:")
        gap_rows = []
        for g in gaps:
            gap_rows.append({
                "Skill": g["skill"],
                "Current Rating": f"{g['current']:.1f} / 5.0",
                "Target Requirement": f"{g['importance']:.2f} / 5.0",
                "Deficiency Gap": f"+{g['gap']:.2f}",
                "Actionable Exercise": g["actionable_exercise"],
            })
        st.dataframe(pd.DataFrame(gap_rows), use_container_width=True)
    else:
        st.success("All assessed skills meet or exceed target occupational benchmarks!")

    st.markdown("---")

    # STEP 3: PRACTICAL PROJECTS
    st.markdown("### 🛠️ Step 3: Practical Projects & Portfolio Proof-of-Work")
    for idx, proj in enumerate(roadmap["projects"], 1):
        st.markdown(f"**{idx}.** {proj}")

    st.markdown("---")

    # STEP 4: CAREER MILESTONES
    st.markdown("### 🚀 Step 4: Career Entry Milestones & Job Zone Preparation (Career)")
    st.markdown("**Educational Requirements:**")
    for item in roadmap["education_guidance"]:
        st.markdown(f"- {item}")

    st.markdown("**Chronological Career Transition Milestones:**")
    for m in roadmap["career_milestones"]:
        st.markdown(f"- 📌 {m}")

# TAB 3: SUPERVISED ML DIAGNOSTICS
with tab_ml:
    st.markdown("## 🧠 CSE395 Supervised Learning: Random Forest Classifier")
    st.markdown(
        """
        In CareerCompass, **Supervised Learning** is implemented using a Scikit-Learn **Random Forest Classifier**
        to predict an occupation's required **Job Zone (Preparation Level 1 to 5)** from 49 numerical attributes
        (essential skills, knowledge domains, and interest vectors).
        """
    )

    metrics_file = PROJECT_ROOT / "outputs" / "evaluation_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model Architecture", "Random Forest (Scikit-Learn)")
        m2.metric("Macro OVR ROC-AUC", f"{metrics_data.get('macro_ovr_roc_auc', 0):.4f}")
        test_m = metrics_data.get("test_metrics", {})
        m3.metric("Test Accuracy", f"{test_m.get('accuracy', 0) * 100:.1f}%")
        m4.metric("Test Macro F1", f"{test_m.get('macro_f1', 0):.4f}")

    col_img1, col_img2 = st.columns(2)
    cm_path = PROJECT_ROOT / "outputs" / "confusion_matrix.png"
    roc_path = PROJECT_ROOT / "outputs" / "roc_curves.png"

    with col_img1:
        st.markdown("### Confusion Matrix")
        if cm_path.exists():
            st.image(str(cm_path), caption="Random Forest Confusion Matrix on Held-Out Test Set")

    with col_img2:
        st.markdown("### ROC Curves (One-vs-Rest)")
        if roc_path.exists():
            st.image(str(roc_path), caption="Multiclass ROC Curves across Job Zones")
