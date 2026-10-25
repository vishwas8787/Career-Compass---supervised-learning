# 🧭 CareerCompass — Career & Education Advisor
> **CSE395 Supervised Learning Project**  
> **Course:** CSE395 Supervised Learning | Lovely Professional University (LPU)  
> **Dataset Source:** Occupational Information Network (O*NET) Database  

---

## 📌 Executive Summary & Problem Statement

Students and career switchers frequently lack a single, unified system for **personalized education-to-career planning**. Conventional career guidance tools typically return a single generic job title or static career description without actionable educational bridges.

**CareerCompass** solves this problem by functioning as an intelligent advisory agent. Given a student's interests and self-reported skills, CareerCompass matches the student to high-affinity careers from **910+ verified occupations** using **K-Nearest Neighbors (KNN)**, predicts the required preparation level using a **Random Forest Classifier**, and generates a **concrete, domain-specific multi-step roadmap**:

$$\mathbf{Courses \longrightarrow Skills \longrightarrow Projects \longrightarrow Career}$$

---

## 🎯 Deliverables Checklist

| Deliverable | Location in Repository | Status | Description |
| :--- | :--- | :---: | :--- |
| **Agent Demo (Web UI)** | [`app/app.py`](./app/app.py) | ✅ **Complete** | Interactive Streamlit Web Demo with interest sliders, skill ratings, live match visualization, and one-click roadmap exports. |
| **Agent Demo (CLI)** | [`main.py`](./main.py) | ✅ **Complete** | Command-line interface with interactive assessment and automated 1-click demo (`python main.py --demo`). |
| **3 Sample Generated Roadmaps** | [`outputs/sample_roadmaps/`](./outputs/sample_roadmaps/) | ✅ **Complete** | Three comprehensive, domain-accurate multi-step roadmaps across distinct student personas. |
| **Supervised Learning Model** | [`models/job_zone_model.joblib`](./models/job_zone_model.joblib) | ✅ **Complete** | Scikit-Learn `RandomForestClassifier` predicting required preparation/education Job Zone (ROC-AUC: **0.9505**). |
| **Automated Test Suite** | [`tests/`](./tests/) | ✅ **Complete** | 18 unit tests covering recommendation logic, roadmap generation, and model inference (`python -m unittest discover tests`). |

---

## 🗺️ The 3 Sample Generated Roadmaps

As required by the project deliverables, 3 comprehensive sample roadmaps have been generated and documented in [`outputs/sample_roadmaps/`](./outputs/sample_roadmaps/):

1. **[Sample 1: Computer Science, Data Systems & AI](./outputs/sample_roadmaps/sample_roadmap_1_tech_ai.md)**  
   * **Persona:** Alex Chen (Second-year undergraduate student interested in data engineering, AI, and databases).  
   * **Target Career:** **Database Architects** (`15-1243.00`) | Job Zone: `4.0` | Random Forest Predicted Zone: `4.0` | Match Score: `94.3%`.  
   * **Pathway:** Programming Fundamentals $\rightarrow$ Data Structures & SQL $\rightarrow$ Cloud & Distributed Systems $\rightarrow$ Enterprise Solutions Architect Credential $\rightarrow$ Production Data Pipeline Project.

2. **[Sample 2: Creative Digital Media & Video Game Design](./outputs/sample_roadmaps/sample_roadmap_2_creative_game_design.md)**  
   * **Persona:** Maya Rodriguez (Design student combining visual artistry, storytelling, and interactive gameplay mechanics).  
   * **Target Career:** **Video Game Designers** (`15-1255.01`) | Job Zone: `4.0` | Random Forest Predicted Zone: `4.0` | Match Score: `91.3%`.  
   * **Pathway:** Visual Design Fundamentals $\rightarrow$ 3D Asset Modeling & Game Engine Systems $\rightarrow$ Interactive Scripting $\rightarrow$ Playable Core Gameplay Prototype $\rightarrow$ Game Design Document (GDD).

3. **[Sample 3: Healthcare Informatics & Biotechnology](./outputs/sample_roadmaps/sample_roadmap_3_healthcare_biotech.md)**  
   * **Persona:** Marcus Patel (Biomedical science student focused on health analytics, epidemiology, and clinical data).  
   * **Target Career:** **Health Informatics Specialists** (`15-1211.01`) | Job Zone: `5.0` | Random Forest Predicted Zone: `5.0` | Match Score: `92.6%`.  
   * **Pathway:** General Biology & Anatomy $\rightarrow$ Health Information Systems & Biostatistics $\rightarrow$ Clinical Epidemiology $\rightarrow$ Health Information Credential $\rightarrow$ Clinical Trial Workflow Audit.

An executive index summary is available in [`outputs/sample_roadmaps/README.md`](./outputs/sample_roadmaps/README.md).

---

## 🧠 CSE395 Supervised Learning Architecture

The system uses standard **Scikit-Learn** algorithms taught in CSE395:
1. **Candidate Retrieval & Matching:** Scikit-Learn `NearestNeighbors` (KNN with cosine metric) matches student preference and skill vectors to target career profiles.
2. **Supervised Classification:** Scikit-Learn `RandomForestClassifier` (200 estimators, balanced class weights) predicts the required **Job Zone (Preparation Levels 1–5)** based on 49 numerical features (skills, knowledge domains, and interest attributes).

```
+----------------------------------------------------------------------------------+
|                              INPUT PROFILE (Student)                             |
|  * 6 Dimension Interest Vector (Practical, Analytical, Creative, Social, ...)    |
|  * 10 Essential Skill Ratings (Mathematics, Critical Thinking, Science, ...)     |
+-----------------------------------------+----------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                    K-NEAREST NEIGHBORS (KNN) AFFINITY MATCHING                   |
|  * Scikit-Learn NearestNeighbors on Standardized Occupational Feature Space      |
|  * Distance-to-Affinity Mapping with Skill Deficiency Penalty                    |
|  * Major SOC Stratification (prevents over-concentration in a single sector)     |
+-----------------------------------------+----------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                SUPERVISED RANDOM FOREST CLASSIFIER (Scikit-Learn)                |
|  * Multi-class classification: Job Zones 2, 3, 4, 5                              |
|  * 5-Fold Cross-Validation Accuracy: 77.1%                                       |
|  * Held-Out Test Accuracy: 76.9% | Macro F1: 0.757 | Macro OVR ROC-AUC: 0.9505   |
+-----------------------------------------+----------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                DOMAIN-SPECIFIC MULTI-STEP ROADMAP (Courses -> Career)            |
|  1. Courses: Domain-Matched Curriculum (Foundational -> Core -> Advanced)        |
|  2. Skills: Gap Analysis + Domain-Specific Remedial Exercises                    |
|  3. Projects: 3 Tailored Portfolio Milestones with Proof-of-Work                 |
|  4. Career: Job Zone Education Level + Credentialing + 4-Year Milestone Plan     |
+----------------------------------------------------------------------------------+
```

### Model Performance Summary

| Metric | Random Forest Model Value |
| :--- | :---: |
| **5-Fold CV Accuracy** | **77.07% (± 3.98%)** |
| **Held-Out Test Accuracy** | **76.92%** |
| **Balanced Accuracy** | **75.62%** |
| **Macro F1-Score** | **0.7569** |
| **Macro OVR ROC-AUC** | **0.9505** |

*Diagnostics and evaluation artifacts:*
- Confusion Matrix: [`outputs/confusion_matrix.png`](./outputs/confusion_matrix.png)
- Multiclass ROC Curves: [`outputs/roc_curves.png`](./outputs/roc_curves.png)
- Serialized JSON metrics: [`outputs/evaluation_metrics.json`](./outputs/evaluation_metrics.json)

---

## 📂 Project Directory Structure

```text
career-compass-project/
│
├── app/
│   └── app.py                      # Interactive Streamlit Web Demo Application
│
├── data/
│   ├── all datasets/               # Complete Occupational Database (CSV files)
│   ├── raw/                        # Extracted raw tables (skills, knowledge, interests, etc.)
│   └── processed/
│       ├── occupation_profiles.csv # Merged occupation text + numerical features
│       └── occupation_features.csv # Cleaned numerical matrix for ML training
│
├── models/
│   └── job_zone_model.joblib       # Trained RandomForestClassifier model & metadata
│
├── outputs/
│   ├── sample_roadmaps/            # Deliverable: 3 sample generated roadmaps
│   │   ├── README.md               # Summary index of sample roadmaps
│   │   ├── sample_roadmap_1_tech_ai.md
│   │   ├── sample_roadmap_2_creative_game_design.md
│   │   └── sample_roadmap_3_healthcare_biotech.md
│   ├── classification_report.txt   # Model evaluation report
│   ├── confusion_matrix.png        # Multiclass confusion matrix
│   ├── roc_curves.png              # One-vs-Rest multiclass ROC curves
│   └── evaluation_metrics.json     # Serialized performance metrics
│
├── src/
│   ├── data_loader.py              # Data ingestion and inspection utilities
│   ├── preprocessing.py            # Feature aggregation and pivot transformation
│   ├── feature_engineering.py      # Feature matrix validation and selection
│   ├── train.py                    # Random Forest model training & evaluation
│   ├── recommend.py                # KNN career ranking & Random Forest prediction
│   ├── roadmap.py                  # Domain-specific (Courses -> Skills -> Career) generator
│   └── generate_sample_roadmaps.py # Automated generator for sample deliverables
│
├── tests/
│   ├── test_recommend.py           # Unit tests for similarity and ranking logic
│   ├── test_roadmap.py             # Unit tests for roadmap generation
│   └── test_model.py               # Unit tests for ML inference and feature alignment
│
├── main.py                         # CLI entry point (Interactive & Demo modes)
├── requirements.txt                # Project dependencies
└── README.md                       # Academic project documentation
```

---

## 🚀 How to Run

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Web Demo (Streamlit)

```bash
streamlit run app/app.py
```

### 3. Run the CLI Agent Demo

```bash
python main.py --demo
```

Or for interactive testing:

```bash
python main.py
```

### 4. Run the Automated Test Suite

```bash
python -m unittest discover tests
```

### 5. Retrain the Random Forest Model

```bash
python src/train.py
```

---

## 👥 Authors & Academic Context

- **Course:** CSE395 Supervised Learning  
- **Institution:** Lovely Professional University (LPU)
