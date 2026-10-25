# 🧭 CareerCompass — Career & Education Advisor
> **CSE395 Supervised Learning Project**  
> **Course:** CSE395 Supervised Learning | Lovely Professional University (LPU)  
> **Dataset:** O*NET Database (v31.0) — [onetcenter.org](https://www.onetcenter.org)  

---

## 📌 Executive Summary & Problem Statement

Students and career switchers frequently lack a single, unified system for **personalized education-to-career planning**. Conventional career guidance tools typically return a single generic job title or static career description without actionable educational bridges.

**CareerCompass** solves this problem by functioning as an intelligent advisory agent. Given a student's Holland RIASEC interests and self-reported skills, CareerCompass matches the student to high-affinity careers from **910+ verified O\*NET occupations** and produces a **concrete multi-step roadmap**:

$$\mathbf{Courses \longrightarrow Skills \longrightarrow Projects \longrightarrow Career}$$

---

## 🎯 Deliverables Checklist

| Deliverable | Location in Repository | Status | Description |
| :--- | :--- | :---: | :--- |
| **Agent Demo (Web UI)** | [`app/app.py`](./app/app.py) | ✅ **Complete** | Interactive Streamlit Web Demo with RIASEC sliders, skill ratings, live match visualization, and one-click roadmap exports. |
| **Agent Demo (CLI)** | [`main.py`](./main.py) | ✅ **Complete** | Command-line interface with interactive assessment and automated 1-click demo (`python main.py --demo`). |
| **3 Sample Generated Roadmaps** | [`outputs/sample_roadmaps/`](./outputs/sample_roadmaps/) | ✅ **Complete** | Three comprehensive, publication-ready multi-step roadmaps across distinct student personas. |
| **Supervised Learning Model** | [`models/job_zone_model.joblib`](./models/job_zone_model.joblib) | ✅ **Complete** | ExtraTreesClassifier predicting O*NET Job Zone with out-of-fold threshold tuning (ROC-AUC: **0.950**). |
| **Automated Test Suite** | [`tests/`](./tests/) | ✅ **Complete** | 18 unit tests covering recommendation logic, roadmap generation, and model inference (`python -m unittest discover tests`). |

---

## 🗺️ The 3 Sample Generated Roadmaps

As required by the project deliverables, 3 comprehensive sample roadmaps have been generated and documented in [`outputs/sample_roadmaps/`](./outputs/sample_roadmaps/):

1. **[Sample 1: Computer Science, Data Systems & AI](./outputs/sample_roadmaps/sample_roadmap_1_tech_ai.md)**  
   * **Persona:** Alex Chen (Second-year undergraduate student interested in data engineering, AI, and databases).  
   * **Target Career:** **Database Architects** (`15-1243.00`) | Actual Job Zone: `4.0` | ML Predicted Zone: `4.0` | Match Score: `94.3%`.  
   * **Pathway:** Foundational CS/Math $\rightarrow$ Data Structures & SQL $\rightarrow$ Cloud & Distributed Systems $\rightarrow$ AWS Solutions Architect certification $\rightarrow$ Hands-on Data Warehouse pipeline.

2. **[Sample 2: Creative Digital Media & Video Game Design](./outputs/sample_roadmaps/sample_roadmap_2_creative_game_design.md)**  
   * **Persona:** Maya Rodriguez (Design student combining visual artistry, storytelling, and interactive gameplay mechanics).  
   * **Target Career:** **Video Game Designers** (`15-1255.01`) | Actual Job Zone: `4.0` | ML Predicted Zone: `4.0` | Match Score: `91.3%`.  
   * **Pathway:** Visual Design & Figma $\rightarrow$ 3D Modeling & Unity Engine $\rightarrow$ Playable Core Gameplay Prototype $\rightarrow$ Game Design Document (GDD) $\rightarrow$ Junior Systems Designer.

3. **[Sample 3: Healthcare Informatics & Biotechnology](./outputs/sample_roadmaps/sample_roadmap_3_healthcare_biotech.md)**  
   * **Persona:** Marcus Patel (Biomedical science student focused on health analytics, epidemiology, and clinical data).  
   * **Target Career:** **Health Informatics Specialists** (`15-1211.01`) | Actual Job Zone: `5.0` | ML Predicted Zone: `5.0` | Match Score: `92.6%`.  
   * **Pathway:** Cellular Biology & General Chemistry $\rightarrow$ Health Informatics & Biostatistics $\rightarrow$ Clinical Epidemiology $\rightarrow$ CPHIMS Credential $\rightarrow$ Clinical Trial Analytics Project.

An executive index summary is available in [`outputs/sample_roadmaps/README.md`](./outputs/sample_roadmaps/README.md).

---

## 🧠 CSE395 Supervised Learning Architecture

The system uses Supervised Machine Learning to predict an occupation's required **Job Zone (Preparation Levels 1–5)** based on 49 numerical attributes from the O\*NET Content Model.

```
+----------------------------------------------------------------------------------+
|                              INPUT PROFILE (Student)                             |
|  * 6 Holland RIASEC Interests (Realistic, Investigative, Artistic, Social, ...)   |
|  * 10 Essential Skill Self-Ratings (Mathematics, Critical Thinking, ...)        |
+-----------------------------------------+----------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                         CAREER MATCHING & DIVERSITY FILTER                       |
|  * Cosine Similarity on Holland Interest Vector                                  |
|  * Penalty function on Essential Skill Deficiencies                              |
|  * Major Group SOC Stratification (prevents over-concentration in 1 domain)       |
+-----------------------------------------+----------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                   SUPERVISED LEARNING CLASSIFIER (ExtraTrees)                    |
|  * 49 Features (Skills + Knowledge Domains + Interests)                          |
|  * Multi-class classification: Job Zone 2, 3, 4, 5                               |
|  * Out-of-fold probability threshold tuning for Zone 3 recall                    |
|  * Test Accuracy: 78.6% | Macro F1: 0.781 | Macro OVR ROC-AUC: 0.950             |
+-----------------------------------------+----------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                MULTI-STEP ROADMAP GENERATION (Courses -> Career)                 |
|  1. Courses: Foundational (Sem 1-2) -> Core (Sem 3-4) -> Advanced Electives      |
|  2. Skills: O*NET Gap Analysis + Priority Remedial Exercises                     |
|  3. Projects: 3 Tiered Portfolio Milestones with Proof-of-Work                   |
|  4. Career: Job Zone Education Level + Credentialing + 4-Year Milestone Plan     |
+----------------------------------------------------------------------------------+
```

### Model Performance Summary

| Metric | Argmax Baseline | Threshold-Adjusted Rule (`threshold=0.375`) |
| :--- | :---: | :---: |
| **Accuracy** | 77.5% | **78.6%** |
| **Balanced Accuracy** | 75.8% | **78.0%** |
| **Macro F1-Score** | 0.758 | **0.781** |
| **Zone 3 Precision** | 70.4% | **66.7%** |
| **Zone 3 Recall** | 46.3% | **63.4% (+17.1%)** |
| **Macro OVR ROC-AUC** | **0.9503** | **0.9503** |

*Diagnostics and curves:*
- Confusion Matrix: [`outputs/confusion_matrix.png`](./outputs/confusion_matrix.png)
- Multiclass ROC Curves: [`outputs/roc_curves.png`](./outputs/roc_curves.png)
- Detailed JSON metrics: [`outputs/evaluation_metrics.json`](./outputs/evaluation_metrics.json)

---

## 📂 Project Directory Structure

```text
career-compass-project/
│
├── app/
│   └── app.py                      # Interactive Streamlit Web Demo Application
│
├── data/
│   ├── all datasets/               # Complete O*NET 31.0 Database (CSV files)
│   ├── raw/                        # Extracted raw tables (skills, knowledge, interests, etc.)
│   └── processed/
│       ├── occupation_profiles.csv # Merged occupation text + numerical features
│       └── occupation_features.csv # Cleaned numerical matrix for ML training
│
├── models/
│   └── job_zone_model.joblib       # Trained ExtraTreesClassifier model & metadata
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
│   ├── evaluation_metrics.json     # Serialized performance metrics
│   └── threshold_comparison.csv    # Zone 3 sweep comparison table
│
├── src/
│   ├── data_loader.py              # Data ingestion and inspection utilities
│   ├── preprocessing.py            # Feature aggregation and pivot transformation
│   ├── feature_engineering.py      # Feature matrix validation and selection
│   ├── train.py                    # Supervised learning model training & evaluation
│   ├── recommend.py                # Career ranking & cosine similarity engine
│   ├── roadmap.py                  # Multi-step (Courses -> Skills -> Career) generator
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

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Web Demo (Streamlit)

Launch the interactive UI in your web browser:

```bash
streamlit run app/app.py
```

*Features available in the UI:*
- Select pre-configured student personas (Persona 1, 2, 3) or adjust custom sliders.
- View real-time career match scores and supervised ML Job Zone predictions.
- Explore the interactive multi-step roadmap across tabs.
- Download customized roadmaps directly as Markdown files.

### 3. Run the CLI Agent Demo

To run the automated agent demonstration without typing manual inputs:

```bash
python main.py --demo
```

To run an interactive session where you input your own ratings:

```bash
python main.py
```

To run a specific preset student persona from the terminal:

```bash
python main.py --persona 2
```

### 4. Regenerate Deliverable Roadmaps

To regenerate the 3 deliverable roadmaps in `outputs/sample_roadmaps/`:

```bash
python -m src.generate_sample_roadmaps
```

### 5. Run the Automated Test Suite

Run all 18 unit tests using Python's built-in `unittest` runner:

```bash
python -m unittest discover tests
```

### 6. Retrain the Supervised Learning Model

To reproduce the model training, cross-validation, and metric evaluations:

```bash
python src/train.py
```

---

## 👥 Authors & Academic Context

- **Course:** CSE395 Supervised Learning  
- **Institution:** Lovely Professional University  
- **Dataset Reference:** U.S. Department of Labor, *Occupational Information Network (O\*NET)* Database Release 31.0.
