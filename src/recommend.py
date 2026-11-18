from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "job_zone_model.joblib"

INTEREST_WEIGHT = 0.80
SKILL_WEIGHT = 0.20

# Canonical feature mapping (accepts both plain names and dataset labels)
INTEREST_COLUMNS = {
    "practical": "interest_Realistic",
    "realistic": "interest_Realistic",
    "analytical": "interest_Investigative",
    "investigative": "interest_Investigative",
    "creative": "interest_Artistic",
    "artistic": "interest_Artistic",
    "social": "interest_Social",
    "business": "interest_Enterprising",
    "enterprising": "interest_Enterprising",
    "organized": "interest_Conventional",
    "conventional": "interest_Conventional",
}


def load_resources():
    """Load occupation profiles and the trained Random Forest model."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Occupation features not found. Run preprocessing.py first."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model not found. Run src/train.py first."
        )

    profiles = pd.read_csv(DATA_FILE)
    saved = joblib.load(MODEL_FILE)

    return (
        profiles,
        saved["model"],
        saved["feature_columns"],
        saved.get("classes", [2, 3, 4, 5]),
    )


def validate_ratings(ratings, label):
    """Validate student ratings on a 0-5 scale."""
    validated = {}

    for name, value in ratings.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} rating for '{name}' must be numeric.")

        if not np.isfinite(value) or not 0 <= value <= 5:
            raise ValueError(
                f"{label} rating for '{name}' must be between 0 and 5."
            )

        validated[name] = value

    return validated


def calculate_interest_scores(profiles, interests):
    """Calculate career affinity using Scikit-Learn NearestNeighbors metric."""
    resolved_columns = {}
    for user_key, val in interests.items():
        col = INTEREST_COLUMNS.get(user_key.strip().lower())
        if col and col in profiles.columns:
            resolved_columns[col] = val

    if not resolved_columns:
        raise ValueError("At least one interest rating is required.")

    cols = list(set(resolved_columns.keys()))
    user_vec = np.array([resolved_columns[c] for c in cols], dtype=float).reshape(1, -1)

    career_mat = profiles[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

    # Use Scikit-Learn NearestNeighbors with cosine metric
    nn = NearestNeighbors(n_neighbors=min(len(profiles), 200), metric="cosine")
    nn.fit(career_mat)
    distances, indices = nn.kneighbors(user_vec, n_neighbors=len(profiles))

    # Convert cosine distance (0 to 2) to match percentage (0 to 100)
    scores = np.zeros(len(profiles), dtype=float)
    cos_dists = distances[0]
    matched_idx = indices[0]

    similarity = np.clip(1.0 - cos_dists, 0.0, 1.0) * 100.0
    scores[matched_idx] = similarity

    return pd.Series(scores, index=profiles.index)


def calculate_skill_scores(profiles, skills):
    """Score how well the user's current skills meet career requirements."""
    if not skills:
        return pd.Series(0.0, index=profiles.index)

    skill_cols = {
        f"skill_{name.strip()}": rating
        for name, rating in skills.items()
        if f"skill_{name.strip()}" in profiles.columns
    }

    if not skill_cols:
        return pd.Series(0.0, index=profiles.index)

    cols = list(skill_cols.keys())
    user_vec = np.array([skill_cols[c] for c in cols], dtype=float)

    career_mat = profiles[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

    # Penalize only when student's rating falls below career requirement
    gaps = np.maximum(career_mat - user_vec, 0)
    mean_gap = gaps.mean(axis=1)

    scores = (1.0 - mean_gap / 5.0).clip(0.0, 1.0) * 100.0
    return pd.Series(scores, index=profiles.index)


def select_diverse_careers(results, top_n=5, max_per_group=2):
    """Select high-scoring careers while avoiding excessive duplication in a single domain."""
    results = results.copy()
    results["Major Group"] = (
        results["O*NET-SOC Code"]
        .astype(str)
        .str.extract(r"^(\d{2})", expand=False)
    )

    selected_indices = []
    group_counts = {}

    for index, row in results.iterrows():
        group = row["Major Group"] or "Unknown"
        if group_counts.get(group, 0) >= max_per_group:
            continue

        selected_indices.append(index)
        group_counts[group] = group_counts.get(group, 0) + 1

        if len(selected_indices) >= top_n:
            break

    if len(selected_indices) < min(top_n, len(results)):
        for index in results.index:
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= min(top_n, len(results)):
                break

    return (
        results.loc[selected_indices]
        .drop(columns=["Major Group"])
        .reset_index(drop=True)
    )


def recommend_careers(interests, skills=None, top_n=5):
    """Rank careers using Scikit-Learn KNN and predict Job Zone with Random Forest."""
    profiles, rf_model, model_features, classes = load_resources()

    interests = {k.strip().lower(): v for k, v in interests.items()}
    skills = skills or {}

    interests = validate_ratings(interests, "Interest")
    skills = validate_ratings(skills, "Skill")

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    interest_scores = calculate_interest_scores(profiles, interests)
    skill_scores = calculate_skill_scores(profiles, skills)

    if skills:
        match_scores = (
            INTEREST_WEIGHT * interest_scores + SKILL_WEIGHT * skill_scores
        )
    else:
        match_scores = interest_scores

    # Predict Job Zone using the trained Random Forest Classifier
    X = profiles[model_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rf_predictions = rf_model.predict(X)

    results = profiles[["O*NET-SOC Code", "Title", "Job Zone"]].copy()
    results["Interest Match (%)"] = interest_scores
    results["Skill Match (%)"] = skill_scores
    results["Match Score (%)"] = match_scores
    results["Predicted Job Zone"] = rf_predictions.astype(float)

    results = results.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)

    return select_diverse_careers(results, top_n=top_n, max_per_group=2)


def main():
    print("CAREERCOMPASS - CAREER RECOMMENDATIONS (RANDOM FOREST + KNN)")

    interests = {
        "creative": 4.5,
        "practical": 3.0,
        "analytical": 2.0,
        "social": 1.5,
        "business": 2.0,
        "organized": 2.0,
    }

    skills = {
        "Critical Thinking": 3.0,
        "Writing": 3.0,
    }

    recommendations = recommend_careers(interests, skills, top_n=5)
    print("\nTop Career Matches:")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
