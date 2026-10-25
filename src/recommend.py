
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"
)

MODEL_FILE = PROJECT_ROOT / "models" / "job_zone_model.joblib"

# Initial ranking weights; we'll evaluate these later.
INTEREST_WEIGHT = 0.80
SKILL_WEIGHT = 0.20

INTEREST_COLUMNS = {
    "realistic": "interest_Realistic",
    "investigative": "interest_Investigative",
    "artistic": "interest_Artistic",
    "social": "interest_Social",
    "enterprising": "interest_Enterprising",
    "conventional": "interest_Conventional",
}


def load_resources():
    """Load occupation profiles and the trained Job Zone model."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Occupation features not found. "
            "Run preprocessing.py and feature_engineering.py first."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Job Zone model not found. Run train.py first."
        )

    profiles = pd.read_csv(DATA_FILE)
    saved = joblib.load(MODEL_FILE)

    return (
    profiles,
    saved["model"],
    saved["feature_columns"],
    saved.get("zone3_threshold", 1.01),
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
    """Calculate career-interest similarity scores on a 0–100 scale."""
    interest_columns = {
        name: column
        for name, column in INTEREST_COLUMNS.items()
        if name in interests
    }

    if not interest_columns:
        raise ValueError("At least one interest rating is required.")

    columns = list(interest_columns.values())
    user_vector = np.array(
        [interests[name] for name in interest_columns],
        dtype=float,
    )

    career_matrix = profiles[columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    user_norm = np.linalg.norm(user_vector)
    career_norms = np.linalg.norm(career_matrix, axis=1)

    scores = np.zeros(len(profiles), dtype=float)
    valid = (
        np.isfinite(career_matrix).all(axis=1)
        & (career_norms > 0)
        & (user_norm > 0)
    )

    if valid.any():
        similarities = (
            career_matrix[valid] @ user_vector
        ) / (career_norms[valid] * user_norm)

        scores[valid] = np.clip(similarities, 0, 1) * 100

    return pd.Series(scores, index=profiles.index)


def calculate_skill_scores(profiles, skills):
    """Score how well the user's skills meet career requirements."""
    if not skills:
        return pd.Series(0.0, index=profiles.index)

    skill_columns = {
        f"skill_{name.strip()}": rating
        for name, rating in skills.items()
        if f"skill_{name.strip()}" in profiles.columns
    }

    if not skill_columns:
        return pd.Series(0.0, index=profiles.index)

    columns = list(skill_columns.keys())

    user_vector = np.array(
        [skill_columns[column] for column in columns],
        dtype=float,
    )

    career_matrix = profiles[columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    valid = np.isfinite(career_matrix).all(axis=1)
    scores = np.zeros(len(profiles), dtype=float)

    if valid.any():
        requirements = career_matrix[valid]

        # Penalize only the amount by which the user's skill
        # rating falls below the occupation's requirement.
        gaps = np.maximum(requirements - user_vector, 0)

        mean_gap = gaps.mean(axis=1)

        scores[valid] = (
            1 - mean_gap / 5.0
        ).clip(0, 1) * 100

    return pd.Series(scores, index=profiles.index)


def select_diverse_careers(results, top_n=5, max_per_group=2):
    """Select high-scoring careers while limiting same-group repetition."""
    results = results.copy()

    # O*NET-SOC major groups are represented by the first two digits
    # of the occupation code, e.g. 15 = Computer and Mathematical.
    results["Major Group"] = (
        results["O*NET-SOC Code"]
        .astype(str)
        .str.extract(r"^(\d{2})", expand=False)
    )

    selected_indices = []
    group_counts = {}

    # First pass: respect the group limit.
    for index, row in results.iterrows():
        group = row["Major Group"]

        if group is None or pd.isna(group):
            group = "Unknown"

        if group_counts.get(group, 0) >= max_per_group:
            continue

        selected_indices.append(index)
        group_counts[group] = group_counts.get(group, 0) + 1

        if len(selected_indices) >= top_n:
            break

    # Fallback: fill remaining places if the group limit leaves gaps.
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
    """Rank occupations using interest and skill similarity."""
    profiles, model, model_features, zone3_threshold = load_resources()

    interests = {
        name.strip().lower(): value
        for name, value in interests.items()
    }
    skills = skills or {}

    interests = validate_ratings(interests, "Interest")
    skills = validate_ratings(skills, "Skill")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    unknown_interests = set(interests) - set(INTEREST_COLUMNS)
    if unknown_interests:
        raise ValueError(
            f"Unknown interests: {', '.join(sorted(unknown_interests))}"
        )

    interest_scores = calculate_interest_scores(profiles, interests)
    skill_scores = calculate_skill_scores(profiles, skills)

    if skills:
        match_scores = (
            INTEREST_WEIGHT * interest_scores
            + SKILL_WEIGHT * skill_scores
        )
    else:
        match_scores = interest_scores

    # Predict Job Zone independently of the career match score.
    X = profiles[model_features].apply(pd.to_numeric, errors="coerce")
    valid_rows = X.notna().all(axis=1)
    predicted_zones = np.full(len(profiles), np.nan)

    if valid_rows.any():
        X_valid = X.loc[valid_rows]
        probabilities = model.predict_proba(X_valid)
        classes = model.classes_

        predictions = classes[np.argmax(probabilities, axis=1)]

        zone3_positions = np.flatnonzero(classes == 3)

        if len(zone3_positions) > 0 and zone3_threshold <= 1.0:
            zone3_position = zone3_positions[0]
            zone3_probabilities = probabilities[:, zone3_position]
            predictions[zone3_probabilities >= zone3_threshold] = 3

        predicted_zones[valid_rows.to_numpy()] = predictions

    # Build results regardless of whether any valid model rows exist.
    results = profiles[
        ["O*NET-SOC Code", "Title", "Job Zone"]
    ].copy()

    results["Interest Match (%)"] = interest_scores
    results["Skill Match (%)"] = skill_scores
    results["Match Score (%)"] = match_scores
    results["Predicted Job Zone"] = predicted_zones

    results = results.sort_values(
        "Match Score (%)", ascending=False
    ).reset_index(drop=True)

    return select_diverse_careers(
        results,
        top_n=top_n,
        max_per_group=2,
    )



def main():
    print("CAREERCOMPASS - CAREER RECOMMENDATIONS")

    interests = {
        "realistic": 0,
        "investigative": 5,
        "artistic": 0,
        "social": 0,
        "enterprising": 0,
        "conventional": 0,
    }

    skills = {
        "Critical Thinking": 1,
        "Mathematics": 1,
        "Writing": 1,
    }

    recommendations = recommend_careers(
        interests=interests,
        skills=skills,
        top_n=5,
    )

    print("\nTop career matches:")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
