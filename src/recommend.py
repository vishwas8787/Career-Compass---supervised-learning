
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
INTEREST_WEIGHT = 0.60
SKILL_WEIGHT = 0.40

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

    return profiles, saved["model"], saved["feature_columns"]


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
    """Calculate cosine similarity between student and occupation interests."""
    student_vector = np.zeros(len(INTEREST_COLUMNS), dtype=float)
    occupation_vectors = []

    for index, name in enumerate(INTEREST_COLUMNS):
        column = INTEREST_COLUMNS[name]

        if column not in profiles.columns:
            raise ValueError(f"Missing interest feature: {column}")

        student_vector[index] = interests.get(name, 0)

        occupation_vectors.append(
            pd.to_numeric(profiles[column], errors="coerce").to_numpy()
        )

    occupation_matrix = np.column_stack(occupation_vectors)

    if np.linalg.norm(student_vector) == 0:
        raise ValueError(
            "At least one interest rating must be above zero."
        )

    student_norm = np.linalg.norm(student_vector)
    occupation_norms = np.linalg.norm(occupation_matrix, axis=1)

    similarities = np.zeros(len(profiles), dtype=float)

    valid = (
        np.isfinite(occupation_matrix).all(axis=1)
        & (occupation_norms > 0)
    )

    similarities[valid] = (
        occupation_matrix[valid] @ student_vector
    ) / (occupation_norms[valid] * student_norm)

    # Convert cosine similarity into a percentage-like score.
    return np.clip(similarities, 0, 1) * 100


def calculate_skill_scores(profiles, skills):
    """
    Estimate skill fit using only skills the student actually rated.

    O*NET importance is not a direct proficiency requirement, so this
    is a heuristic rather than a calibrated measure of skill readiness.
    """
    if not skills:
        return np.full(len(profiles), 50.0)

    skill_columns = {}

    for name, value in skills.items():
        column = f"skill_{name.strip()}"

        if column not in profiles.columns:
            raise ValueError(
                f"Unknown skill '{name}'. Use an exact O*NET skill name."
            )

        skill_columns[column] = value

    gaps = []

    for column, student_level in skill_columns.items():
        occupation_importance = pd.to_numeric(
            profiles[column], errors="coerce"
        ).to_numpy()

        # Both values are compared on an approximate 0-5 scale.
        gap = np.abs(occupation_importance - student_level) / 5
        gaps.append(np.clip(gap, 0, 1))

    gap_matrix = np.column_stack(gaps)

    # Do not interpret missing occupation ratings as zero.
    valid_counts = np.isfinite(gap_matrix).sum(axis=1)

    mean_gaps = np.divide(
        np.nansum(gap_matrix, axis=1),
        valid_counts,
        out=np.ones(len(profiles), dtype=float),
        where=valid_counts > 0,
    )

    return (1 - mean_gaps) * 100


def recommend_careers(interests, skills=None, top_n=5):
    """Rank occupations using interest and skill similarity."""
    profiles, model, model_features = load_resources()

    interests = {
        name.strip().lower(): value
        for name, value in interests.items()
    }

    skills = skills or {}

    interests = validate_ratings(interests, "Interest")
    skills = validate_ratings(skills, "Skill")

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    unknown_interests = set(interests) - set(INTEREST_COLUMNS)

    if unknown_interests:
        raise ValueError(
            f"Unknown interests: {', '.join(sorted(unknown_interests))}"
        )

    interest_scores = calculate_interest_scores(profiles, interests)
    skill_scores = calculate_skill_scores(profiles, skills)

    # Combine normalized components using explicit weights.
    if skills:
        match_scores = (
            INTEREST_WEIGHT * interest_scores
            + SKILL_WEIGHT * skill_scores
        )
    else:
        match_scores = interest_scores

    # Predict Job Zone independently of the career match score.
    X = profiles[model_features].apply(
        pd.to_numeric, errors="coerce"
    )

    valid_rows = X.notna().all(axis=1)

    predicted_zones = np.full(len(profiles), np.nan)

    if valid_rows.any():
        predicted_zones[valid_rows.to_numpy()] = model.predict(
            X.loc[valid_rows]
        )

    results = profiles[
        ["O*NET-SOC Code", "Title", "Job Zone"]
    ].copy()

    results["Interest Match (%)"] = interest_scores
    results["Skill Match (%)"] = skill_scores
    results["Match Score (%)"] = match_scores
    results["Predicted Job Zone"] = predicted_zones

    return (
        results.sort_values("Match Score (%)", ascending=False)
        .head(min(top_n, len(results)))
        .reset_index(drop=True)
    )


def main():
    print("CAREERCOMPASS - CAREER RECOMMENDATIONS")

    interests = {
        "investigative": 5,
        "realistic": 2,
        "artistic": 3,
        "social": 1,
        "enterprising": 2,
        "conventional": 3,
    }

    skills = {
        "Critical Thinking": 4,
        "Mathematics": 3,
        "Writing": 3,
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
