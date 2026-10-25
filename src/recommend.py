
import pandas as pd
import joblib

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "job_zone_model.joblib"


# Map user-friendly interest names to dataset columns.
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
            "Run preprocessing.py and feature_engineering.py first."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Run train.py first to create the Job Zone model."
        )

    profiles = pd.read_csv(DATA_FILE)
    saved = joblib.load(MODEL_FILE)

    model = saved["model"]
    model_features = saved["feature_columns"]

    return profiles, model, model_features


def recommend_careers(
    interests,
    skills=None,
    top_n=5,
):
    """
    Rank occupations by similarity to a student's interests and skills.

    interests: dict, e.g. {"investigative": 5, "artistic": 2}
    skills: dict, e.g. {"Critical Thinking": 4, "Mathematics": 3}
    top_n: number of recommendations to return
    """
    profiles, model, model_features = load_resources()

    skills = skills or {}

    if not interests:
        raise ValueError("Provide at least one career interest.")

    # Start with a zero-weighted score for every occupation.
    scores = pd.Series(0.0, index=profiles.index)
    total_weight = 0.0

    # Normalize interest weights so their scale is consistent.
    valid_interests = {}

    for name, value in interests.items():
        key = name.strip().lower()

        if key not in INTEREST_COLUMNS:
            raise ValueError(
                f"Unknown interest '{name}'. Choose from: "
                f"{', '.join(INTEREST_COLUMNS)}"
            )

        value = float(value)

        if not 0 <= value <= 5:
            raise ValueError(
                f"Interest '{name}' must be between 0 and 5."
            )

        valid_interests[INTEREST_COLUMNS[key]] = value

    # Normalize the student's interest ratings.
    interest_total = sum(valid_interests.values())

    if interest_total == 0:
        raise ValueError("At least one interest rating must be above zero.")

    for column, value in valid_interests.items():
        scores += (
            profiles[column].fillna(0) * (value / interest_total)
        )
        total_weight += value / interest_total

    # Match skills against their corresponding O*NET ratings.
    skill_weight = 0.0

    for name, value in skills.items():
        column = f"skill_{name.strip()}"

        if column not in profiles.columns:
            raise ValueError(
                f"Skill '{name}' was not found. "
                "Use a skill name from the O*NET skill columns."
            )

        value = float(value)

        if not 0 <= value <= 5:
            raise ValueError(
                f"Skill '{name}' must be between 0 and 5."
            )

        # Penalize a mismatch between student ability and job importance.
        required = profiles[column].fillna(0)
        gap = (required - value).clip(lower=0)

        # Smaller gaps contribute more to the match score.
        scores += 1 / (1 + gap)
        skill_weight += 1

    if skill_weight:
        # Keep the overall skill contribution comparable to interests.
        scores = scores / (1 + skill_weight)

    # Use the model to estimate Job Zone.
    X = profiles[model_features].apply(
        pd.to_numeric, errors="coerce"
    )

    valid_rows = X.notna().all(axis=1)

    predictions = pd.Series(
        float("nan"), index=profiles.index
    )

    predictions.loc[valid_rows] = model.predict(
        X.loc[valid_rows]
    )

    results = profiles[
        ["O*NET-SOC Code", "Title", "Job Zone"]
    ].copy()

    results["Match Score"] = scores
    results["Predicted Job Zone"] = predictions

    results = results.sort_values(
        "Match Score", ascending=False
    ).head(top_n)

    return results.reset_index(drop=True)


def main():
    print("CAREERCOMPASS - CAREER RECOMMENDATIONS")

    # Example profile for testing; we'll replace this with user input later.
    interests = {
        "investigative": 5,
        "realistic": 3,
        "artistic": 2,
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
