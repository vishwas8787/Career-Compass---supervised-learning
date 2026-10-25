
import pandas as pd
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_csv(filename):
    """Load a CSV file from the raw data directory."""
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path, low_memory=False)


def create_rating_features(filename, prefix):
    """Convert occupation ratings into separate feature columns."""
    df = load_csv(filename)

    # Use Importance ratings only; do not mix them with Level ratings.
    df = df[df["Scale ID"] == "IM"].copy()

    df["Data Value"] = pd.to_numeric(
        df["Data Value"], errors="coerce"
    )

    # Each occupation becomes one row.
    features = df.pivot_table(
        index="O*NET-SOC Code",
        columns="Element Name",
        values="Data Value",
        aggfunc="mean"
    )

    # Prefix names to prevent collisions between datasets.
    features.columns = [
        f"{prefix}_{column}"
        for column in features.columns
    ]

    return features.reset_index()


def create_interest_features():
    """Create one feature for each career interest type."""
    df = load_csv("career_interest_types.csv")

    df["Data Value"] = pd.to_numeric(
        df["Data Value"], errors="coerce"
    )

    features = df.pivot_table(
        index="O*NET-SOC Code",
        columns="Element Name",
        values="Data Value",
        aggfunc="mean"
    )

    features.columns = [
        f"interest_{column}"
        for column in features.columns
    ]

    return features.reset_index()


def build_occupation_profiles():
    """Merge occupation information and numerical features."""
    occupations = load_csv("occupation_data.csv")
    job_zones = load_csv("job_zones.csv")

    # Start with the complete occupation list.
    profiles = occupations[
        ["O*NET-SOC Code", "Title", "Description"]
    ].copy()

    # Merge skills and knowledge importance features.
    skills = create_rating_features(
        "essential_skills.csv", "skill"
    )
    knowledge = create_rating_features(
        "knowledge.csv", "knowledge"
    )
    interests = create_interest_features()

    profiles = profiles.merge(
        skills, on="O*NET-SOC Code", how="left"
    )
    profiles = profiles.merge(
        knowledge, on="O*NET-SOC Code", how="left"
    )
    profiles = profiles.merge(
        interests, on="O*NET-SOC Code", how="left"
    )

    # Add job preparation zone.
    profiles = profiles.merge(
        job_zones[["O*NET-SOC Code", "Job Zone"]],
        on="O*NET-SOC Code",
        how="left"
    )

    return profiles


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Building occupation profiles...")

    profiles = build_occupation_profiles()

    print("\nTop 15 columns with the most missing values:")
    print(
        profiles.isna().sum()
        .sort_values(ascending=False)
        .head(15)
        .to_string()
    )

    print("\nPercentage of occupations with any missing features:")
    feature_columns = [
        col for col in profiles.columns
        if col.startswith(("skill_", "knowledge_", "interest_"))
    ]

    print(
        f"{profiles[feature_columns].isna().any(axis=1).mean() * 100:.2f}%"
    )

    print("\nOccupations with the most missing features:")
    print(
        profiles.assign(
            missing_features=profiles[feature_columns].isna().sum(axis=1)
        )[["Title", "missing_features"]]
        .sort_values("missing_features", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    output_path = PROCESSED_DIR / "occupation_profiles.csv"
    profiles.to_csv(output_path, index=False)

    feature_columns = [
        column for column in profiles.columns
        if column.startswith(("skill_", "knowledge_", "interest_"))
    ]

    print("\nPreprocessing complete.")
    print(f"Occupations: {len(profiles)}")
    print(f"Total columns: {len(profiles.columns)}")
    print(f"Numerical feature columns: {len(feature_columns)}")
    print(f"Missing values: {profiles.isna().sum().sum()}")
    print(f"Saved to: {output_path}")

    print("\nFirst 5 occupations:")
    print(
        profiles[
            ["O*NET-SOC Code", "Title", "Job Zone"]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()
