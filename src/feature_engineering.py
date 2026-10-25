
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "occupation_profiles.csv"
OUTPUT_FILE = PROCESSED_DIR / "occupation_features.csv"


def load_profiles():
    """Load the processed occupation profiles."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run preprocessing.py first."
        )

    return pd.read_csv(INPUT_FILE)


def select_features(profiles):
    """Select numerical occupation features."""
    feature_columns = [
        column for column in profiles.columns
        if column.startswith(
            ("skill_", "knowledge_", "interest_")
        )
    ]

    if not feature_columns:
        raise ValueError("No numerical feature columns were found.")

    return feature_columns


def prepare_features(profiles, feature_columns):
    """Keep occupations with complete feature ratings."""
    complete_mask = profiles[feature_columns].notna().all(axis=1)

    usable_profiles = profiles.loc[complete_mask].copy()

    X = usable_profiles[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Guard against unexpected non-numeric values.
    valid_mask = X.notna().all(axis=1)

    X = X.loc[valid_mask].copy()
    usable_profiles = usable_profiles.loc[valid_mask].copy()

    return usable_profiles, X


def main():
    print("CAREERCOMPASS - FEATURE ENGINEERING")

    profiles = load_profiles()
    feature_columns = select_features(profiles)

    usable_profiles, X = prepare_features(
        profiles, feature_columns
    )

    # Save features together with occupation identifiers.
    output = usable_profiles[
        ["O*NET-SOC Code", "Title", "Job Zone"]
    ].reset_index(drop=True)

    output = pd.concat(
        [output, X.reset_index(drop=True)],
        axis=1
    )

    output.to_csv(OUTPUT_FILE, index=False)

    print(f"\nTotal occupations: {len(profiles)}")
    print(f"Selected features: {len(feature_columns)}")
    print(f"Usable occupations: {len(output)}")
    print(f"Excluded occupations: {len(profiles) - len(output)}")
    print(f"Missing values in features: {X.isna().sum().sum()}")
    print(f"\nSaved features to: {OUTPUT_FILE}")

    print("\nSample:")
    print(output.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
