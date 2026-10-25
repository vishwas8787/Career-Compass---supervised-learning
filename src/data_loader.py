
import pandas as pd
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Core datasets for the first inspection
DATASETS = [
    "occupation_data.csv",
    "essential_skills.csv",
    "knowledge.csv",
    "education.csv",
    "career_interest_types.csv",
    "job_zones.csv",
]


def load_dataset(filename):
    """Load one CSV file from the raw data directory."""
    file_path = DATA_DIR / filename

    if not file_path.exists():
        print(f"\n[ERROR] File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except Exception as error:
        print(f"\n[ERROR] Could not load {filename}: {error}")
        return None


def inspect_dataset(filename):
    """Display basic information about a dataset."""
    print("\n" + "=" * 70)
    print(f"DATASET: {filename}")
    print("=" * 70)

    df = load_dataset(filename)

    if df is None:
        return

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print(f"\nTotal missing values: {df.isna().sum().sum()}")


def main():
    print("CAREERCOMPASS - DATA INSPECTION")
    print(f"Data directory: {DATA_DIR}")

    for filename in DATASETS:
        inspect_dataset(filename)

    print("\nInspection complete.")

if __name__ == "__main__":
    main()