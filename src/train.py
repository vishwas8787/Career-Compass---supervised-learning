
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "job_zone_model.joblib"


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Run preprocessing.py and feature_engineering.py first."
        )

    df = pd.read_csv(DATA_FILE)

    # Exclude identifiers, titles and the target from input features.
    feature_columns = [
        col for col in df.columns
        if col.startswith(("skill_", "knowledge_", "interest_"))
        and "High-Point" not in col
    ]

    if not feature_columns:
        raise ValueError("No suitable feature columns found.")

    # Job Zone is our supervised learning target.
    df["Job Zone"] = pd.to_numeric(
        df["Job Zone"], errors="coerce"
    )

    df = df.dropna(subset=["Job Zone"]).copy()

    X = df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    y = df["Job Zone"].astype(int)

    # Remove any rows with unexpected missing feature values.
    valid_rows = X.notna().all(axis=1)
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    return X, y, feature_columns


def main():
    print("CAREERCOMPASS - MODEL TRAINING")

    X, y, feature_columns = load_data()

    print(f"\nTraining examples: {len(X)}")
    print(f"Input features: {len(feature_columns)}")

    print("\nJob Zone distribution:")
    print(y.value_counts().sort_index().to_string())

    # Stratification preserves class proportions in both sets.
    if y.value_counts().min() < 2:
        raise ValueError(
            "At least two examples per Job Zone are needed "
            "for a stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nEvaluation results")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )
    print(
        "Macro F1: "
        f"{f1_score(y_test, predictions, average='macro'):.4f}"
    )

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test, predictions, labels=sorted(y.unique())
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
        },
        MODEL_FILE,
    )

    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
