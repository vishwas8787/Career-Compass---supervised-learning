import json
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import label_binarize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_FILE = MODEL_DIR / "job_zone_model.joblib"


def load_dataset():
    """Load features and target labels from processed data."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. Run preprocessing first."
        )

    df = pd.read_csv(DATA_FILE)

    feature_columns = [
        col for col in df.columns
        if col.startswith(("skill_", "knowledge_", "interest_"))
        and "High-Point" not in col
    ]

    if not feature_columns:
        raise ValueError("No input features found.")

    df["Job Zone"] = pd.to_numeric(df["Job Zone"], errors="coerce")
    df = df.dropna(subset=["Job Zone"]).copy()
    df["Job Zone"] = df["Job Zone"].astype(int)

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = df["Job Zone"]

    valid = X.notna().all(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]

    return X, y, feature_columns


def train_random_forest(X_train, y_train):
    """Train a standard Scikit-Learn Random Forest Classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, classes):
    """Compute standard classification metrics."""
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_precision": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
    }

    report = classification_report(
        y_test, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    text_report = classification_report(
        y_test, y_pred, labels=classes, zero_division=0, digits=4
    )

    return metrics, report, text_report, y_pred, probs


def save_plots(y_test, y_pred, probs, classes):
    """Save confusion matrix and multiclass ROC curves."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Random Forest — Job Zone Confusion Matrix")
    ax.set_xlabel("Predicted Job Zone")
    ax.set_ylabel("Actual Job Zone")
    fig.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. ROC Curves
    y_test_bin = label_binarize(y_test, classes=classes)
    fig, ax = plt.subplots(figsize=(7, 5))
    per_class_auc = {}

    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], probs[:, idx])
        score = auc(fpr, tpr)
        per_class_auc[str(int(cls))] = float(score)
        ax.plot(fpr, tpr, label=f"Zone {cls} (AUC = {score:.3f})")

    macro_auc = float(
        roc_auc_score(
            y_test, probs, labels=classes, multi_class="ovr", average="macro"
        )
    )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
    ax.set_title(f"Random Forest Multiclass ROC (Macro AUC = {macro_auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    roc_path = OUTPUT_DIR / "roc_curves.png"
    fig.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return cm, macro_auc, per_class_auc, cm_path, roc_path


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("CAREERCOMPASS — RANDOM FOREST MODEL TRAINING (CSE395)")
    print("=" * 65)

    X, y, feature_columns = load_dataset()
    classes = np.sort(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Total Occupations:  {len(X)}")
    print(f"Training Samples:   {len(X_train)}")
    print(f"Testing Samples:    {len(X_test)}")
    print(f"Input Features:     {len(feature_columns)}")
    print(f"Target Classes:     Job Zones {list(classes)}")

    print("\nTraining Random Forest Classifier (Scikit-Learn)...")
    model = train_random_forest(X_train, y_train)

    # 5-Fold Cross Validation Score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\nEvaluating on Held-Out Test Set...")
    metrics, report_dict, report_text, y_pred, probs = evaluate_model(
        model, X_test, y_test, classes
    )

    cm, macro_auc, per_class_auc, cm_path, roc_path = save_plots(
        y_test, y_pred, probs, classes
    )

    print("\nTest Set Classification Report:")
    print(report_text)
    print(f"Macro OVR ROC-AUC:  {macro_auc:.4f}")
    print(f"Test Accuracy:      {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:     {metrics['macro_f1']:.4f}")

    # Save artifacts
    joblib.dump(
        {
            "model": model,
            "model_type": "RandomForestClassifier",
            "feature_columns": feature_columns,
            "classes": classes.tolist(),
        },
        MODEL_FILE,
    )

    results = {
        "model": "RandomForestClassifier",
        "library": "scikit-learn",
        "training_examples": int(len(X_train)),
        "test_examples": int(len(X_test)),
        "input_features": int(len(feature_columns)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "test_metrics": metrics,
        "macro_ovr_roc_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    with open(OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("RANDOM FOREST CLASSIFICATION REPORT (SCIKIT-LEARN)\n\n")
        f.write(report_text)

    print(f"\nModel saved to: {MODEL_FILE}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()
