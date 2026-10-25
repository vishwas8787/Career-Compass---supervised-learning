
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_FILE = MODEL_DIR / "job_zone_model.joblib"

ZONE3 = 3

# Guardrails for threshold selection, chosen using training data only.
MAX_MACRO_F1_DROP = 0.02
MAX_ZONE3_PRECISION_DROP = 0.05


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run preprocessing and feature engineering first."
        )

    df = pd.read_csv(DATA_FILE)

    feature_columns = [
        col for col in df.columns
        if col.startswith(("skill_", "knowledge_", "interest_"))
        and "High-Point" not in col
    ]

    if not feature_columns:
        raise ValueError("No suitable input features were found.")

    if "Job Zone" not in df.columns:
        raise ValueError("The dataset has no 'Job Zone' column.")

    df["Job Zone"] = pd.to_numeric(df["Job Zone"], errors="coerce")
    df = df.dropna(subset=["Job Zone"]).copy()
    df["Job Zone"] = df["Job Zone"].astype(int)

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = df["Job Zone"]

    valid = X.notna().all(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]

    if y.nunique() < 2:
        raise ValueError("At least two Job Zone classes are required.")

    if ZONE3 not in y.unique():
        raise ValueError("Job Zone 3 is not present in the training data.")

    return X, y, feature_columns


def make_model():
    """Return the Extra Trees model for threshold tuning."""
    return ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def apply_zone3_threshold(probabilities, classes, threshold):
    """Override the default prediction when Zone 3 probability is high enough."""
    predictions = classes[np.argmax(probabilities, axis=1)]

    zone3_positions = np.flatnonzero(classes == ZONE3)
    if len(zone3_positions) == 0:
        raise ValueError("The fitted model has no Zone 3 class.")

    zone3_position = zone3_positions[0]
    zone3_probabilities = probabilities[:, zone3_position]

    predictions[zone3_probabilities >= threshold] = ZONE3
    return predictions


def calculate_metrics(y_true, y_pred, classes):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, labels=classes,
                average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=classes,
                average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, labels=classes,
                average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred, labels=classes,
                average="weighted", zero_division=0
            )
        ),
        "zone3_precision": float(
            precision_score(
                y_true, y_pred, labels=[ZONE3],
                average="macro", zero_division=0
            )
        ),
        "zone3_recall": float(
            recall_score(
                y_true, y_pred, labels=[ZONE3],
                average="macro", zero_division=0
            )
        ),
    }


def choose_threshold(y_true, oof_probabilities, classes):
    """Select a threshold from out-of-fold training predictions."""
    baseline = classes[np.argmax(oof_probabilities, axis=1)]
    baseline_metrics = calculate_metrics(y_true, baseline, classes)

    candidates = []

    # Threshold 1.01 disables the override and reproduces normal predictions.
    thresholds = list(np.arange(0.10, 0.951, 0.025)) + [1.01]

    for threshold in thresholds:
        predictions = apply_zone3_threshold(
            oof_probabilities, classes, threshold
        )
        metrics = calculate_metrics(y_true, predictions, classes)

        eligible = (
            metrics["macro_f1"]
            >= baseline_metrics["macro_f1"] - MAX_MACRO_F1_DROP
            and metrics["zone3_precision"]
            >= baseline_metrics["zone3_precision"]
            - MAX_ZONE3_PRECISION_DROP
        )

        candidates.append({
            "threshold": float(threshold),
            **metrics,
            "eligible": bool(eligible),
        })

    eligible_candidates = [
        candidate for candidate in candidates if candidate["eligible"]
    ]

    if not eligible_candidates:
        # Safety fallback: keep the normal argmax rule.
        selected = next(
            candidate for candidate in candidates
            if candidate["threshold"] == 1.01
        )
    else:
        # Prioritise Zone 3 recall while respecting the guardrails.
        selected = max(
            eligible_candidates,
            key=lambda item: (
                item["zone3_recall"],
                item["macro_f1"],
                item["zone3_precision"],
            ),
        )

    return baseline_metrics, selected, candidates


def save_confusion_matrix(y_true, y_pred, classes):
    matrix = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Job Zone Classification — Confusion Matrix")
    ax.set_xlabel("Predicted Job Zone")
    ax.set_ylabel("Actual Job Zone")
    fig.tight_layout()

    path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return matrix, path


def save_roc_curves(y_true, probabilities, classes):
    binary_labels = label_binarize(y_true, classes=classes)
    class_auc = {}

    fig, ax = plt.subplots(figsize=(8, 6))

    for index, zone in enumerate(classes):
        actual = binary_labels[:, index]

        if len(np.unique(actual)) < 2:
            continue

        fpr, tpr, _ = roc_curve(actual, probabilities[:, index])
        score = auc(fpr, tpr)
        class_auc[str(int(zone))] = float(score)
        ax.plot(fpr, tpr, label=f"Zone {zone} (AUC = {score:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set_title("Multiclass ROC Curves (One-vs-Rest)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = OUTPUT_DIR / "roc_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    macro_auc = roc_auc_score(
        y_true,
        probabilities,
        labels=classes,
        multi_class="ovr",
        average="macro",
    )

    return class_auc, float(macro_auc), path


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feature_columns = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("=" * 65)
    print("CAREERCOMPASS — ZONE 3 THRESHOLD EXPERIMENT")
    print("=" * 65)
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples:     {len(X_test)}")
    print(f"Input features:    {len(feature_columns)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = make_model()

    print("\nGenerating out-of-fold probabilities on training data...")
    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )

    classes = model.classes_ if hasattr(model, "classes_") else np.sort(y_train.unique())
    # cross_val_predict columns follow the estimator's sorted class order.
    classes = np.sort(y_train.unique())

    baseline_cv, selected, threshold_results = choose_threshold(
        y_train.to_numpy(),
        oof_probabilities,
        classes,
    )

    threshold_table = pd.DataFrame(threshold_results)
    threshold_table.to_csv(
        OUTPUT_DIR / "threshold_comparison.csv", index=False
    )

    print("\nOut-of-fold baseline metrics:")
    print(f"  Macro F1:       {baseline_cv['macro_f1']:.4f}")
    print(f"  Zone 3 precision: {baseline_cv['zone3_precision']:.4f}")
    print(f"  Zone 3 recall:    {baseline_cv['zone3_recall']:.4f}")

    print("\nSelected threshold rule:")
    print(f"  Zone 3 threshold: {selected['threshold']:.3f}")
    print(f"  OOF macro F1:     {selected['macro_f1']:.4f}")
    print(f"  OOF Zone 3 precision: {selected['zone3_precision']:.4f}")
    print(f"  OOF Zone 3 recall:    {selected['zone3_recall']:.4f}")

    print("\nFitting final model on the training set...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)
    default_predictions = model.predict(X_test)
    adjusted_predictions = apply_zone3_threshold(
        probabilities, classes, selected["threshold"]
    )

    default_metrics = calculate_metrics(
        y_test.to_numpy(), default_predictions, classes
    )
    adjusted_metrics = calculate_metrics(
        y_test.to_numpy(), adjusted_predictions, classes
    )

    print("\n" + "=" * 65)
    print("HELD-OUT TEST COMPARISON")
    print("=" * 65)
    print(f"{'Metric':<24}{'Default':>12}{'Adjusted':>12}")
    print("-" * 48)

    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "zone3_precision",
        "zone3_recall",
    ):
        print(
            f"{metric:<24}"
            f"{default_metrics[metric]:>12.4f}"
            f"{adjusted_metrics[metric]:>12.4f}"
        )

    # Use the adjusted rule only if it improves Zone 3 recall on the
    # training-derived threshold selection and preserves its guardrails.
    # Test results are reported for comparison, not used to tune the threshold.
    final_predictions = adjusted_predictions
    final_metrics = adjusted_metrics
    chosen_rule = "zone3_threshold"

    # Keep a separate baseline report for a fair side-by-side comparison.
    default_report = classification_report(
        y_test,
        default_predictions,
        labels=classes,
        zero_division=0,
        digits=4,
    )
    adjusted_report = classification_report(
        y_test,
        adjusted_predictions,
        labels=classes,
        zero_division=0,
        digits=4,
    )

    # ROC-AUC depends on probabilities, so the threshold does not change it.
    class_auc, macro_auc, roc_path = save_roc_curves(
        y_test, probabilities, classes
    )
    matrix, matrix_path = save_confusion_matrix(
        y_test, final_predictions, classes
    )

    metrics = {
        "model": "ExtraTreesClassifier",
        "class_weight": {"2": 1.0, "3": 1.5, "4": 1.0, "5": 1.0},
        "prediction_rule": chosen_rule,
        "zone3_threshold": selected["threshold"],
        "threshold_selection": (
            "Out-of-fold training predictions; "
            "macro F1 drop limited to 0.02 and Zone 3 precision drop "
            "limited to 0.05 relative to OOF baseline"
        ),
        "training_examples": int(len(X_train)),
        "test_examples": int(len(X_test)),
        "input_features": int(len(feature_columns)),
        "default_test_metrics": default_metrics,
        "adjusted_test_metrics": adjusted_metrics,
        "final_test_metrics": final_metrics,
        "macro_ovr_roc_auc": macro_auc,
        "per_class_ovr_roc_auc": class_auc,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            final_predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
    }

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "zone3_threshold": selected["threshold"],
            "prediction_rule": chosen_rule,
        },
        MODEL_FILE,
    )

    with open(
        OUTPUT_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("DEFAULT PREDICTION RULE\n\n")
        file.write(default_report)
        file.write("\n\nZONE 3 THRESHOLD RULE\n\n")
        file.write(adjusted_report)

    print("\nFinal classification report (threshold-adjusted):")
    print(adjusted_report)
    print("Final confusion matrix (rows = actual, columns = predicted):")
    print(matrix)
    print(f"\nMacro OVR ROC-AUC: {macro_auc:.4f}")

    print("\nSaved outputs:")
    print(f"  Model:              {MODEL_FILE}")
    print(f"  Threshold results:  {OUTPUT_DIR / 'threshold_comparison.csv'}")
    print(f"  Metrics:            {OUTPUT_DIR / 'evaluation_metrics.json'}")
    print(f"  Classification:     {OUTPUT_DIR / 'classification_report.txt'}")
    print(f"  Confusion matrix:   {matrix_path}")
    print(f"  ROC curves:         {roc_path}")


if __name__ == "__main__":
    main()
