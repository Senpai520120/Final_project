"""Train the loan approval model and save the full pipeline.

Run with:  python -m ml.train_model

Builds a preprocessing + RandomForest pipeline, tunes it with GridSearchCV,
evaluates it on a held-out test set and stores the fitted pipeline in
models/loan_model.joblib together with a text report in ml/model_report.txt.
"""

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import (
    CATEGORICAL_FEATURES,
    DATA_FILE,
    FEATURES,
    MODEL_FILE,
    NUMERIC_FEATURES,
    PLOTS_DIR,
    REPORT_FILE,
    TARGET,
)

RANDOM_STATE = 42

# Small grid on purpose: enough to tune the forest, fast enough to rerun.
PARAM_GRID = {
    "classifier__n_estimators": [100, 300],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 3],
}

sns.set_theme(style="whitegrid")


def load_data():
    """Return the raw feature frame X and the binary target y (1 = approved)."""
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df[TARGET].eq("Y").astype(int)
    return X, y


def build_pipeline():
    """Preprocessing + classifier in a single estimator.

    Imputation and scaling live inside the pipeline, so they are fitted on the
    training folds only and never leak information from the test set.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",  # the dataset is 71% approved / 29% rejected
        )),
    ])


def evaluate(model, X_test, y_test):
    """Return the test-set metrics as a printable report string."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    lines = [
        "TEST SET METRICS",
        f"Accuracy : {accuracy_score(y_test, y_pred):.3f}",
        f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}",
        f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.3f}",
        f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.3f}",
        f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}",
        "",
        "Classification report:",
        classification_report(y_test, y_pred, target_names=["Rejected", "Approved"],
                              zero_division=0),
        "Confusion matrix (rows = actual, columns = predicted):",
        str(confusion_matrix(y_test, y_pred)),
    ]
    return "\n".join(lines), confusion_matrix(y_test, y_pred)


def feature_importances(model):
    """Map RandomForest importances back to readable feature names."""
    names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    return (
        pd.Series(importances, index=names)
        .sort_values(ascending=False)
    )


def plot_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    _save(fig, "confusion_matrix.png")


def plot_feature_importance(importances):
    top = importances.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color="#4c72b0")
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 features used by the Random Forest")
    _save(fig, "feature_importance.png")


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=100)
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"train: {len(X_train)} rows, test: {len(X_test)} rows")

    search = GridSearchCV(
        build_pipeline(),
        PARAM_GRID,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    report, matrix = evaluate(model, X_test, y_test)
    importances = feature_importances(model)

    text = "\n".join([
        "LOAN APPROVAL MODEL REPORT",
        "=" * 60,
        f"Rows: {len(X)} (train {len(X_train)} / test {len(X_test)})",
        "",
        "BEST PARAMETERS (GridSearchCV, 5-fold cross-validation, scoring=f1)",
        *(f"  {key} = {value}" for key, value in search.best_params_.items()),
        f"  best cross-validation F1: {search.best_score_:.3f}",
        "",
        report,
        "",
        "FEATURE IMPORTANCE (top 15)",
        importances.head(15).round(4).to_string(),
    ])
    print("\n" + text)

    REPORT_FILE.write_text(text, encoding="utf-8")
    plot_confusion_matrix(matrix)
    plot_feature_importance(importances)

    joblib.dump(model, MODEL_FILE)
    print(f"\nsaved model:  {MODEL_FILE}")
    print(f"saved report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
