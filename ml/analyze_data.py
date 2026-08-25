"""Exploratory analysis of data/loan_data.csv.

Run with:  python -m ml.analyze_data

Prints a text summary of the dataset and writes the analysis charts into
static/images/.
"""

import matplotlib

matplotlib.use("Agg")  # render to files, no GUI window needed

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.config import (
    CATEGORICAL_FEATURES,
    DATA_FILE,
    ID_COLUMN,
    NUMERIC_FEATURES,
    PLOTS_DIR,
    TARGET,
)

sns.set_theme(style="whitegrid")


def describe_dataset(df):
    """Print the basic facts about the dataset: size, types, gaps, target."""
    print("=" * 60)
    print("1. DATASET SIZE")
    print(f"rows: {df.shape[0]}, columns: {df.shape[1]}")

    print("\n2. DATA TYPES")
    print(df.dtypes.to_string())

    print("\n3. MISSING VALUES")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() or "no missing values")

    print("\n4. DUPLICATES")
    print(f"full duplicate rows: {df.duplicated().sum()}")
    print(f"unique {ID_COLUMN}: {df[ID_COLUMN].nunique()} (identifier, dropped)")

    print("\n5. TARGET DISTRIBUTION")
    counts = df[TARGET].value_counts()
    shares = df[TARGET].value_counts(normalize=True).mul(100).round(1)
    print(pd.DataFrame({"count": counts, "%": shares}).to_string())

    print("\n6. NUMERIC FEATURES")
    print(df[NUMERIC_FEATURES].describe().round(2).to_string())

    print("\n7. CATEGORICAL FEATURES")
    for column in CATEGORICAL_FEATURES:
        values = df[column].value_counts(dropna=False).to_dict()
        print(f"{column}: {values}")


def describe_relationships(df):
    """Print correlations between numeric features and their link to the target."""
    print("\n8. CORRELATION BETWEEN NUMERIC FEATURES")
    print(df[NUMERIC_FEATURES].corr().round(2).to_string())

    approved = df[TARGET].eq("Y")

    print("\n9. FEATURES VS TARGET")
    print("Approval rate per category (%):")
    for column in CATEGORICAL_FEATURES:
        rates = approved.groupby(df[column]).mean().mul(100).round(1).to_dict()
        print(f"  {column}: {rates}")

    print("\nMean numeric value per target class:")
    print(df.groupby(TARGET)[NUMERIC_FEATURES].mean().round(1).to_string())

    print("\nPoint-biserial correlation with approval (Y=1, N=0):")
    print(df[NUMERIC_FEATURES].corrwith(approved.astype(float)).round(3).to_string())


def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette="Set2", legend=False, ax=ax)
    ax.set_title("Loan_Status distribution")
    ax.set_xlabel("Loan approved (Y) / rejected (N)")
    _save(fig, "target_distribution.png")


def plot_numeric_distributions(df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, column in zip(axes.flat, NUMERIC_FEATURES):
        sns.histplot(data=df, x=column, kde=True, color="#4c72b0", ax=ax)
        ax.set_title(column)
    fig.suptitle("Distribution of the numeric features")
    _save(fig, "numeric_distributions.png")


def plot_categorical_vs_target(df):
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, column in zip(axes.flat, CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=column, hue=TARGET, palette="Set2", ax=ax)
        ax.set_title(f"{column} vs {TARGET}")
        ax.tick_params(axis="x", rotation=20)
    for ax in axes.flat[len(CATEGORICAL_FEATURES):]:
        ax.remove()
    fig.suptitle("Categorical features against the target")
    _save(fig, "categorical_vs_target.png")


def plot_numeric_vs_target(df):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, column in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=column, hue=TARGET, palette="Set2",
                    legend=False, ax=ax)
        ax.set_title(column)
    fig.suptitle("Numeric features against the target")
    _save(fig, "numeric_vs_target.png")


def plot_correlation_heatmap(df):
    numeric = df[NUMERIC_FEATURES].copy()
    # Encode the binary columns so they can join the same correlation matrix.
    numeric["Credit_History"] = df["Credit_History"]
    numeric["Loan_Approved"] = df[TARGET].eq("Y").astype(int)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    _save(fig, "correlation_heatmap.png")


def _save(fig, filename):
    fig.tight_layout()
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"saved chart: {path.relative_to(PLOTS_DIR.parent.parent)}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE)

    describe_dataset(df)
    describe_relationships(df)

    print("\n10. CHARTS")
    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_categorical_vs_target(df)
    plot_numeric_vs_target(df)
    plot_correlation_heatmap(df)


if __name__ == "__main__":
    main()
