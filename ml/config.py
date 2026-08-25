"""Shared dataset configuration for the analysis and training scripts.

Every constant here was derived from the real ``data/loan_data.csv`` file
(see ``ml/analyze_data.py`` for the exploration that produced them).
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "loan_data.csv"
MODEL_FILE = BASE_DIR / "models" / "loan_model.joblib"
PLOTS_DIR = BASE_DIR / "static" / "images"
REPORT_FILE = BASE_DIR / "ml" / "model_report.txt"

# Name of the column the model has to predict ("Y" = approved, "N" = rejected).
TARGET = "Loan_Status"

# Pure identifier: 381 unique values for 381 rows, so it carries no signal.
ID_COLUMN = "Loan_ID"

# Continuous features -> median imputation + StandardScaler.
NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]

# Discrete features -> most-frequent imputation + OneHotEncoder.
# Credit_History is stored as 0.0/1.0 but is a binary flag, not a quantity.
CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
