"""
Train and Save Model
====================
Run this ONCE to train the model and save artifacts.
After this, users only need the saved .joblib files to run the app.

Usage:
    python train_and_save.py

Outputs:
    - model.joblib     (trained XGBoost model)
    - imputer.joblib   (fitted imputer for handling missing values)
    - config.joblib    (feature names and median values)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

print("=" * 60)
print("LOAN DEFAULT MODEL - TRAINING")
print("=" * 60)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = "lending_club_sample_scored.parquet"

FEATURES = [
    "loan_amnt", "term", "int_rate", "grade",
    "emp_length", "annual_inc", "dti",
    "delinq_2yrs", "open_acc", "revol_util",
    "finbert_pos", "finbert_neg", "finbert_neu"
]

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/5] Loading data...")
if not Path(DATA_FILE).exists():
    print(f"ERROR: {DATA_FILE} not found!")
    exit(1)

df = pd.read_parquet(DATA_FILE)
print(f"      Loaded {len(df):,} rows, {len(df.columns)} columns")

# ============================================================
# CLEAN DATA
# ============================================================
print("\n[2/5] Cleaning data...")

# Clean term
def clean_term(val):
    if pd.isna(val): return np.nan
    digits = ''.join(c for c in str(val) if c.isdigit())
    return float(digits) if digits else np.nan

df["term"] = df["term"].apply(clean_term)

# Clean grade
def clean_grade(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().upper()
    return float(GRADE_MAP.get(s, np.nan))

df["grade"] = df["grade"].apply(clean_grade)

# Clean percentages
def clean_pct(val):
    if pd.isna(val): return np.nan
    try:
        return float(str(val).replace("%", "").strip())
    except:
        return np.nan

df["int_rate"] = df["int_rate"].apply(clean_pct)
df["revol_util"] = df["revol_util"].apply(clean_pct)

# Clean employment length
def clean_emp(val):
    if pd.isna(val): return np.nan
    s = str(val).lower().strip()
    if "< 1" in s: return 0.0
    if "10+" in s: return 10.0
    digits = ''.join(c for c in s if c.isdigit())
    return float(digits) if digits else np.nan

df["emp_length"] = df["emp_length"].apply(clean_emp)

# Ensure numeric
for col in ["loan_amnt", "annual_inc", "dti", "delinq_2yrs", "open_acc", 
            "finbert_pos", "finbert_neg", "finbert_neu"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("      Done")

# ============================================================
# PREPARE FEATURES
# ============================================================
print("\n[3/5] Preparing features...")

available = [f for f in FEATURES if f in df.columns]
print(f"      Using {len(available)} features: {available}")

X = df[available].copy()
y = df["label"].copy()

# Remove rows with missing labels
mask = y.notna()
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].astype(int).values

# Calculate medians
medians = {col: float(X[col].median()) if pd.notna(X[col].median()) else 0.0 
           for col in available}

print(f"      Samples: {len(X):,}")
print(f"      Default rate: {y.mean():.1%}")

# ============================================================
# TRAIN MODEL
# ============================================================
print("\n[4/5] Training model...")

# Impute
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42, stratify=y
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# Class weight
spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric="auc",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

# Evaluate
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"      Test AUC: {auc:.4f}")

# ============================================================
# SAVE ARTIFACTS
# ============================================================
print("\n[5/5] Saving model artifacts...")

joblib.dump(model, "model.joblib")
joblib.dump(imputer, "imputer.joblib")
joblib.dump({
    "features": available,
    "medians": medians,
    "auc": auc
}, "config.joblib")

print("      Saved: model.joblib")
print("      Saved: imputer.joblib")
print("      Saved: config.joblib")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nYou can now run the app with:")
print("    streamlit run loan_app.py")
print("\nUsers only need these files:")
print("    - loan_app.py")
print("    - model.joblib")
print("    - imputer.joblib")
print("    - config.joblib")
