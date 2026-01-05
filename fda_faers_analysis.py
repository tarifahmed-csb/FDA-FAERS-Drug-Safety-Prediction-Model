# -*- coding: utf-8 -*-
"""
FDA FAERS Drug Safety Prediction Model
=======================================

Project Goal: Predict adverse events from patient and drug characteristics
              to help drug safety teams prioritize reviews.

Dataset: FDA FAERS Q2 2025

Sections:
    1. Configuration & Imports
    2. Data Loading
    3. Exploratory Data Analysis
    4. Data Cleaning & Preparation
    5. Feature Engineering
    6. Models (Logistic Regression, Decision Tree, Random Forest, KNN)
    7. Results Summary
"""

# =============================================================================
# 1. CONFIGURATION & IMPORTS
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Configuration - Update this path if data is elsewhere
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ASCII')

# Serious outcome codes per FDA FAERS documentation
SERIOUS_CODES = ['DE', 'LT', 'HO', 'DS', 'CA', 'RI']

print("Libraries loaded successfully")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Data path: {DATA_PATH}")


# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_faers_data(data_path):
    """
    Load all 7 FAERS tables from the specified directory.
    
    Table Structure:
        - DEMO: Patient demographics
        - DRUG: Medications involved
        - REAC: Adverse reactions/symptoms
        - OUTC: Patient outcomes (death, hospitalization, etc.)
        - INDI: Drug indications
        - THER: Therapy start/stop dates
        - RPSR: Report sources (who made the report)
    
    Returns:
        dict: Dictionary containing all loaded DataFrames
    """
    tables = {}
    file_mapping = {
        'demo': 'DEMO25Q2.txt',
        'drug': 'DRUG25Q2.txt',
        'reac': 'REAC25Q2.txt',
        'outc': 'OUTC25Q2.txt',
        'indi': 'INDI25Q2.txt',
        'ther': 'THER25Q2.txt',
        'rpsr': 'RPSR25Q2.txt'
    }
    
    for name, filename in file_mapping.items():
        filepath = os.path.join(data_path, filename)
        tables[name] = pd.read_csv(
            filepath,
            delimiter='$',
            encoding='utf-8',
            low_memory=False,
            quotechar='"',
            on_bad_lines='skip'
        )
        print(f"Loaded {name.upper()}: {len(tables[name]):,} rows × {len(tables[name].columns)} columns")
    
    return tables


def clean_column_names(df):
    """Standardize column names: strip spaces, uppercase, remove special chars."""
    df.columns = df.columns.str.strip().str.upper().str.replace(r'[^A-Z0-9_]', '', regex=True)
    return df


# Load data
print("\n" + "=" * 60)
print("Loading FAERS Data Tables")
print("=" * 60)
data = load_faers_data(DATA_PATH)

# Extract individual DataFrames
demo = clean_column_names(data['demo'])
drug = clean_column_names(data['drug'])
reac = clean_column_names(data['reac'])
outc = clean_column_names(data['outc'])
indi = clean_column_names(data['indi'])
ther = clean_column_names(data['ther'])
rpsr = clean_column_names(data['rpsr'])


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Demographics Table Structure
print("\n--- Demographics Table ---")
print(f"Shape: {demo.shape[0]:,} rows × {demo.shape[1]} columns")
print(f"Columns: {demo.columns.tolist()}")

# Missing data summary
print("\n--- Missing Data Summary (DEMO) ---")
missing_count = demo.isnull().sum()
missing_percent = (missing_count / len(demo) * 100)
print(f"{'Column':<20} {'Missing':>15} {'Percent':>10}")
print("-" * 45)
for col in missing_count[missing_count > 0].sort_values(ascending=False).head(10).index:
    print(f"{col:<20} {missing_count[col]:>15,} {missing_percent[col]:>9.2f}%")

# Outcome Distribution
print("\n--- Outcome Distribution ---")
outcome_counts = outc['OUTC_COD'].value_counts()
print(outcome_counts)

# Define serious outcomes
outc['IS_SERIOUS'] = outc['OUTC_COD'].isin(SERIOUS_CODES).astype(int)
serious_count = outc['IS_SERIOUS'].sum()
total_outcomes = len(outc)
print(f"\nSerious outcomes: {serious_count:,} ({serious_count/total_outcomes*100:.1f}%)")
print(f"Non-serious: {total_outcomes - serious_count:,} ({(total_outcomes-serious_count)/total_outcomes*100:.1f}%)")

# Collapse outcomes to report level
outc_report = outc.groupby('PRIMARYID')['IS_SERIOUS'].max().reset_index()
print(f"\nUnique reports with outcomes: {len(outc_report):,}")

# Merge DEMO with outcomes for exploration
demo_outc = demo.merge(outc_report, on='PRIMARYID', how='inner')
serious_rate = demo_outc['IS_SERIOUS'].mean() * 100
print(f"Overall serious outcome rate: {serious_rate:.1f}%")

# Sex differences
print("\n--- Serious Outcome Rate by Sex ---")
demo_with_sex = demo_outc[demo_outc['SEX'].notna()]
sex_analysis = demo_with_sex.groupby('SEX')['IS_SERIOUS'].agg([
    ('Total', 'count'),
    ('Serious', 'sum'),
    ('Rate_%', lambda x: round(x.mean() * 100, 1))
])
print(sex_analysis)

# Age distribution
print("\n--- Serious Outcome Rate by Age Group ---")
demo_with_age = demo_outc[demo_outc['AGE'].notna()].copy()
demo_with_age['AGE'] = pd.to_numeric(demo_with_age['AGE'], errors='coerce')
demo_with_age = demo_with_age[demo_with_age['AGE'].notna()]
demo_with_age['AGE_CATEGORY'] = pd.cut(
    demo_with_age['AGE'],
    bins=[0, 18, 45, 65, 150],
    labels=['0-17 (Child)', '18-44 (Adult)', '45-64 (Middle)', '65+ (Elderly)']
)
age_analysis = demo_with_age.groupby('AGE_CATEGORY', observed=True)['IS_SERIOUS'].agg([
    ('Total', 'count'),
    ('Serious', 'sum'),
    ('Rate_%', lambda x: round(x.mean() * 100, 1))
])
print(age_analysis)

# Top drugs
print("\n--- Top 20 Most Reported Drugs ---")
print(drug['DRUGNAME'].value_counts().head(20))


# =============================================================================
# 4. DATA CLEANING & PREPARATION
# =============================================================================

print("\n" + "=" * 60)
print("DATA CLEANING & PREPARATION")
print("=" * 60)

# Drop rows without PRIMARYID (cannot be linked)
demo = demo.dropna(subset=['PRIMARYID'])
drug = drug.dropna(subset=['PRIMARYID'])
reac = reac.dropna(subset=['PRIMARYID'])
outc = outc.dropna(subset=['PRIMARYID'])
indi = indi.dropna(subset=['PRIMARYID'])
ther = ther.dropna(subset=['PRIMARYID'])
rpsr = rpsr.dropna(subset=['PRIMARYID'])

# Drop rows with missing AGE or SEX from DEMO (critical features)
original_len = len(demo)
demo = demo.dropna(subset=['AGE', 'SEX'])
print(f"Dropped {original_len - len(demo):,} rows from DEMO due to missing AGE/SEX")

# Remove duplicates
demo = demo.drop_duplicates()
drug = drug.drop_duplicates()
reac = reac.drop_duplicates()
outc = outc.drop_duplicates()
indi = indi.drop_duplicates()
ther = ther.drop_duplicates()
rpsr = rpsr.drop_duplicates()

print(f"\nAfter cleaning - DEMO: {len(demo):,} rows")

# Filter to Primary Suspect drugs only
if 'ROLE_COD' in drug.columns:
    drug = drug[drug['ROLE_COD'].astype(str).str.upper() == 'PS']
    print(f"Filtered to {len(drug):,} primary suspect drug records")

# Create serious outcome flag
outc['SERIOUS'] = outc['OUTC_COD'].isin(SERIOUS_CODES).astype(int)
outcome_flags = outc.groupby('PRIMARYID')['SERIOUS'].max().reset_index()

# Merge tables
merged = pd.merge(demo, drug, on='PRIMARYID', how='inner')
merged = pd.merge(merged, outcome_flags, on='PRIMARYID', how='left')
merged = pd.merge(merged, ther[['PRIMARYID', 'START_DT', 'END_DT']], on='PRIMARYID', how='left')
merged['SERIOUS'] = merged['SERIOUS'].fillna(0).astype(int)

# Convert types
merged['AGE'] = pd.to_numeric(merged['AGE'], errors='coerce')
merged['SEX'] = merged['SEX'].replace({'M': 1, 'F': 0, 'UNK': np.nan})

# Handle country (group to top 10)
if 'REPORTER_COUNTRY' in merged.columns:
    top_countries = merged['REPORTER_COUNTRY'].value_counts().nlargest(10).index
    merged['REPORTER_COUNTRY'] = merged['REPORTER_COUNTRY'].where(
        merged['REPORTER_COUNTRY'].isin(top_countries), 'OTHER'
    )

# Drop highly missing columns (>80%)
missing_ratio = merged.isna().mean().sort_values(ascending=False)
print("\nMissing value ratio (top 10 columns):")
print(missing_ratio.head(10))
merged = merged.loc[:, missing_ratio < 0.8]

print(f"\nMerged dataset shape: {merged.shape}")
print(f"Serious outcome rate: {merged['SERIOUS'].mean():.3f}")

# Save cleaned dataset
merged.to_csv(os.path.join(os.path.dirname(DATA_PATH), "FAERS25Q2_CLEANED.csv"), index=False)
print("Saved cleaned dataset to FAERS25Q2_CLEANED.csv")

# Merge with reactions
df = merged.merge(reac[['PRIMARYID', 'PT']], on='PRIMARYID', how='left')
print(f"\nDataset with reactions: {df.shape}")


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Age groups
df['AGE_GROUP'] = pd.cut(
    df['AGE'],
    bins=[0, 18, 40, 65, 120],
    labels=['Child', 'YoungAdult', 'Adult', 'Senior']
)

# Drug count per report
drug_count = df.groupby('PRIMARYID')['DRUGNAME'].count().reset_index()
drug_count.columns = ['PRIMARYID', 'NUM_DRUGS']
df = df.merge(drug_count, on='PRIMARYID', how='left')

# Reaction count per report
reac_count = df.groupby('PRIMARYID')['PT'].count().reset_index()
reac_count.columns = ['PRIMARYID', 'NUM_REACTIONS']
df = df.merge(reac_count, on='PRIMARYID', how='left')

# Encode categorical variables
categorical_cols = ['SEX', 'ROUTE', 'AGE_GROUP']
categorical_cols = [col for col in categorical_cols if col in df.columns]

for col in categorical_cols:
    if df[col].dtype.name == 'category':
        df[col] = df[col].astype(object)
    df[col] = df[col].fillna('UNK').astype(str)

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop unneeded columns for modeling
drop_cols = ['PRIMARYID', 'DRUGNAME', 'PT', 'START_DT', 'END_DT', 'OUTC_COD', 'REPORTER_COUNTRY']
df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])

print(f"Feature-engineered dataset shape: {df_model.shape}")


# =============================================================================
# 6. HELPER FUNCTIONS FOR MODELING
# =============================================================================

def preprocess_features(X):
    """Convert all features to numeric and handle missing values."""
    X_processed = X.copy()
    
    numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Label encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = X_processed[col].astype(str).fillna('MISSING')
        X_processed[col] = le.fit_transform(X_processed[col])
        label_encoders[col] = le
    
    # Impute missing numeric values
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy='median')
        X_processed[numeric_cols] = imputer.fit_transform(X_processed[numeric_cols])
    
    return X_processed, label_encoders


def evaluate_model(name, y_test, y_pred, y_proba, cmap='Blues'):
    """Evaluate and display model performance metrics."""
    print("=" * 60)
    print(f"{name} RESULTS")
    print("=" * 60)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Serious', 'Serious']))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Serious', 'Serious'])
    disp.plot(cmap=cmap, values_format='d')
    plt.title(f'{name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=150)
    plt.show()
    
    return {'accuracy': accuracy, 'balanced_acc': balanced_acc, 'roc_auc': roc_auc, 
            'precision': precision, 'recall': recall, 'f1': f1}


# =============================================================================
# 7. MODELS
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

# Prepare features and target
X = df_model.drop('SERIOUS', axis=1)
y = df_model['SERIOUS']

print(f"Original data: {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"Target distribution:\n{y.value_counts()}")
print(f"Serious rate: {y.mean():.3f}")

# Preprocess features
X_processed, label_encoders = preprocess_features(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store results
results = {}

# -----------------------------------------------------------------------------
# 7.1 LOGISTIC REGRESSION
# -----------------------------------------------------------------------------

print("\n--- Training Logistic Regression ---")
logr = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    C=1.0,
    solver='liblinear'
)
logr.fit(X_train_scaled, y_train)

y_pred_logr = logr.predict(X_test_scaled)
y_proba_logr = logr.predict_proba(X_test_scaled)[:, 1]

results['Logistic Regression'] = evaluate_model(
    'Logistic Regression', y_test, y_pred_logr, y_proba_logr, 'Blues'
)

# Feature importance
print("\nTop 15 Most Important Features (Logistic Regression):")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': logr.coef_[0],
    'abs_importance': np.abs(logr.coef_[0])
}).sort_values('abs_importance', ascending=False)
print(feature_importance.head(15))

# Cross-validation
cv_scores = cross_val_score(logr, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\nCross-Validation ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# -----------------------------------------------------------------------------
# 7.2 DECISION TREE
# -----------------------------------------------------------------------------

print("\n--- Training Decision Tree ---")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
y_proba_dt = dt_model.predict_proba(X_test)[:, 1]

results['Decision Tree'] = evaluate_model(
    'Decision Tree', y_test, y_pred_dt, y_proba_dt, 'Oranges'
)

# -----------------------------------------------------------------------------
# 7.3 RANDOM FOREST
# -----------------------------------------------------------------------------

print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

results['Random Forest'] = evaluate_model(
    'Random Forest', y_test, y_pred_rf, y_proba_rf, 'Greens'
)

# Feature importance
print("\nTop 10 Features (Random Forest):")
rf_importance = pd.Series(rf_model.feature_importances_, index=X_train.columns)
print(rf_importance.nlargest(10))

# -----------------------------------------------------------------------------
# 7.4 K-NEAREST NEIGHBORS
# -----------------------------------------------------------------------------

print("\n--- Training K-Nearest Neighbors ---")
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
y_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

results['KNN'] = evaluate_model(
    'K-Nearest Neighbors', y_test, y_pred_knn, y_proba_knn, 'Purples'
)


# =============================================================================
# 8. RESULTS SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
print(results_df)

# Best model
best_model = results_df['roc_auc'].idxmax()
print(f"\nBest Model (by ROC AUC): {best_model} ({results_df.loc[best_model, 'roc_auc']:.4f})")

# Save results
results_df.to_csv(os.path.join(os.path.dirname(DATA_PATH), "model_comparison_results.csv"))
print("\nSaved model comparison to model_comparison_results.csv")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

"""
KEY FINDINGS:
=============

DEMOGRAPHIC PATTERNS:
- Males have 7.7% higher serious outcome rate than females
- Elderly (65+): 59.5% serious - highest risk group
- Children (0-17): 53.5% serious - surprisingly high
- Young adults (18-44): 47.0% serious - lowest risk

PREDICTIVE FEATURES:
- Number of drugs taken
- Dose amount
- Route of administration
- Patient demographics (age, sex)

OUTCOMES:
- Hospitalization is the most common serious outcome (~27%)
- Death accounts for ~9.5% of serious outcomes

These models can help drug safety teams prioritize review of high-risk reports.
"""
