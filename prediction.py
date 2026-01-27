# Predict whether a GitHub PR will be merged (is_merged) using a Random Forest classifier.
#Load data — 31,284 PRs with 73 features
#Clean data — Drops leaky columns (like merged_at which would reveal the answer) and redundant columns (highly correlated features like additions ≈ total_changes)
#Prepare features — Encodes categorical variables (agent, repo_language, etc.) into numbers, converts booleans to 0/1
#Train/test split — 80% train, 20% test, stratified to maintain the 77%/23% merged/rejected ratio
# Train Random Forest — 200 trees, class balancing to handle the imbalanced target
# Evaluate — Reports accuracy (82%), ROC-AUC (0.85), confusion matrix, and feature importances
#Cross-validate — 5-fold CV to confirm results aren't due to lucky split

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_recall_curve, 
                             average_precision_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_parquet('/home/niruthi/ai_code/data/engineered_features.parquet')
print(f"Original shape: {df.shape}")

# =============================================================================
# 2. DROP LEAKY & USELESS COLUMNS
# =============================================================================
leaky_cols = [
    # Direct leakage
    'merged_at',      # Directly reveals target
    'closed_at',      # Timing after decision
    'state',          # Only 1 value anyway
    
    # Identifiers (no predictive value + temporal leakage risk)
    'id', 'number', 'user_id', 'repo_id',
    
    # Text/URLs (not using NLP)
    'title', 'body', 'html_url', 'repo_url',
    
    # Timestamps (features already extracted)
    'created_at', 'user_account_created',
    
    # High cardinality / redundant
    'user',           # 1562 unique, user metrics already captured
    'month',          # Correlated 0.89 with id - temporal leakage
]

df_clean = df.drop(columns=leaky_cols, errors='ignore')
print(f"After dropping leaky cols: {df_clean.shape}")

# =============================================================================
# 3. DROP REDUNDANT FEATURES (high correlation)
# =============================================================================
redundant_cols = [
    'additions',      # corr 0.96 with total_changes
    'deletions',      # corr 0.82 with total_changes  
    'repo_forks',     # corr 0.88 with repo_stars
]

df_clean = df_clean.drop(columns=redundant_cols, errors='ignore')
print(f"After dropping redundant cols: {df_clean.shape}")

# =============================================================================
# 4. SEPARATE FEATURES AND TARGET
# =============================================================================
X = df_clean.drop(columns=['is_merged'])
y = df_clean['is_merged']

print(f"\nFeatures: {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts(normalize=True).round(3)}")

# =============================================================================
# 5. IDENTIFY COLUMN TYPES
# =============================================================================
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()

print(f"\nCategorical ({len(cat_cols)}): {cat_cols}")
print(f"Numerical ({len(num_cols)}): {num_cols}")
print(f"Boolean ({len(bool_cols)}): {bool_cols}")

# =============================================================================
# 6. ENCODE CATEGORICAL FEATURES
# =============================================================================
# Convert booleans to int
for col in bool_cols:
    X[col] = X[col].astype(int)

# Label encode categoricals (for tree-based baseline)
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded {col}: {len(le.classes_)} classes")

print(f"\nFinal feature matrix: {X.shape}")

# =============================================================================
# 7. TRAIN-TEST SPLIT (STRATIFIED)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Train target dist: {y_train.value_counts(normalize=True).round(3).to_dict()}")
print(f"Test target dist:  {y_test.value_counts(normalize=True).round(3).to_dict()}")

# =============================================================================
# 8. BASELINE MODEL (Random Forest with class balancing)
# =============================================================================
print("\n" + "="*70)
print("TRAINING BASELINE MODEL (Random Forest)")
print("="*70)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  # Handle imbalance
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =============================================================================
# 9. EVALUATION
# =============================================================================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred, target_names=['Rejected (0)', 'Merged (1)']))

print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)
cm = confusion_matrix(y_test, y_pred)
print(f"""
                 Predicted
              Rejected  Merged
Actual Rejected   {cm[0,0]:5}   {cm[0,1]:5}
Actual Merged     {cm[1,0]:5}   {cm[1,1]:5}
""")

# Key metrics for minority class
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (Correctly rejected):  {tn}")
print(f"False Positives (Should be rejected): {fp}")
print(f"False Negatives (Should be merged):   {fn}")
print(f"True Positives (Correctly merged):    {tp}")

print("\n" + "="*70)
print("MINORITY CLASS (Rejected) METRICS")
print("="*70)
rejected_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
rejected_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
rejected_f1 = 2 * (rejected_precision * rejected_recall) / (rejected_precision + rejected_recall) if (rejected_precision + rejected_recall) > 0 else 0
print(f"Precision (Rejected): {rejected_precision:.4f}")
print(f"Recall (Rejected):    {rejected_recall:.4f}")
print(f"F1-Score (Rejected):  {rejected_f1:.4f}")

print("\n" + "="*70)
print("OVERALL METRICS")
print("="*70)
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)
print(f"ROC-AUC:     {roc_auc:.4f}")
print(f"PR-AUC:      {pr_auc:.4f}")

# =============================================================================
# 10. FEATURE IMPORTANCE
# =============================================================================
print("\n" + "="*70)
print("TOP 20 FEATURE IMPORTANCES")
print("="*70)
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feat_imp.head(20).iterrows():
    print(f"  {row['feature']:45} | {row['importance']:.4f}")

# =============================================================================
# 11. CROSS-VALIDATION (5-Fold Stratified)
# =============================================================================
print("\n" + "="*70)
print("5-FOLD CROSS-VALIDATION")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_cv_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
y_cv_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

print("\nCV Classification Report:")
print(classification_report(y, y_cv_pred, target_names=['Rejected (0)', 'Merged (1)']))

cv_roc = roc_auc_score(y, y_cv_proba)
cv_pr = average_precision_score(y, y_cv_proba)
print(f"CV ROC-AUC: {cv_roc:.4f}")
print(f"CV PR-AUC:  {cv_pr:.4f}")

# =============================================================================
# 12. SAVE FEATURE LIST
# =============================================================================
print("\n" + "="*70)
print("FINAL FEATURE LIST")
print("="*70)
print(f"features = {X.columns.tolist()}")
