"""
STRATEGIES WITH FIXED 0.5 THRESHOLD
====================================
Since threshold must stay at 0.5, we modify what happens BEFORE the threshold:
1. Probability Calibration (Platt/Isotonic) - makes 0.5 meaningful
2. Class Weight Tuning - adjust model's internal bias
3. Ensemble Weight Optimization - find best model blend
4. Cost-Sensitive Learning - penalize FN vs FP differently
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, 
                             recall_score, accuracy_score, brier_score_loss)
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

THRESHOLD = 0.5  # FIXED as per professor's requirement

# =============================================================================
# 1. LOAD DATA (same as before)
# =============================================================================
print("="*70)
print("STRATEGIES WITH FIXED 0.5 THRESHOLD")
print("="*70)

df = pd.read_parquet('/home/niruthi/ai_code/data/engineered_features_v2.parquet')

leaky_cols = ['merged_at', 'closed_at', 'state', 'id', 'number', 'user_id', 'repo_id',
              'title', 'body', 'html_url', 'repo_url', 'created_at', 'user_account_created',
              'user', 'month', 'ci_passed', 'review_count']
redundant_cols = ['additions', 'deletions', 'repo_forks']
df_clean = df.drop(columns=leaky_cols + redundant_cols, errors='ignore')

X = df_clean.drop(columns=['is_merged'])
y = df_clean['is_merged']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df_clean['agent']
)
agents_test = X_test['agent'].values

# Preprocessor
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
num_cols = [c for c in X.select_dtypes(include=['number']).columns if c not in bool_cols]

preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', 'passthrough', num_cols),
    ('bool', 'passthrough', bool_cols)
], verbose_feature_names_out=False)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Merge rate: {y_train.mean():.3f}")

# =============================================================================
# 2. EVALUATION FUNCTION
# =============================================================================
def evaluate(y_true, y_proba, name=""):
    """Evaluate with fixed 0.5 threshold"""
    y_pred = (y_proba >= THRESHOLD).astype(int)
    return {
        'Strategy': name,
        'Threshold': THRESHOLD,
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'Brier Score': brier_score_loss(y_true, y_proba),  # calibration quality
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision_Merged': precision_score(y_true, y_pred, pos_label=1),
        'Recall_Merged': recall_score(y_true, y_pred, pos_label=1),
        'F1_Merged': f1_score(y_true, y_pred, pos_label=1),
        'Precision_Rejected': precision_score(y_true, y_pred, pos_label=0),
        'Recall_Rejected': recall_score(y_true, y_pred, pos_label=0),
        'F1_Rejected': f1_score(y_true, y_pred, pos_label=0),
    }

results = []

# =============================================================================
# 3. STRATEGY 1: BASELINE MODELS (no modification)
# =============================================================================
print("\n" + "="*70)
print("STRATEGY 1: BASELINE MODELS")
print("="*70)

# RF with balanced weights
rf_balanced = RandomForestClassifier(
    n_estimators=400, max_depth=20, min_samples_split=3, min_samples_leaf=4,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
)
rf_balanced.fit(X_train_prep, y_train)
rf_proba = rf_balanced.predict_proba(X_test_prep)[:, 1]
results.append(evaluate(y_test, rf_proba, "RF (balanced weights)"))
print("✓ RF baseline done")

# XGB low scale_pos_weight
xgb_low = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=8, min_child_weight=5,
    scale_pos_weight=1.5, random_state=42, n_jobs=-1
)
xgb_low.fit(X_train_prep, y_train)
xgb_low_proba = xgb_low.predict_proba(X_test_prep)[:, 1]
results.append(evaluate(y_test, xgb_low_proba, "XGB (spw=1.5)"))
print("✓ XGB baseline done")

# =============================================================================
# 4. STRATEGY 2: PROBABILITY CALIBRATION
# =============================================================================
print("\n" + "="*70)
print("STRATEGY 2: PROBABILITY CALIBRATION")
print("="*70)
print("Makes probabilities meaningful so 0.5 = true 50% chance")

# Platt Scaling (sigmoid)
rf_platt = CalibratedClassifierCV(rf_balanced, method='sigmoid', cv=5)
rf_platt.fit(X_train_prep, y_train)
rf_platt_proba = rf_platt.predict_proba(X_test_prep)[:, 1]
results.append(evaluate(y_test, rf_platt_proba, "RF + Platt Scaling"))
print("✓ RF + Platt done")

# Isotonic Regression
rf_isotonic = CalibratedClassifierCV(rf_balanced, method='isotonic', cv=5)
rf_isotonic.fit(X_train_prep, y_train)
rf_isotonic_proba = rf_isotonic.predict_proba(X_test_prep)[:, 1]
results.append(evaluate(y_test, rf_isotonic_proba, "RF + Isotonic"))
print("✓ RF + Isotonic done")

# XGB + Calibration
xgb_platt = CalibratedClassifierCV(xgb_low, method='sigmoid', cv=5)
xgb_platt.fit(X_train_prep, y_train)
xgb_platt_proba = xgb_platt.predict_proba(X_test_prep)[:, 1]
results.append(evaluate(y_test, xgb_platt_proba, "XGB + Platt Scaling"))
print("✓ XGB + Platt done")

# =============================================================================
# 5. STRATEGY 3: CLASS WEIGHT TUNING
# =============================================================================
print("\n" + "="*70)
print("STRATEGY 3: CLASS WEIGHT TUNING")
print("="*70)
print("Adjust model bias so 0.5 threshold catches more rejections")

# Higher weight on rejected class (0)
for reject_weight in [1.5, 2.0, 2.5, 3.0]:
    rf_weighted = RandomForestClassifier(
        n_estimators=400, max_depth=20, min_samples_split=3, min_samples_leaf=4,
        max_features='sqrt', 
        class_weight={0: reject_weight, 1: 1.0},  # Weight rejected class higher
        random_state=42, n_jobs=-1
    )
    rf_weighted.fit(X_train_prep, y_train)
    proba = rf_weighted.predict_proba(X_test_prep)[:, 1]
    results.append(evaluate(y_test, proba, f"RF (reject_weight={reject_weight})"))
    print(f"✓ RF reject_weight={reject_weight} done")

# XGB with different scale_pos_weight (lower = more conservative on merges)
for spw in [0.5, 0.8, 1.0, 1.2]:
    xgb_tuned = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=8, min_child_weight=5,
        scale_pos_weight=spw, random_state=42, n_jobs=-1
    )
    xgb_tuned.fit(X_train_prep, y_train)
    proba = xgb_tuned.predict_proba(X_test_prep)[:, 1]
    results.append(evaluate(y_test, proba, f"XGB (spw={spw})"))
    print(f"✓ XGB spw={spw} done")

# =============================================================================
# 6. STRATEGY 4: OPTIMIZED ENSEMBLE WEIGHTS
# =============================================================================
print("\n" + "="*70)
print("STRATEGY 4: OPTIMIZED ENSEMBLE WEIGHTS")
print("="*70)

# Get calibrated probabilities for ensembling
rf_cal_proba = rf_platt_proba
xgb_cal_proba = xgb_platt_proba

def ensemble_f1_rejected(weights, probas, y_true):
    """Objective: maximize F1_Rejected with 0.5 threshold"""
    w1, w2 = weights
    w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)  # normalize
    combined = w1 * probas[0] + w2 * probas[1]
    y_pred = (combined >= 0.5).astype(int)
    return -f1_score(y_true, y_pred, pos_label=0)

# Use validation set for optimization
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_prep, y_train, test_size=0.2, random_state=42
)

# Refit on training portion
rf_opt = RandomForestClassifier(
    n_estimators=400, max_depth=20, min_samples_split=3, min_samples_leaf=4,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
)
rf_opt.fit(X_tr, y_tr)
rf_val_proba = rf_opt.predict_proba(X_val)[:, 1]

xgb_opt = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=8, scale_pos_weight=1.5,
    random_state=42, n_jobs=-1
)
xgb_opt.fit(X_tr, y_tr)
xgb_val_proba = xgb_opt.predict_proba(X_val)[:, 1]

# Optimize weights
result_opt = minimize(
    ensemble_f1_rejected,
    x0=[0.5, 0.5],
    args=([rf_val_proba, xgb_val_proba], y_val),
    bounds=[(0.1, 0.9), (0.1, 0.9)],
    method='L-BFGS-B'
)
opt_w1, opt_w2 = result_opt.x
opt_w1, opt_w2 = opt_w1 / (opt_w1 + opt_w2), opt_w2 / (opt_w1 + opt_w2)
print(f"Optimal weights: RF={opt_w1:.3f}, XGB={opt_w2:.3f}")

# Apply to test set
ens_opt_proba = opt_w1 * rf_proba + opt_w2 * xgb_low_proba
results.append(evaluate(y_test, ens_opt_proba, f"Ensemble (RF={opt_w1:.2f}, XGB={opt_w2:.2f})"))

# Also try calibrated ensemble
ens_cal_proba = opt_w1 * rf_cal_proba + opt_w2 * xgb_cal_proba
results.append(evaluate(y_test, ens_cal_proba, f"Calibrated Ensemble (opt weights)"))

# Standard ensembles
for rf_w in [0.6, 0.7, 0.8]:
    xgb_w = 1 - rf_w
    ens_proba = rf_w * rf_cal_proba + xgb_w * xgb_cal_proba
    results.append(evaluate(y_test, ens_proba, f"Cal. Ensemble ({int(rf_w*100)}/{int(xgb_w*100)})"))

print("✓ Ensemble optimization done")

# =============================================================================
# 7. STRATEGY 5: COST-SENSITIVE LEARNING (Custom)
# =============================================================================
print("\n" + "="*70)
print("STRATEGY 5: ADJUSTED SAMPLE WEIGHTS")
print("="*70)

# Give higher weight to rejected samples during training
reject_mask = (y_train == 0)
sample_weights = np.ones(len(y_train))

for multiplier in [1.5, 2.0, 2.5]:
    sample_weights = np.ones(len(y_train))
    sample_weights[reject_mask] = multiplier
    
    rf_sw = RandomForestClassifier(
        n_estimators=400, max_depth=20, min_samples_split=3, min_samples_leaf=4,
        max_features='sqrt', random_state=42, n_jobs=-1
    )
    rf_sw.fit(X_train_prep, y_train, sample_weight=sample_weights)
    proba = rf_sw.predict_proba(X_test_prep)[:, 1]
    results.append(evaluate(y_test, proba, f"RF (sample_weight={multiplier}x reject)"))
    print(f"✓ RF sample_weight={multiplier}x done")

# =============================================================================
# 8. COMPILE AND ANALYZE RESULTS
# =============================================================================
print("\n" + "="*70)
print("RESULTS SUMMARY (All with 0.5 threshold)")
print("="*70)

results_df = pd.DataFrame(results)
results_df['Balanced_F1'] = 2 * (results_df['F1_Merged'] * results_df['F1_Rejected']) / \
                            (results_df['F1_Merged'] + results_df['F1_Rejected'])

# Sort by F1_Rejected
print("\n--- TOP 10 BY F1_REJECTED (threshold=0.5) ---")
top_rej = results_df.nlargest(10, 'F1_Rejected')
print(top_rej[['Strategy', 'F1_Rejected', 'Recall_Rejected', 'Precision_Rejected', 
               'F1_Merged', 'Brier Score']].to_string(index=False, float_format='%.4f'))

print("\n--- TOP 10 BY BALANCED F1 (threshold=0.5) ---")
top_bal = results_df.nlargest(10, 'Balanced_F1')
print(top_bal[['Strategy', 'Balanced_F1', 'F1_Merged', 'F1_Rejected',
               'ROC-AUC']].to_string(index=False, float_format='%.4f'))

print("\n--- TOP 10 BY RECALL_REJECTED (threshold=0.5) ---")
top_rec = results_df.nlargest(10, 'Recall_Rejected')
print(top_rec[['Strategy', 'Recall_Rejected', 'Precision_Rejected', 
               'F1_Rejected', 'F1_Merged']].to_string(index=False, float_format='%.4f'))

# =============================================================================
# 9. BEST MODEL SELECTION
# =============================================================================
print("\n" + "="*70)
print("RECOMMENDATION (WITH 0.5 THRESHOLD)")
print("="*70)

best_rej = results_df.loc[results_df['F1_Rejected'].idxmax()]
best_bal = results_df.loc[results_df['Balanced_F1'].idxmax()]
best_cal = results_df.loc[results_df['Brier Score'].idxmin()]

print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  BEST FOR CATCHING REJECTIONS:                                          │
│    Strategy: {best_rej['Strategy']:<50}      │
│    F1_Rejected: {best_rej['F1_Rejected']:.4f}  |  Recall_Rejected: {best_rej['Recall_Rejected']:.4f}       │
│    F1_Merged: {best_rej['F1_Merged']:.4f}                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  BEST BALANCED:                                                         │
│    Strategy: {best_bal['Strategy']:<50}      │
│    Balanced_F1: {best_bal['Balanced_F1']:.4f}                                        │
│    F1_Merged: {best_bal['F1_Merged']:.4f}  |  F1_Rejected: {best_bal['F1_Rejected']:.4f}              │
├─────────────────────────────────────────────────────────────────────────┤
│  BEST CALIBRATED (most meaningful 0.5 threshold):                       │
│    Strategy: {best_cal['Strategy']:<50}      │
│    Brier Score: {best_cal['Brier Score']:.4f} (lower = better calibration)           │
│    F1_Rejected: {best_cal['F1_Rejected']:.4f}                                        │
└─────────────────────────────────────────────────────────────────────────┘
""")

# Save results
results_df.to_csv('threshold_05_strategies_comparison.csv', index=False)
print("Saved: threshold_05_strategies_comparison.csv")
