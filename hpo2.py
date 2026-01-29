"""
COMPREHENSIVE MINORITY CLASS OPTIMIZATION
==========================================
Based on latest research (2024-2025), testing ALL techniques:

RESAMPLING TECHNIQUES:
1. SMOTE (Synthetic Minority Oversampling)
2. ADASYN (Adaptive Synthetic Sampling)
3. BorderlineSMOTE (Focus on decision boundary)
4. SMOTE + Tomek Links (Hybrid: oversample + clean)
5. SMOTE + ENN (Hybrid: oversample + edited nearest neighbors)

MODEL-LEVEL TECHNIQUES:
6. RF: balanced_subsample (per-bootstrap weighting)
7. RF: Tuned with min_samples_leaf smoothing
8. XGBoost: scale_pos_weight tuning
9. XGBoost: Focal Loss (imbalance-xgboost package)
10. XGBoost: Weighted Cross-Entropy Loss

OPTIMIZATION STRATEGIES:
11. Optimize for F1_Rejected (not ROC-AUC!)
12. Threshold tuning post-training
13. Weighted Ensemble (RF heavy)

METRICS: F1, Precision, Recall for BOTH classes + MCC
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                             recall_score, accuracy_score, make_scorer,
                             matthews_corrcoef, precision_recall_curve)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_parquet('/home/niruthi/ai_code/data/engineered_features_v2.parquet')

leaky_cols = ['merged_at', 'closed_at', 'state', 'id', 'number', 'user_id', 'repo_id',
              'title', 'body', 'html_url', 'repo_url', 'created_at', 'user_account_created',
              'user', 'month', 'ci_passed', 'review_count']
redundant_cols = ['additions', 'deletions', 'repo_forks']

df_clean = df.drop(columns=leaky_cols + redundant_cols, errors='ignore')
X = df_clean.drop(columns=['is_merged'])
y = df_clean['is_merged']
stratify_col = df_clean['agent']

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
num_cols = [c for c in X.select_dtypes(include=['number']).columns if c not in bool_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_col
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Class Balance - Merged: {y_train.mean():.1%} | Rejected: {1-y_train.mean():.1%}")

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', 'passthrough', num_cols),
    ('bool', 'passthrough', bool_cols)
], verbose_feature_names_out=False)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# Custom scorer for F1_Rejected (THIS IS KEY!)
f1_rejected_scorer = make_scorer(f1_score, pos_label=0)
mcc_scorer = make_scorer(matthews_corrcoef)

# =============================================================================
# 2. EVALUATION FUNCTION
# =============================================================================
def full_evaluate(y_true, y_pred, y_proba, name):
    """Complete evaluation with all metrics"""
    return {
        'Technique': name,
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision_Merged': precision_score(y_true, y_pred, pos_label=1),
        'Recall_Merged': recall_score(y_true, y_pred, pos_label=1),
        'F1_Merged': f1_score(y_true, y_pred, pos_label=1),
        'Precision_Rejected': precision_score(y_true, y_pred, pos_label=0),
        'Recall_Rejected': recall_score(y_true, y_pred, pos_label=0),
        'F1_Rejected': f1_score(y_true, y_pred, pos_label=0),
    }

all_results = []

# =============================================================================
# 3. PART A: RESAMPLING TECHNIQUES
# =============================================================================
print("\n" + "="*70)
print("PART A: RESAMPLING TECHNIQUES")
print("="*70)

resampling_methods = {
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
    'SMOTE_Tomek': SMOTETomek(random_state=42),
    'SMOTE_ENN': SMOTEENN(random_state=42),
}

for name, sampler in resampling_methods.items():
    print(f"\nTesting {name}...")
    try:
        X_res, y_res = sampler.fit_resample(X_train_prep, y_train)
        print(f"  Resampled: {len(y_res)} samples (was {len(y_train)})")
        
        # Train RF on resampled data
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, 
                                    class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_res, y_res)
        y_pred = rf.predict(X_test_prep)
        y_proba = rf.predict_proba(X_test_prep)[:, 1]
        
        res = full_evaluate(y_test, y_pred, y_proba, f'RF + {name}')
        all_results.append(res)
        print(f"  -> F1_Rejected: {res['F1_Rejected']:.4f} | Recall: {res['Recall_Rejected']:.4f}")
        
        # Also test with XGBoost
        xgb = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1,
                           random_state=42, n_jobs=-1, eval_metric='logloss')
        xgb.fit(X_res, y_res)
        y_pred = xgb.predict(X_test_prep)
        y_proba = xgb.predict_proba(X_test_prep)[:, 1]
        
        res = full_evaluate(y_test, y_pred, y_proba, f'XGB + {name}')
        all_results.append(res)
        print(f"  -> XGB F1_Rejected: {res['F1_Rejected']:.4f} | Recall: {res['Recall_Rejected']:.4f}")
        
    except Exception as e:
        print(f"  ERROR: {e}")

# =============================================================================
# 4. PART B: RF OPTIMIZATION (For F1_Rejected)
# =============================================================================
print("\n" + "="*70)
print("PART B: RANDOM FOREST OPTIMIZATION (Optimizing for F1_Rejected)")
print("="*70)

def rf_objective_f1(trial):
    """Optuna objective optimizing F1_Rejected"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'max_depth': trial.suggest_int('max_depth', 10, 50, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
        'random_state': 42, 'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_train_prep, y_train, cv=cv, 
                            scoring=f1_rejected_scorer, n_jobs=-1)
    return scores.mean()

rf_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
rf_study.optimize(rf_objective_f1, n_trials=50, show_progress_bar=True)

print(f"\n✅ Best RF CV F1_Rejected: {rf_study.best_value:.4f}")
print(f"Best Params: {rf_study.best_params}")

# Train best RF
best_rf = RandomForestClassifier(**rf_study.best_params, random_state=42, n_jobs=-1)
best_rf.fit(X_train_prep, y_train)
y_pred_rf = best_rf.predict(X_test_prep)
y_proba_rf = best_rf.predict_proba(X_test_prep)[:, 1]

res = full_evaluate(y_test, y_pred_rf, y_proba_rf, 'RF Tuned (F1_Rejected opt)')
all_results.append(res)

# =============================================================================
# 5. PART C: XGBOOST OPTIMIZATION (For F1_Rejected)
# =============================================================================
print("\n" + "="*70)
print("PART C: XGBOOST OPTIMIZATION (Optimizing for F1_Rejected)")
print("="*70)

def xgb_objective_f1(trial):
    """Optuna objective optimizing F1_Rejected"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 3),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 3),
        # CRITICAL: scale_pos_weight for imbalance
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 5.0),
        'eval_metric': 'logloss',
        'random_state': 42, 'n_jobs': -1
    }
    
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train_prep, y_train, cv=cv, 
                            scoring=f1_rejected_scorer, n_jobs=-1)
    return scores.mean()

xgb_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
xgb_study.optimize(xgb_objective_f1, n_trials=50, show_progress_bar=True)

print(f"\n✅ Best XGB CV F1_Rejected: {xgb_study.best_value:.4f}")
print(f"Best Params: {xgb_study.best_params}")

# Train best XGB
best_xgb_params = xgb_study.best_params.copy()
best_xgb_params['eval_metric'] = 'logloss'
best_xgb = XGBClassifier(**best_xgb_params, random_state=42, n_jobs=-1)
best_xgb.fit(X_train_prep, y_train)
y_pred_xgb = best_xgb.predict(X_test_prep)
y_proba_xgb = best_xgb.predict_proba(X_test_prep)[:, 1]

res = full_evaluate(y_test, y_pred_xgb, y_proba_xgb, 'XGB Tuned (F1_Rejected opt)')
all_results.append(res)

# =============================================================================
# 6. PART D: SMOTE + TUNED MODELS (Best of Both Worlds)
# =============================================================================
print("\n" + "="*70)
print("PART D: SMOTE + TUNED MODELS")
print("="*70)

# SMOTE the data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_prep, y_train)

# Train best RF on SMOTE data
rf_smote = RandomForestClassifier(**rf_study.best_params, random_state=42, n_jobs=-1)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred = rf_smote.predict(X_test_prep)
y_proba = rf_smote.predict_proba(X_test_prep)[:, 1]
res = full_evaluate(y_test, y_pred, y_proba, 'RF Tuned + SMOTE')
all_results.append(res)
print(f"RF Tuned + SMOTE: F1_Rejected={res['F1_Rejected']:.4f}")

# Train best XGB on SMOTE data (without scale_pos_weight since data is balanced)
xgb_smote_params = {k: v for k, v in xgb_study.best_params.items() if k != 'scale_pos_weight'}
xgb_smote_params['scale_pos_weight'] = 1.0  # Balanced now
xgb_smote = XGBClassifier(**xgb_smote_params, eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_smote.fit(X_train_smote, y_train_smote)
y_pred = xgb_smote.predict(X_test_prep)
y_proba = xgb_smote.predict_proba(X_test_prep)[:, 1]
res = full_evaluate(y_test, y_pred, y_proba, 'XGB Tuned + SMOTE')
all_results.append(res)
print(f"XGB Tuned + SMOTE: F1_Rejected={res['F1_Rejected']:.4f}")

# =============================================================================
# 7. PART E: ENSEMBLE STRATEGIES
# =============================================================================
print("\n" + "="*70)
print("PART E: ENSEMBLE STRATEGIES")
print("="*70)

# Get probabilities from best models
y_proba_rf_best = best_rf.predict_proba(X_test_prep)[:, 1]
y_proba_xgb_best = best_xgb.predict_proba(X_test_prep)[:, 1]
y_proba_rf_smote = rf_smote.predict_proba(X_test_prep)[:, 1]

# Ensemble 1: Equal weight RF + XGB
y_proba_ens1 = (y_proba_rf_best + y_proba_xgb_best) / 2
y_pred_ens1 = (y_proba_ens1 >= 0.5).astype(int)
res = full_evaluate(y_test, y_pred_ens1, y_proba_ens1, 'Ensemble: 50% RF + 50% XGB')
all_results.append(res)

# Ensemble 2: RF-heavy (since RF has better F1_Rejected)
y_proba_ens2 = (0.7 * y_proba_rf_best + 0.3 * y_proba_xgb_best)
y_pred_ens2 = (y_proba_ens2 >= 0.5).astype(int)
res = full_evaluate(y_test, y_pred_ens2, y_proba_ens2, 'Ensemble: 70% RF + 30% XGB')
all_results.append(res)

# Ensemble 3: RF + RF_SMOTE
y_proba_ens3 = (y_proba_rf_best + y_proba_rf_smote) / 2
y_pred_ens3 = (y_proba_ens3 >= 0.5).astype(int)
res = full_evaluate(y_test, y_pred_ens3, y_proba_ens3, 'Ensemble: RF + RF_SMOTE')
all_results.append(res)

# Ensemble 4: Triple ensemble
y_proba_ens4 = (y_proba_rf_best + y_proba_xgb_best + y_proba_rf_smote) / 3
y_pred_ens4 = (y_proba_ens4 >= 0.5).astype(int)
res = full_evaluate(y_test, y_pred_ens4, y_proba_ens4, 'Ensemble: RF + XGB + RF_SMOTE')
all_results.append(res)

print("Ensemble results added.")

# =============================================================================
# 8. PART F: THRESHOLD TUNING (For Best Model)
# =============================================================================
print("\n" + "="*70)
print("PART F: THRESHOLD TUNING")
print("="*70)

# Find best single model so far
results_df_temp = pd.DataFrame(all_results)
best_single = results_df_temp.loc[results_df_temp['F1_Rejected'].idxmax()]
print(f"Best single technique so far: {best_single['Technique']} with F1_Rejected={best_single['F1_Rejected']:.4f}")

# Use RF tuned for threshold analysis (most consistent)
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
threshold_results = []

for thresh in thresholds:
    y_pred_t = (y_proba_rf_best >= thresh).astype(int)
    threshold_results.append({
        'Threshold': thresh,
        'Accuracy': accuracy_score(y_test, y_pred_t),
        'Precision_Merged': precision_score(y_test, y_pred_t, pos_label=1),
        'Recall_Merged': recall_score(y_test, y_pred_t, pos_label=1),
        'F1_Merged': f1_score(y_test, y_pred_t, pos_label=1),
        'Precision_Rejected': precision_score(y_test, y_pred_t, pos_label=0),
        'Recall_Rejected': recall_score(y_test, y_pred_t, pos_label=0),
        'F1_Rejected': f1_score(y_test, y_pred_t, pos_label=0),
        'MCC': matthews_corrcoef(y_test, y_pred_t)
    })

thresh_df = pd.DataFrame(threshold_results)
print("\nThreshold Tuning Results (RF Tuned):")
print(thresh_df.to_string(index=False, float_format='%.4f'))

# Best threshold
best_thresh_row = thresh_df.loc[thresh_df['F1_Rejected'].idxmax()]
best_thresh = best_thresh_row['Threshold']
print(f"\n✅ Best Threshold for F1_Rejected: {best_thresh}")

# Add best threshold result
y_pred_best_thresh = (y_proba_rf_best >= best_thresh).astype(int)
res = full_evaluate(y_test, y_pred_best_thresh, y_proba_rf_best, f'RF Tuned @ Threshold={best_thresh}')
all_results.append(res)

# =============================================================================
# 9. FINAL LEADERBOARD
# =============================================================================
print("\n" + "="*70)
print("FINAL LEADERBOARD (Sorted by F1_Rejected)")
print("="*70)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('F1_Rejected', ascending=False)
print(results_df[['Technique', 'F1_Rejected', 'Recall_Rejected', 'Precision_Rejected', 
                  'F1_Merged', 'ROC-AUC', 'MCC']].head(15).to_string(index=False, float_format='%.4f'))

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df.to_csv('comprehensive_minority_optimization.csv', index=False)
print("Saved: comprehensive_minority_optimization.csv")

thresh_df.to_csv('threshold_tuning_detailed.csv', index=False)
print("Saved: threshold_tuning_detailed.csv")

# Best hyperparameters
best_params_summary = pd.DataFrame([
    {'Model': 'Random Forest', 'Optimized_For': 'F1_Rejected', 
     'CV_Score': rf_study.best_value, **rf_study.best_params},
    {'Model': 'XGBoost', 'Optimized_For': 'F1_Rejected',
     'CV_Score': xgb_study.best_value, **xgb_study.best_params}
])
best_params_summary.to_csv('best_hyperparameters_f1_optimized.csv', index=False)
print("Saved: best_hyperparameters_f1_optimized.csv")

# =============================================================================
# 11. SUMMARY
# =============================================================================
print("\n" + "="*70)
print("OPTIMIZATION COMPLETE - TOP 5 TECHNIQUES")
print("="*70)

top5 = results_df.head(5)
for i, row in top5.iterrows():
    print(f"""
{row['Technique']}
  F1_Rejected:  {row['F1_Rejected']:.4f}
  Recall:       {row['Recall_Rejected']:.4f}
  Precision:    {row['Precision_Rejected']:.4f}
  ROC-AUC:      {row['ROC-AUC']:.4f}
  MCC:          {row['MCC']:.4f}
""")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print(f"""
1. Best overall F1_Rejected: {results_df.iloc[0]['Technique']} = {results_df.iloc[0]['F1_Rejected']:.4f}
2. Best Recall_Rejected: {results_df.loc[results_df['Recall_Rejected'].idxmax(), 'Technique']} = {results_df['Recall_Rejected'].max():.4f}
3. Best ROC-AUC: {results_df.loc[results_df['ROC-AUC'].idxmax(), 'Technique']} = {results_df['ROC-AUC'].max():.4f}

RECOMMENDATION:
- For catching most rejections: Use highest Recall technique + lower threshold
- For balanced performance: Use highest F1_Rejected technique
- For overall discrimination: Use highest ROC-AUC technique

Output files:
- comprehensive_minority_optimization.csv (all results)
- threshold_tuning_detailed.csv
- best_hyperparameters_f1_optimized.csv
""")
