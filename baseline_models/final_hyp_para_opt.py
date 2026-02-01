"""
FULL HYPERPARAMETER OPTIMIZATION
=================================
Models: XGBoost, Random Forest, Optimized Ensemble
Objective: Maximize F1_Rejected @ threshold=0.5
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
import joblib
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

THRESHOLD = 0.5

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("="*70)
print("FULL HYPERPARAMETER OPTIMIZATION")
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

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# =============================================================================
# 2. OPTIMIZE XGBOOST
# =============================================================================
print("\n" + "="*70)
print("OPTIMIZING XGBOOST (50 trials)")
print("="*70)

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.3, 0.8),
        'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'
    }
    
    model = XGBClassifier(**params)
    oof_proba = cross_val_predict(model, X_train_prep, y_train, cv=cv, 
                                   method='predict_proba', n_jobs=-1)[:, 1]
    oof_pred = (oof_proba >= THRESHOLD).astype(int)
    return f1_score(y_train, oof_pred, pos_label=0)

xgb_study = optuna.create_study(direction='maximize', 
                                 sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

print(f"✓ Best XGB F1_Rejected (CV): {xgb_study.best_value:.4f}")
print(f"  Best scale_pos_weight: {xgb_study.best_params['scale_pos_weight']:.3f}")

# =============================================================================
# 3. OPTIMIZE RANDOM FOREST
# =============================================================================
print("\n" + "="*70)
print("OPTIMIZING RANDOM FOREST (50 trials)")
print("="*70)

def rf_objective(trial):
    # Class weight for rejected class
    reject_weight = trial.suggest_float('reject_weight', 1.5, 4.0)
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=100),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
        'class_weight': {0: reject_weight, 1: 1.0},
        'random_state': 42, 'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    oof_proba = cross_val_predict(model, X_train_prep, y_train, cv=cv, 
                                   method='predict_proba', n_jobs=-1)[:, 1]
    oof_pred = (oof_proba >= THRESHOLD).astype(int)
    return f1_score(y_train, oof_pred, pos_label=0)

rf_study = optuna.create_study(direction='maximize', 
                                sampler=optuna.samplers.TPESampler(seed=42))
rf_study.optimize(rf_objective, n_trials=50, show_progress_bar=True)

print(f"✓ Best RF F1_Rejected (CV): {rf_study.best_value:.4f}")
print(f"  Best reject_weight: {rf_study.best_params['reject_weight']:.3f}")

# =============================================================================
# 4. TRAIN OPTIMIZED MODELS
# =============================================================================
print("\n" + "="*70)
print("TRAINING OPTIMIZED MODELS")
print("="*70)

# XGBoost
xgb_params = xgb_study.best_params.copy()
xgb_params.update({'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'})
xgb_opt = XGBClassifier(**xgb_params)
xgb_opt.fit(X_train_prep, y_train)
xgb_proba = xgb_opt.predict_proba(X_test_prep)[:, 1]
print("✓ XGBoost trained")

# Random Forest
rf_params = rf_study.best_params.copy()
reject_weight = rf_params.pop('reject_weight')
rf_params['class_weight'] = {0: reject_weight, 1: 1.0}
rf_params.update({'random_state': 42, 'n_jobs': -1})
rf_opt = RandomForestClassifier(**rf_params)
rf_opt.fit(X_train_prep, y_train)
rf_proba = rf_opt.predict_proba(X_test_prep)[:, 1]
print("✓ Random Forest trained")

# =============================================================================
# 5. OPTIMIZE ENSEMBLE WEIGHTS
# =============================================================================
print("\n" + "="*70)
print("OPTIMIZING ENSEMBLE WEIGHTS (30 trials)")
print("="*70)

# Get OOF predictions for ensemble optimization
xgb_oof = cross_val_predict(xgb_opt, X_train_prep, y_train, cv=cv, 
                             method='predict_proba', n_jobs=-1)[:, 1]
rf_oof = cross_val_predict(rf_opt, X_train_prep, y_train, cv=cv, 
                            method='predict_proba', n_jobs=-1)[:, 1]

def ensemble_objective(trial):
    rf_weight = trial.suggest_float('rf_weight', 0.2, 0.8)
    xgb_weight = 1 - rf_weight
    
    ens_proba = rf_weight * rf_oof + xgb_weight * xgb_oof
    ens_pred = (ens_proba >= THRESHOLD).astype(int)
    return f1_score(y_train, ens_pred, pos_label=0)

ens_study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
ens_study.optimize(ensemble_objective, n_trials=30, show_progress_bar=True)

best_rf_w = ens_study.best_params['rf_weight']
best_xgb_w = 1 - best_rf_w
print(f"✓ Optimal weights: RF={best_rf_w:.3f}, XGB={best_xgb_w:.3f}")

# =============================================================================
# 6. FINAL EVALUATION
# =============================================================================
print("\n" + "="*70)
print("FINAL TEST SET EVALUATION (threshold=0.5)")
print("="*70)

def evaluate(name, y_proba):
    y_pred = (y_proba >= THRESHOLD).astype(int)
    return {
        'Model': name,
        'F1_Rejected': f1_score(y_test, y_pred, pos_label=0),
        'Recall_Rejected': recall_score(y_test, y_pred, pos_label=0),
        'Precision_Rejected': precision_score(y_test, y_pred, pos_label=0),
        'F1_Merged': f1_score(y_test, y_pred, pos_label=1),
    }

# Ensemble predictions
ens_proba = best_rf_w * rf_proba + best_xgb_w * xgb_proba

results = [
    evaluate('XGBoost (optimized)', xgb_proba),
    evaluate('Random Forest (optimized)', rf_proba),
    evaluate(f'Ensemble (RF={best_rf_w:.2f}/XGB={best_xgb_w:.2f})', ens_proba),
]

# Add unoptimized baselines for comparison
xgb_baseline = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=8,
                              scale_pos_weight=0.5, random_state=42, n_jobs=-1)
xgb_baseline.fit(X_train_prep, y_train)
results.append(evaluate('XGBoost (unoptimized)', xgb_baseline.predict_proba(X_test_prep)[:, 1]))

rf_baseline = RandomForestClassifier(n_estimators=400, max_depth=20, 
                                      class_weight='balanced', random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_prep, y_train)
results.append(evaluate('RF (unoptimized)', rf_baseline.predict_proba(X_test_prep)[:, 1]))

results_df = pd.DataFrame(results)
results_df['Balanced_F1'] = 2 * (results_df['F1_Merged'] * results_df['F1_Rejected']) / \
                            (results_df['F1_Merged'] + results_df['F1_Rejected'])

print("\n" + results_df.sort_values('F1_Rejected', ascending=False).to_string(index=False, float_format='%.4f'))

# =============================================================================
# 7. WINNER SELECTION
# =============================================================================
print("\n" + "="*70)
print("WINNER")
print("="*70)

best_idx = results_df['F1_Rejected'].idxmax()
best = results_df.loc[best_idx]

print(f"""
┌────────────────────────────────────────────────────────────────────┐
│  🏆 BEST MODEL: {best['Model']:<45}│
├────────────────────────────────────────────────────────────────────┤
│  F1_Rejected:       {best['F1_Rejected']:.4f}                                      │
│  Recall_Rejected:   {best['Recall_Rejected']:.4f}                                      │
│  Precision_Rejected:{best['Precision_Rejected']:.4f}                                      │
│  F1_Merged:         {best['F1_Merged']:.4f}                                      │
│  Balanced_F1:       {best['Balanced_F1']:.4f}                                      │
└────────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# 8. SAVE EVERYTHING
# =============================================================================
print("="*70)
print("SAVING MODELS")
print("="*70)

joblib.dump(xgb_opt, 'xgb_optimized.joblib')
joblib.dump(rf_opt, 'rf_optimized.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump({
    'xgb_params': xgb_study.best_params,
    'rf_params': rf_study.best_params,
    'ensemble_weights': {'rf': best_rf_w, 'xgb': best_xgb_w}
}, 'best_hyperparameters.joblib')

print("✓ xgb_optimized.joblib")
print("✓ rf_optimized.joblib")
print("✓ preprocessor.joblib")
print("✓ best_hyperparameters.joblib")

# =============================================================================
# 9. PRINT FINAL CONFIGURATIONS
# =============================================================================
print("\n" + "="*70)
print("FINAL CONFIGURATIONS (for your report)")
print("="*70)

print("\n--- XGBoost ---")
print("XGBClassifier(")
for k, v in xgb_study.best_params.items():
    if isinstance(v, float):
        print(f"    {k}={v:.4f},")
    else:
        print(f"    {k}={v},")
print(")")

print("\n--- Random Forest ---")
print("RandomForestClassifier(")
for k, v in rf_study.best_params.items():
    if isinstance(v, float):
        print(f"    {k}={v:.4f},")
    else:
        print(f"    {k}={v},")
print(f"    class_weight={{0: {reject_weight:.4f}, 1: 1.0}},")
print(")")

print(f"\n--- Ensemble ---")
print(f"ensemble_proba = {best_rf_w:.3f} * rf_proba + {best_xgb_w:.3f} * xgb_proba")

results_df.to_csv('optimization_results.csv', index=False)
print("\n✓ Saved: optimization_results.csv")
