"""
FULL COMPARISON: MODEL × STRATEGY MATRIX
========================================
Models: RF Only, XGBoost Only, Ensemble (70/30), Ensemble (50/50)
Strategies: Baseline, Per-Agent Thresholds, Risk Tiers, Equalized Odds

This finds the BEST combination of model + post-processing strategy.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, 
                             recall_score, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("="*70)
print("MODEL × STRATEGY FULL COMPARISON")
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
agents_train = X_train['agent'].values
agents_test = X_test['agent'].values

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

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

# =============================================================================
# 2. TRAIN ALL MODELS
# =============================================================================
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

# Model 1: Random Forest (Tuned for F1_Rejected)
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=400, max_depth=20, min_samples_split=3, min_samples_leaf=4,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
)
rf_model.fit(X_train_prep, y_train)

# Model 2: XGBoost (FIXED - low scale_pos_weight)
print("Training XGBoost (Fixed)...")
xgb_model_fixed = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=8, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.7, scale_pos_weight=1.5,  # LOW
    gamma=1.0, reg_alpha=0.5, reg_lambda=1.0,
    eval_metric='logloss', random_state=42, n_jobs=-1
)
xgb_model_fixed.fit(X_train_prep, y_train)

# Model 3: XGBoost (Original - high scale_pos_weight for comparison)
print("Training XGBoost (Original - high weight)...")
xgb_model_orig = XGBClassifier(
    n_estimators=800, learning_rate=0.015, max_depth=12, min_child_weight=7,
    subsample=0.83, colsample_bytree=0.60, scale_pos_weight=8.8,  # HIGH (original)
    eval_metric='logloss', random_state=42, n_jobs=-1
)
xgb_model_orig.fit(X_train_prep, y_train)

print("✅ All models trained.")

# Get probabilities
rf_proba = rf_model.predict_proba(X_test_prep)[:, 1]
xgb_fixed_proba = xgb_model_fixed.predict_proba(X_test_prep)[:, 1]
xgb_orig_proba = xgb_model_orig.predict_proba(X_test_prep)[:, 1]

# Ensembles
ens_70_30_proba = 0.7 * rf_proba + 0.3 * xgb_fixed_proba
ens_50_50_proba = 0.5 * rf_proba + 0.5 * xgb_fixed_proba

# All model probabilities
models = {
    'RF Only': rf_proba,
    'XGB Fixed (spw=1.5)': xgb_fixed_proba,
    'XGB Original (spw=8.8)': xgb_orig_proba,
    'Ensemble 70/30': ens_70_30_proba,
    'Ensemble 50/50': ens_50_50_proba,
}

# =============================================================================
# 3. GENERATE OOF PREDICTIONS FOR CALIBRATION
# =============================================================================
print("\n" + "="*70)
print("GENERATING OOF PREDICTIONS")
print("="*70)

rf_oof = cross_val_predict(
    RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_split=3,
                          min_samples_leaf=4, max_features='sqrt', 
                          class_weight='balanced', random_state=42, n_jobs=-1),
    X_train_prep, y_train, cv=5, method='predict_proba', n_jobs=-1
)[:, 1]
print("OOF predictions generated.")

# =============================================================================
# 4. DEFINE STRATEGIES
# =============================================================================
print("\n" + "="*70)
print("CALCULATING STRATEGY PARAMETERS")
print("="*70)

# Helper function
def find_threshold_for_recall(y_true, y_proba, target_recall=0.60):
    best_thresh, best_diff = 0.5, 1.0
    for thresh in np.arange(0.2, 0.8, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        if abs(recall - target_recall) < best_diff:
            best_diff = abs(recall - target_recall)
            best_thresh = thresh
    return best_thresh

# Calculate per-agent parameters
global_merge_rate = y_train.mean()
agent_params = {}

for agent in np.unique(agents_train):
    mask = (agents_train == agent)
    merge_rate = y_train.values[mask].mean()
    
    # Risk tier
    if merge_rate >= 0.80:
        tier = 'LOW_RISK'
    elif merge_rate >= 0.65:
        tier = 'MEDIUM_RISK'
    else:
        tier = 'HIGH_RISK'
    
    # Per-agent threshold (60% recall target)
    if mask.sum() > 50:
        thresh_60 = find_threshold_for_recall(y_train.values[mask], rf_oof[mask], 0.60)
        thresh_55 = find_threshold_for_recall(y_train.values[mask], rf_oof[mask], 0.55)
    else:
        thresh_60 = {'LOW_RISK': 0.65, 'MEDIUM_RISK': 0.55, 'HIGH_RISK': 0.45}[tier]
        thresh_55 = thresh_60
    
    agent_params[agent] = {
        'merge_rate': merge_rate,
        'tier': tier,
        'threshold_60': thresh_60,
        'threshold_55': thresh_55,
    }
    print(f"  {agent}: rate={merge_rate:.3f}, tier={tier}, thresh_60={thresh_60:.2f}")

# Tier thresholds
tier_thresholds = {'LOW_RISK': 0.65, 'MEDIUM_RISK': 0.55, 'HIGH_RISK': 0.45}

# =============================================================================
# 5. DEFINE STRATEGY FUNCTIONS
# =============================================================================
def apply_strategy(y_proba, agents, strategy_name):
    """Apply a post-processing strategy and return predictions"""
    y_pred = np.zeros(len(y_proba))
    
    if strategy_name == 'Baseline (0.5)':
        y_pred = (y_proba >= 0.5).astype(int)
    
    elif strategy_name == 'Per-Agent (60% Recall)':
        for i, (agent, prob) in enumerate(zip(agents, y_proba)):
            thresh = agent_params.get(agent, {}).get('threshold_60', 0.55)
            y_pred[i] = 1 if prob >= thresh else 0
    
    elif strategy_name == 'Per-Agent (55% Recall)':
        for i, (agent, prob) in enumerate(zip(agents, y_proba)):
            thresh = agent_params.get(agent, {}).get('threshold_55', 0.55)
            y_pred[i] = 1 if prob >= thresh else 0
    
    elif strategy_name == 'Risk Tiers':
        for i, (agent, prob) in enumerate(zip(agents, y_proba)):
            tier = agent_params.get(agent, {}).get('tier', 'MEDIUM_RISK')
            thresh = tier_thresholds[tier]
            y_pred[i] = 1 if prob >= thresh else 0
    
    elif strategy_name == 'Aggressive (0.6)':
        y_pred = (y_proba >= 0.6).astype(int)
    
    elif strategy_name == 'Conservative (0.4)':
        y_pred = (y_proba >= 0.4).astype(int)
    
    return y_pred.astype(int)

strategies = [
    'Baseline (0.5)',
    'Per-Agent (60% Recall)',
    'Per-Agent (55% Recall)',
    'Risk Tiers',
    'Aggressive (0.6)',
    'Conservative (0.4)',
]

# =============================================================================
# 6. RUN ALL COMBINATIONS
# =============================================================================
print("\n" + "="*70)
print("EVALUATING ALL MODEL × STRATEGY COMBINATIONS")
print("="*70)

all_results = []

for model_name, proba in models.items():
    for strategy in strategies:
        y_pred = apply_strategy(proba, agents_test, strategy)
        
        result = {
            'Model': model_name,
            'Strategy': strategy,
            'ROC-AUC': roc_auc_score(y_test, proba),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision_Merged': precision_score(y_test, y_pred, pos_label=1),
            'Recall_Merged': recall_score(y_test, y_pred, pos_label=1),
            'F1_Merged': f1_score(y_test, y_pred, pos_label=1),
            'Precision_Rejected': precision_score(y_test, y_pred, pos_label=0),
            'Recall_Rejected': recall_score(y_test, y_pred, pos_label=0),
            'F1_Rejected': f1_score(y_test, y_pred, pos_label=0),
        }
        all_results.append(result)

results_df = pd.DataFrame(all_results)

# =============================================================================
# 7. ANALYSIS: BEST COMBINATIONS
# =============================================================================
print("\n" + "="*70)
print("TOP 10 COMBINATIONS BY F1_REJECTED")
print("="*70)
top_f1_rej = results_df.nlargest(10, 'F1_Rejected')
print(top_f1_rej[['Model', 'Strategy', 'F1_Rejected', 'Recall_Rejected', 
                   'Precision_Rejected', 'F1_Merged', 'ROC-AUC']].to_string(index=False, float_format='%.4f'))

print("\n" + "="*70)
print("TOP 10 COMBINATIONS BY RECALL_REJECTED")
print("="*70)
top_recall = results_df.nlargest(10, 'Recall_Rejected')
print(top_recall[['Model', 'Strategy', 'Recall_Rejected', 'Precision_Rejected',
                   'F1_Rejected', 'F1_Merged', 'Accuracy']].to_string(index=False, float_format='%.4f'))

print("\n" + "="*70)
print("BEST BALANCED (Harmonic Mean of F1_Merged & F1_Rejected)")
print("="*70)
results_df['Balanced_F1'] = 2 * (results_df['F1_Merged'] * results_df['F1_Rejected']) / \
                            (results_df['F1_Merged'] + results_df['F1_Rejected'])
top_balanced = results_df.nlargest(10, 'Balanced_F1')
print(top_balanced[['Model', 'Strategy', 'Balanced_F1', 'F1_Merged', 
                    'F1_Rejected', 'ROC-AUC']].to_string(index=False, float_format='%.4f'))

# =============================================================================
# 8. PIVOT TABLES: MODEL vs STRATEGY
# =============================================================================
print("\n" + "="*70)
print("F1_REJECTED: MODEL × STRATEGY MATRIX")
print("="*70)
pivot_f1_rej = results_df.pivot_table(index='Model', columns='Strategy', 
                                       values='F1_Rejected', aggfunc='mean')
print(pivot_f1_rej.round(4).to_string())

print("\n" + "="*70)
print("RECALL_REJECTED: MODEL × STRATEGY MATRIX")
print("="*70)
pivot_recall = results_df.pivot_table(index='Model', columns='Strategy', 
                                       values='Recall_Rejected', aggfunc='mean')
print(pivot_recall.round(4).to_string())

print("\n" + "="*70)
print("F1_MERGED: MODEL × STRATEGY MATRIX")
print("="*70)
pivot_f1_merged = results_df.pivot_table(index='Model', columns='Strategy', 
                                          values='F1_Merged', aggfunc='mean')
print(pivot_f1_merged.round(4).to_string())

# =============================================================================
# 9. PER-AGENT ANALYSIS FOR TOP 3 COMBINATIONS
# =============================================================================
print("\n" + "="*70)
print("PER-AGENT BREAKDOWN: TOP 3 COMBINATIONS")
print("="*70)

top3 = results_df.nlargest(3, 'F1_Rejected')[['Model', 'Strategy']].values

for model_name, strategy in top3:
    print(f"\n>>> {model_name} + {strategy}")
    print("-" * 60)
    
    proba = models[model_name]
    y_pred = apply_strategy(proba, agents_test, strategy)
    
    print(f"{'Agent':<15} {'F1_Merged':<12} {'F1_Rejected':<12} {'Recall_Rej':<12}")
    
    for agent in np.unique(agents_test):
        mask = (agents_test == agent)
        if mask.sum() > 10:
            y_t = y_test.values[mask]
            y_p = y_pred[mask]
            f1_m = f1_score(y_t, y_p, pos_label=1, zero_division=0)
            f1_r = f1_score(y_t, y_p, pos_label=0, zero_division=0)
            rec_r = recall_score(y_t, y_p, pos_label=0, zero_division=0)
            print(f"{agent:<15} {f1_m:<12.3f} {f1_r:<12.3f} {rec_r:<12.3f}")

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df.to_csv('model_strategy_full_comparison.csv', index=False)
print("Saved: model_strategy_full_comparison.csv")

pivot_f1_rej.to_csv('pivot_f1_rejected.csv')
print("Saved: pivot_f1_rejected.csv")

pivot_recall.to_csv('pivot_recall_rejected.csv')
print("Saved: pivot_recall_rejected.csv")

pivot_f1_merged.to_csv('pivot_f1_merged.csv')
print("Saved: pivot_f1_merged.csv")

# =============================================================================
# 11. FINAL RECOMMENDATION
# =============================================================================
print("\n" + "="*70)
print("FINAL RECOMMENDATION")
print("="*70)

best_overall = results_df.loc[results_df['Balanced_F1'].idxmax()]
best_rejection = results_df.loc[results_df['F1_Rejected'].idxmax()]
best_recall = results_df.loc[results_df['Recall_Rejected'].idxmax()]

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WINNING COMBINATIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  BEST BALANCED (F1 both classes):                                           │
│    Model:    {best_overall['Model']:<40}                                    │
│    Strategy: {best_overall['Strategy']:<40}                                 │
│    Balanced_F1: {best_overall['Balanced_F1']:.4f}                           │
│    F1_Merged: {best_overall['F1_Merged']:.4f} | F1_Rejected: {best_overall['F1_Rejected']:.4f}              │
├─────────────────────────────────────────────────────────────────────────────┤
│  BEST FOR CATCHING REJECTIONS (F1_Rejected):                                │
│    Model:    {best_rejection['Model']:<40}                                  │
│    Strategy: {best_rejection['Strategy']:<40}                               │
│    F1_Rejected: {best_rejection['F1_Rejected']:.4f}                         │
│    Recall_Rejected: {best_rejection['Recall_Rejected']:.4f}                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  MAXIMUM REJECTION RECALL:                                                  │
│    Model:    {best_recall['Model']:<40}                                     │
│    Strategy: {best_recall['Strategy']:<40}                                  │
│    Recall_Rejected: {best_recall['Recall_Rejected']:.4f}                    │
│    (Precision trade-off: {best_recall['Precision_Rejected']:.4f})           │
└─────────────────────────────────────────────────────────────────────────────┘

Output files:
  - model_strategy_full_comparison.csv (all 30 combinations)
  - pivot_f1_rejected.csv
  - pivot_recall_rejected.csv  
  - pivot_f1_merged.csv
""")
