"""
FINAL AGENTIC PR ANALYSIS SCRIPT V2
===================================
UPDATES:
- Added 5 new Historical CI & Review features
- Added XGBoost model
- Updated leaky columns list
- Added feature group analysis
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score, 
                             accuracy_score, f1_score, precision_score, recall_score)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("="*70)
print("LOADING AND PREPARING DATA (V2 - With New Features)")
print("="*70)

# Load Data - NOW WITH NEW FEATURES
df = pd.read_parquet('/home/niruthi/ai_code/data/engineered_features_v2.parquet')

# Define Leaky & Redundant Columns (UPDATED)
leaky_cols = [
    # Original leaky columns
    'merged_at', 'closed_at', 'state', 'id', 'number', 'user_id', 'repo_id',
    'title', 'body', 'html_url', 'repo_url', 'created_at', 'user_account_created',
    'user', 'month',
    # NEW: Leaky columns from API fetch (post-submission data)
    'ci_passed',      # This PR's actual CI result - LEAKY
    'review_count',   # This PR's actual review count - LEAKY
]
redundant_cols = ['additions', 'deletions', 'repo_forks']

# Drop columns
df_clean = df.drop(columns=leaky_cols + redundant_cols, errors='ignore')

# Define X (Features) and y (Target)
X = df_clean.drop(columns=['is_merged'])
y = df_clean['is_merged']

# Stratify by agent
stratify_col = df_clean['agent']

# Identify Column Types
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()
num_cols = [c for c in num_cols if c not in bool_cols]

# Identify NEW features
new_features = [
    'agent_historical_ci_pass_rate',
    'user_historical_ci_pass_rate', 
    'repo_avg_ci_fail_rate',
    'agent_avg_review_rounds',
    'repo_avg_time_to_merge'
]
new_features_present = [f for f in new_features if f in X.columns]

print(f"Total Features: {X.shape[1]}")
print(f"Categorical: {len(cat_cols)} | Numerical: {len(num_cols)} | Boolean: {len(bool_cols)}")
print(f"\n✅ NEW Features Found: {len(new_features_present)}")
for f in new_features_present:
    print(f"   - {f}: mean={X[f].mean():.3f}, missing={X[f].isna().sum()}")

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_col
)
test_agents = X_test['agent'].copy()

print(f"\nTrain Size: {len(X_train)} | Test Size: {len(X_test)}")
print(f"Target Distribution - Merged: {y.mean():.1%} | Rejected: {1-y.mean():.1%}")

# =============================================================================
# 2. DEFINE ROBUST PIPELINES (Including XGBoost)
# =============================================================================
print("\n" + "="*70)
print("DEFINING MODEL PIPELINES")
print("="*70)

# Preprocessor for Tree Models
tree_preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', 'passthrough', num_cols),
    ('bool', 'passthrough', bool_cols)
], verbose_feature_names_out=False)

# Preprocessor for Linear Models
linear_preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', max_categories=20, sparse_output=False), cat_cols),
    ('num', StandardScaler(), num_cols),
    ('bool', 'passthrough', bool_cols)
])

# Calculate scale_pos_weight for XGBoost (handles imbalance)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Define Models (ADDED XGBoost)
models = {
    'Logistic Regression': Pipeline([
        ('prep', linear_preprocessor),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ]),
    'Decision Tree': Pipeline([
        ('prep', tree_preprocessor),
        ('clf', DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('prep', tree_preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=200, max_depth=15, 
            class_weight='balanced', n_jobs=-1, random_state=42
        ))
    ]),
    'Gradient Boosting': Pipeline([
        ('prep', tree_preprocessor),
        ('clf', HistGradientBoostingClassifier(class_weight='balanced', random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('prep', tree_preprocessor),
        ('clf', XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,  # Handle imbalance
            eval_metric='auc',
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42
        ))
    ])
}

print(f"Models to train: {list(models.keys())}")

# =============================================================================
# 3. TRAIN AND EVALUATE
# =============================================================================
print("\n" + "="*70)
print("TRAINING AND EVALUATING MODELS")
print("="*70)

results = []

for name, pipe in models.items():
    print(f"\nTraining {name}...")
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    res = {
        'Model': name,
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision_Merged': precision_score(y_test, y_pred, pos_label=1),
        'Recall_Merged': recall_score(y_test, y_pred, pos_label=1),
        'F1_Merged': f1_score(y_test, y_pred, pos_label=1),
        'Precision_Rejected': precision_score(y_test, y_pred, pos_label=0),
        'Recall_Rejected': recall_score(y_test, y_pred, pos_label=0),
        'F1_Rejected': f1_score(y_test, y_pred, pos_label=0),
    }
    results.append(res)
    print(f"  -> AUC: {res['ROC-AUC']:.4f} | F1_Merged: {res['F1_Merged']:.4f} | F1_Rejected: {res['F1_Rejected']:.4f}")

# Leaderboard
results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
print("\n" + "="*70)
print("FINAL MODEL LEADERBOARD")
print("="*70)
print(results_df.to_string(index=False, float_format='%.4f'))

results_df.to_csv('model_comparison_results_v2.csv', index=False)
print("\nSaved: model_comparison_results_v2.csv")

# =============================================================================
# 4. FEATURE IMPORTANCE COMPARISON (New vs Old Features)
# =============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE: NEW FEATURES IMPACT")
print("="*70)

# Get feature names after preprocessing
rf_pipe = models['Random Forest']
rf_prep = rf_pipe.named_steps['prep']
rf_clf = rf_pipe.named_steps['clf']

try:
    feature_names = list(rf_prep.get_feature_names_out())
except:
    feature_names = cat_cols + num_cols + bool_cols

# Feature importances
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 20 FEATURES (Random Forest):")
print(feat_imp.head(20).to_string(index=False, float_format='%.4f'))

# Check where NEW features rank
print("\n🆕 NEW FEATURES RANKING:")
for feat in new_features_present:
    if feat in feat_imp['feature'].values:
        rank = feat_imp[feat_imp['feature'] == feat].index[0] + 1
        imp = feat_imp[feat_imp['feature'] == feat]['importance'].values[0]
        print(f"   {feat}: Rank #{rank}, Importance={imp:.4f}")

feat_imp.to_csv('feature_importances_v2.csv', index=False)
print("\nSaved: feature_importances_v2.csv")

# =============================================================================
# 5. PER-AGENT PERFORMANCE (Best Model)
# =============================================================================
print("\n" + "="*70)
print("PER-AGENT PERFORMANCE (Best Model)")
print("="*70)

# Use best model based on ROC-AUC
best_model_name = results_df.iloc[0]['Model']
best_pipe = models[best_model_name]
print(f"Using: {best_model_name}")

y_pred_best = best_pipe.predict(X_test)
y_proba_best = best_pipe.predict_proba(X_test)[:, 1]

agent_metrics = []
for agent in test_agents.unique():
    mask = (test_agents == agent)
    if mask.sum() > 10:
        auc = roc_auc_score(y_test[mask], y_proba_best[mask])
        acc = accuracy_score(y_test[mask], y_pred_best[mask])
        f1_rej = f1_score(y_test[mask], y_pred_best[mask], pos_label=0)
        agent_metrics.append({
            'Agent': agent, 
            'ROC-AUC': auc, 
            'Accuracy': acc, 
            'F1_Rejected': f1_rej,
            'Count': mask.sum()
        })

agent_df = pd.DataFrame(agent_metrics).sort_values('ROC-AUC', ascending=False)
print(agent_df.to_string(index=False, float_format='%.3f'))

agent_df.to_csv('per_agent_performance_v2.csv', index=False)
print("\nSaved: per_agent_performance_v2.csv")

# =============================================================================
# 6. SHAP ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("GENERATING SHAP PLOTS")
print("="*70)

# Sample for speed
X_shap_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
X_shap_transformed = rf_prep.transform(X_shap_sample)

explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_shap_transformed)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_values, 
    X_shap_transformed, 
    feature_names=feature_names,
    max_display=20, 
    show=False
)
plt.title("Feature Impact on PR Merge Prediction (SHAP Values)")
plt.tight_layout()
plt.savefig('shap_analysis_v2.png', dpi=300, bbox_inches='tight')
print("Saved: shap_analysis_v2.png")

# =============================================================================
# 7. ABLATION STUDY: With vs Without New Features
# =============================================================================
print("\n" + "="*70)
print("ABLATION STUDY: Impact of New Features")
print("="*70)

# Train RF without new features
X_train_old = X_train.drop(columns=new_features_present, errors='ignore')
X_test_old = X_test.drop(columns=new_features_present, errors='ignore')

# Update column lists for old features
cat_cols_old = [c for c in cat_cols if c not in new_features_present]
num_cols_old = [c for c in num_cols if c not in new_features_present]
bool_cols_old = [c for c in bool_cols if c not in new_features_present]

tree_prep_old = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols_old),
    ('num', 'passthrough', num_cols_old),
    ('bool', 'passthrough', bool_cols_old)
], verbose_feature_names_out=False)

rf_old = Pipeline([
    ('prep', tree_prep_old),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=15, 
                                   class_weight='balanced', n_jobs=-1, random_state=42))
])

rf_old.fit(X_train_old, y_train)
y_pred_old = rf_old.predict(X_test_old)
y_proba_old = rf_old.predict_proba(X_test_old)[:, 1]

# Compare
ablation_results = {
    'Metric': ['ROC-AUC', 'F1_Rejected', 'Recall_Rejected', 'Precision_Rejected'],
    'Without New Features': [
        roc_auc_score(y_test, y_proba_old),
        f1_score(y_test, y_pred_old, pos_label=0),
        recall_score(y_test, y_pred_old, pos_label=0),
        precision_score(y_test, y_pred_old, pos_label=0)
    ],
    'With New Features': [
        roc_auc_score(y_test, y_proba_best) if best_model_name == 'Random Forest' else roc_auc_score(y_test, rf_pipe.predict_proba(X_test)[:, 1]),
        f1_score(y_test, rf_pipe.predict(X_test), pos_label=0),
        recall_score(y_test, rf_pipe.predict(X_test), pos_label=0),
        precision_score(y_test, rf_pipe.predict(X_test), pos_label=0)
    ]
}

ablation_df = pd.DataFrame(ablation_results)
ablation_df['Improvement'] = ablation_df['With New Features'] - ablation_df['Without New Features']
ablation_df['% Change'] = (ablation_df['Improvement'] / ablation_df['Without New Features'] * 100).round(2)

print(ablation_df.to_string(index=False, float_format='%.4f'))

ablation_df.to_csv('ablation_study_new_features.csv', index=False)
print("\nSaved: ablation_study_new_features.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE - V2")
print("="*70)
print("Output files generated:")
print("  - model_comparison_results_v2.csv")
print("  - feature_importances_v2.csv")
print("  - per_agent_performance_v2.csv")
print("  - shap_analysis_v2.png")
print("  - ablation_study_new_features.csv")
