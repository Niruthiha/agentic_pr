"""
FINAL AGENTIC PR ANALYSIS SCRIPT (AGENT-ONLY STRATIFICATION)
============================================================
1. Stratified Split by AGENT ONLY (Preserves Agent Ratios, ignores Outcome)
2. Robust Pipelines (No Data Leakage / Handles Unknown Categories)
3. "Fair Fight" Model Comparison
4. SHAP Interpretation for "Ambition Trade-off"
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score, 
                             accuracy_score, f1_score, precision_score, recall_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("="*70)
print("LOADING AND PREPARING DATA")
print("="*70)

# Load Data
df = pd.read_parquet('/home/niruthi/ai_code/data/engineered_features.parquet')

# Define Leaky & Redundant Columns
leaky_cols = ['merged_at', 'closed_at', 'state', 'id', 'number', 'user_id', 'repo_id',
              'title', 'body', 'html_url', 'repo_url', 'created_at', 'user_account_created',
              'user', 'month']
redundant_cols = ['additions', 'deletions', 'repo_forks']

# Drop columns but KEEP target and stratification helpers
df_clean = df.drop(columns=leaky_cols + redundant_cols, errors='ignore')

# Define X (Features) and y (Target)
# Note: We keep 'agent' in X for now to use it as a feature
X = df_clean.drop(columns=['is_merged'])
y = df_clean['is_merged']

# --- AGENT-ONLY STRATIFICATION ---
# We use the raw 'agent' column for stratification.
# This ensures the Test Set has the exact same mix of Agents as the real world.
stratify_col = df_clean['agent']

# Identify Column Types automatically
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()
# Remove bools from nums
num_cols = [c for c in num_cols if c not in bool_cols]

print(f"Categorical Features: {len(cat_cols)}")
print(f"Numerical Features:   {len(num_cols)}")

# SPLIT DATA (Stratified by Agent ONLY)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=stratify_col  # <--- CHANGED: Stratifies by Agent only
)

# Extract 'test_agents' for later analysis
test_agents = X_test['agent'].copy()

print(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}")
print("\nVerifying Test Set Ratios (Agent):")
print(test_agents.value_counts(normalize=True).head(5))

# =============================================================================
# 2. DEFINE ROBUST PIPELINES
# =============================================================================
print("\n" + "="*70)
print("DEFINING MODEL PIPELINES")
print("="*70)

# Preprocessor for Tree Models (Random Forest, GBM, DT)
# Use OrdinalEncoder which handles unknown categories (e.g., new repos) safely
tree_preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', 'passthrough', num_cols),
    ('bool', 'passthrough', bool_cols)
], verbose_feature_names_out=False)

# Preprocessor for Linear Models (Logistic Regression)
# Use OneHotEncoder for mathematical validity
linear_preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', max_categories=20, sparse_output=False), cat_cols),
    ('num', StandardScaler(), num_cols),
    ('bool', 'passthrough', bool_cols)
])

# Define Models
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
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('prep', tree_preprocessor),
        ('clf', HistGradientBoostingClassifier(class_weight='balanced', random_state=42))
    ])
}

# =============================================================================
# 3. TRAIN AND EVALUATE
# =============================================================================
results = []

for name, pipe in models.items():
    print(f"\nTraining {name}...")
    pipe.fit(X_train, y_train)
    
    # Predict
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    # Metrics
    res = {
        'Model': name,
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Precision (Rejected)': precision_score(y_test, y_pred, pos_label=0),
        'Recall (Rejected)': recall_score(y_test, y_pred, pos_label=0),
        'F1 (Rejected)': f1_score(y_test, y_pred, pos_label=0),
        'Accuracy': accuracy_score(y_test, y_pred)
    }
    results.append(res)
    print(f"  -> AUC: {res['ROC-AUC']:.4f} | Precision (Rej): {res['Precision (Rejected)']:.4f}")

# Display Comparison
results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
print("\n" + "="*70)
print("FINAL MODEL LEADERBOARD")
print("="*70)
print(results_df.to_string(index=False, float_format='%.4f'))

# =============================================================================
# 4. INTERPRETABLE RULES (From Decision Tree)
# =============================================================================
print("\n" + "="*70)
print("EXTRACTING HUMAN-READABLE RULES")
print("="*70)

# Access the trained Decision Tree
dt_pipe = models['Decision Tree']
dt_model = dt_pipe.named_steps['clf']

# Get feature names from preprocessor
try:
    feature_names = dt_pipe.named_steps['prep'].get_feature_names_out()
except:
    feature_names = cat_cols + num_cols + bool_cols

# Export Rules
tree_rules = export_text(dt_model, feature_names=list(feature_names), max_depth=3)
print(tree_rules)

# =============================================================================
# 5. PER-AGENT PERFORMANCE CHECK
# =============================================================================
print("\n" + "="*70)
print("PER-AGENT PERFORMANCE (Random Forest)")
print("="*70)

rf_pipe = models['Random Forest']
y_pred_rf = rf_pipe.predict(X_test)
y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]

agent_metrics = []
for agent in test_agents.unique():
    mask = (test_agents == agent)
    if mask.sum() > 10: # Only check agents with enough data
        auc = roc_auc_score(y_test[mask], y_proba_rf[mask])
        acc = accuracy_score(y_test[mask], y_pred_rf[mask])
        agent_metrics.append({'Agent': agent, 'ROC-AUC': auc, 'Accuracy': acc, 'Count': mask.sum()})

agent_df = pd.DataFrame(agent_metrics).sort_values('ROC-AUC', ascending=False)
print(agent_df.to_string(index=False, float_format='%.3f'))

# =============================================================================
# 6. SHAP ANALYSIS (The Ambition Trade-off)
# =============================================================================
print("\n" + "="*70)
print("GENERATING SHAP PLOTS")
print("="*70)

# 1. Prepare Data for SHAP
# We must transform the raw X_test using the pipeline's preprocessor first
rf_prep = rf_pipe.named_steps['prep']
rf_clf = rf_pipe.named_steps['clf']

# Sample 1000 points for speed
X_shap_sample = X_test.sample(n=1000, random_state=42)
X_shap_transformed = rf_prep.transform(X_shap_sample)

# 2. Create Explainer
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_shap_transformed)

# Handle Binary Classification output
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# 3. Plot Beeswarm
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values, 
    X_shap_transformed, 
    feature_names=feature_names,
    max_display=15, 
    show=False
)
plt.title("Why are PRs Rejected? (SHAP Feature Impact)")
plt.tight_layout()
plt.savefig('shap_ambition_proof.png', dpi=300, bbox_inches='tight')
print("Saved: shap_ambition_proof.png")

print("\nANALYSIS COMPLETE. Check 'shap_ambition_proof.png' and metrics above.")
