"""
RANDOM FOREST - CATCHING MORE REJECTIONS
=========================================
Techniques tested (one at a time):
1. Baseline (default threshold 0.5)
2. Threshold Tuning (0.3, 0.4, 0.45)
3. Class Weight Adjustments
4. SMOTE Oversampling
5. RandomizedSearchCV Hyperparameter Tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, accuracy_score,
                             f1_score, precision_score, recall_score, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_parquet('/home/niruthi/ai_code/data/engineered_features.parquet')

# Drop leaky & redundant columns
leaky_cols = ['merged_at', 'closed_at', 'state', 'id', 'number', 'user_id', 'repo_id',
              'title', 'body', 'html_url', 'repo_url', 'created_at', 'user_account_created',
              'user', 'month']
redundant_cols = ['additions', 'deletions', 'repo_forks']
df_clean = df.drop(columns=leaky_cols + redundant_cols, errors='ignore')

X = df_clean.drop(columns=['is_merged'])
y = df_clean['is_merged']

# Column types
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
num_cols = [c for c in X.select_dtypes(include=['number']).columns.tolist() if c not in bool_cols]

# Stratified split by agent
stratify_col = df_clean['agent']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_col
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Rejected in test: {(y_test==0).sum()} ({(y_test==0).mean()*100:.1f}%)")

# =============================================================================
# 2. PREPROCESSOR
# =============================================================================
preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', 'passthrough', num_cols),
    ('bool', 'passthrough', bool_cols)
], verbose_feature_names_out=False)

# =============================================================================
# 3. HELPER FUNCTION
# =============================================================================
def evaluate_model(y_true, y_pred, y_proba, technique_name):
    """Return dict of metrics for both classes"""
    return {
        'Technique': technique_name,
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        # Merged (1)
        'Precision_Merged': precision_score(y_true, y_pred, pos_label=1),
        'Recall_Merged': recall_score(y_true, y_pred, pos_label=1),
        'F1_Merged': f1_score(y_true, y_pred, pos_label=1),
        # Rejected (0)
        'Precision_Rejected': precision_score(y_true, y_pred, pos_label=0),
        'Recall_Rejected': recall_score(y_true, y_pred, pos_label=0),
        'F1_Rejected': f1_score(y_true, y_pred, pos_label=0),
    }

results = []

# =============================================================================
# 4. TECHNIQUE 1: BASELINE (threshold=0.5, class_weight='balanced')
# =============================================================================
print("\n" + "="*70)
print("TECHNIQUE 1: BASELINE")
print("="*70)

baseline_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=15, 
                                   class_weight='balanced', random_state=42, n_jobs=-1))
])
baseline_pipe.fit(X_train, y_train)
y_proba_base = baseline_pipe.predict_proba(X_test)[:, 1]
y_pred_base = (y_proba_base >= 0.5).astype(int)

results.append(evaluate_model(y_test, y_pred_base, y_proba_base, 'Baseline (threshold=0.5)'))
print(f"Rejected F1: {results[-1]['F1_Rejected']:.4f} | Recall: {results[-1]['Recall_Rejected']:.4f}")

# =============================================================================
# 5. TECHNIQUE 2: THRESHOLD TUNING
# =============================================================================
print("\n" + "="*70)
print("TECHNIQUE 2: THRESHOLD TUNING")
print("="*70)

for thresh in [0.3, 0.35, 0.4, 0.45, 0.55, 0.6]:
    y_pred_thresh = (y_proba_base >= thresh).astype(int)
    res = evaluate_model(y_test, y_pred_thresh, y_proba_base, f'Threshold={thresh}')
    results.append(res)
    print(f"Threshold {thresh}: Rejected F1={res['F1_Rejected']:.4f}, Recall={res['Recall_Rejected']:.4f}, Precision={res['Precision_Rejected']:.4f}")

# =============================================================================
# 6. TECHNIQUE 3: CLASS WEIGHT ADJUSTMENTS
# =============================================================================
print("\n" + "="*70)
print("TECHNIQUE 3: CLASS WEIGHT ADJUSTMENTS")
print("="*70)

# Higher weight on Rejected (class 0)
weight_configs = [
    {0: 2, 1: 1},
    {0: 3, 1: 1},
    {0: 4, 1: 1},
    {0: 5, 1: 1},
]

for weights in weight_configs:
    pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=15,
                                       class_weight=weights, random_state=42, n_jobs=-1))
    ])
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)
    
    res = evaluate_model(y_test, y_pred, y_proba, f'Class Weight {weights}')
    results.append(res)
    print(f"Weight {weights}: Rejected F1={res['F1_Rejected']:.4f}, Recall={res['Recall_Rejected']:.4f}")

# =============================================================================
# 7. TECHNIQUE 4: SMOTE OVERSAMPLING
# =============================================================================
print("\n" + "="*70)
print("TECHNIQUE 4: SMOTE OVERSAMPLING")
print("="*70)

# Need to preprocess first for SMOTE
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

for sampling_strategy in [0.5, 0.75, 1.0]:
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_prep, y_train)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, 
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_resampled, y_resampled)
    
    y_proba = rf.predict_proba(X_test_prep)[:, 1]
    y_pred = rf.predict(X_test_prep)
    
    res = evaluate_model(y_test, y_pred, y_proba, f'SMOTE (ratio={sampling_strategy})')
    results.append(res)
    print(f"SMOTE {sampling_strategy}: Rejected F1={res['F1_Rejected']:.4f}, Recall={res['Recall_Rejected']:.4f}")

# =============================================================================
# 8. TECHNIQUE 5: ROBUST RANDOMIZED SEARCH (Corrected)
# =============================================================================
print("\n" + "="*70)
print("TECHNIQUE 5: RANDOMIZED SEARCH (With Correct Pipeline)")
print("="*70)

# Define a full pipeline that includes preprocessing and the classifier
# Note: We use imblearn Pipeline just in case you want to add SMOTE later, 
# but it works for standard steps too.
pipeline_for_search = ImbPipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Define parameters targeting the 'clf' step
param_dist = {
    'clf__n_estimators': [100, 200, 300, 400],
    'clf__max_depth': [10, 15, 20, 25, None],
    'clf__min_samples_split': [5, 10, 20],
    'clf__min_samples_leaf': [2, 5, 10],
    'clf__max_features': ['sqrt', 0.3, 0.5],
    'clf__class_weight': ['balanced', {0: 2, 1: 1}, {0: 3, 1: 1}, {0: 4, 1: 1}],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.metrics import make_scorer
f1_rejected = make_scorer(f1_score, pos_label=0)

random_search = RandomizedSearchCV(
    pipeline_for_search, # Pass the PIPELINE, not the model
    param_dist, 
    n_iter=30, # Reduced to 30 to save time given the pipeline overhead
    cv=cv, 
    scoring=f1_rejected,
    random_state=42, 
    n_jobs=-1, 
    verbose=1
)

print("Running RandomizedSearchCV (Pipeline-aware)...")
# Pass RAW X_train, not preprocessed X_train_prep
random_search.fit(X_train, y_train) 

print(f"\nBest Parameters: {random_search.best_params_}")
print(f"Best CV F1 (Rejected): {random_search.best_score_:.4f}")

# Evaluate best estimator
# The best_estimator_ is already a pipeline that will handle raw X_test
best_pipe = random_search.best_estimator_ 
y_proba_best = best_pipe.predict_proba(X_test)[:, 1]
y_pred_best = best_pipe.predict(X_test)

res = evaluate_model(y_test, y_pred_best, y_proba_best, 'RandomSearch Best')
results.append(res)
print(f"Test Set - Rejected F1={res['F1_Rejected']:.4f}, Recall={res['Recall_Rejected']:.4f}")

# Also try threshold tuning on best model
for thresh in [0.35, 0.4, 0.45]:
    y_pred_thresh = (y_proba_best >= thresh).astype(int)
    res = evaluate_model(y_test, y_pred_thresh, y_proba_best, f'RandomSearch + Threshold={thresh}')
    results.append(res)

# =============================================================================
# 9. SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1_Rejected', ascending=False)
results_df.to_csv('rf_rejection_tuning_results.csv', index=False)
print("Saved: rf_rejection_tuning_results.csv")

# Save best hyperparameters
best_params_df = pd.DataFrame([random_search.best_params_])
best_params_df['best_cv_f1_rejected'] = random_search.best_score_
best_params_df.to_csv('rf_best_hyperparameters.csv', index=False)
print("Saved: rf_best_hyperparameters.csv")

# Save full RandomSearch CV results
cv_results_df = pd.DataFrame(random_search.cv_results_)
cv_results_df = cv_results_df.sort_values('rank_test_score')
cv_results_df.to_csv('rf_randomsearch_cv_results.csv', index=False)
print("Saved: rf_randomsearch_cv_results.csv")

# =============================================================================
# 10. DISPLAY LEADERBOARD
# =============================================================================
print("\n" + "="*70)
print("LEADERBOARD (Sorted by Rejected F1)")
print("="*70)
print(results_df.to_string(index=False, float_format='%.4f'))

print("\n" + "="*70)
print("TOP 5 TECHNIQUES FOR CATCHING REJECTIONS")
print("="*70)
top5 = results_df.head(5)[['Technique', 'F1_Rejected', 'Recall_Rejected', 'Precision_Rejected', 'ROC-AUC']]
print(top5.to_string(index=False, float_format='%.4f'))
