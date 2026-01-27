#results here -> ablation_leave_out_results.csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, 
                             average_precision_score, precision_score, 
                             recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DEFINE FEATURE GROUPS
# =============================================================================
FEATURE_GROUPS = {
    'agent_identity': [
        'agent', 
        'agent_prior_merge_rate_global', 
        'agent_prior_merge_rate_in_repo'
    ],
    'agent_interactions': [
        'agent_x_size_zscore', 
        'agent_x_has_tests', 
        'agent_x_has_issue'
    ],
    'pr_size_complexity': [
        'num_files', 'num_unique_extensions', 'total_changes', 
        'net_change', 'add_del_ratio', 'avg_changes_per_file', 
        'num_commits', 'num_unique_authors', 'num_unique_committers',
        'pr_size_zscore', 'pr_file_count_zscore', 'pr_changes_per_file_zscore'
    ],
    'pr_content_type': [
        'is_test', 'is_doc', 'is_config'
    ],
    'pr_metadata': [
        'has_body', 'body_length', 'title_length', 
        'task_type', 'task_confidence', 
        'num_linked_issues', 'has_linked_issue'
    ],
    'user_reputation': [
        'user_followers', 'user_following', 'follower_to_following_ratio', 
        'account_age_days', 'is_unknown_user'
    ],
    'user_history': [
        'user_prior_pr_count_total', 'user_prior_merge_rate_total', 
        'user_prior_merge_rate_in_repo', 'days_since_last_pr', 
        'is_first_pr_total', 'is_first_pr_in_repo', 
        'user_trust_x_size', 'tests_x_size_zscore'
    ],
    'repo_context': [
        'repo_stars', 'repo_language', 'repo_license', 'stars_to_forks_ratio',
        'repo_prior_pr_count', 'repo_prior_merge_rate', 
        'repo_avg_pr_size', 'repo_std_pr_size', 'repo_avg_file_count', 
        'is_new_repo'
    ],
    'temporal': [
        'day_of_week', 'is_weekend', 'is_holiday_season'
    ]
}

# Combined groups for higher-level analysis
COMBINED_GROUPS = {
    'all_agent': FEATURE_GROUPS['agent_identity'] + FEATURE_GROUPS['agent_interactions'],
    'all_user': FEATURE_GROUPS['user_reputation'] + FEATURE_GROUPS['user_history'],
    'all_pr_quality': FEATURE_GROUPS['pr_size_complexity'] + FEATURE_GROUPS['pr_content_type'] + FEATURE_GROUPS['pr_metadata'],
}

# =============================================================================
# 2. LOAD AND PREPARE DATA (same as your baseline)
# =============================================================================
def load_and_prepare_data(filepath):
    """Load data and prepare features - NO LEAKAGE"""
    df = pd.read_parquet(filepath)
    
    # Drop leaky columns
    leaky_cols = [
        'merged_at', 'closed_at', 'state',
        'id', 'number', 'user_id', 'repo_id',
        'title', 'body', 'html_url', 'repo_url',
        'created_at', 'user_account_created',
        'user', 'month',
    ]
    df_clean = df.drop(columns=leaky_cols, errors='ignore')
    
    # Drop redundant columns
    redundant_cols = ['additions', 'deletions', 'repo_forks']
    df_clean = df_clean.drop(columns=redundant_cols, errors='ignore')
    
    # Separate features and target
    X = df_clean.drop(columns=['is_merged'])
    y = df_clean['is_merged']
    
    return X, y

def encode_features(X, label_encoders=None):
    """Encode categorical and boolean features"""
    X = X.copy()
    
    # Convert booleans to int
    bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)
    
    # Label encode categoricals
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if label_encoders is None:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    else:
        for col in cat_cols:
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unseen categories
                X[col] = X[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    return X, label_encoders

# =============================================================================
# 3. MODEL TRAINING AND EVALUATION
# =============================================================================
def get_model():
    """Return consistent model configuration"""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

def evaluate_model(y_true, y_pred, y_proba):
    """Calculate all metrics for both classes"""
    return {
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
        # Rejected (0) metrics
        'rejected_precision': precision_score(y_true, y_pred, pos_label=0),
        'rejected_recall': recall_score(y_true, y_pred, pos_label=0),
        'rejected_f1': f1_score(y_true, y_pred, pos_label=0),
        # Merged (1) metrics
        'merged_precision': precision_score(y_true, y_pred, pos_label=1),
        'merged_recall': recall_score(y_true, y_pred, pos_label=1),
        'merged_f1': f1_score(y_true, y_pred, pos_label=1),
    }

def run_cv_evaluation(X, y, n_splits=5):
    """Run stratified cross-validation and return metrics"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = get_model()
    
    y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
    y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    
    return evaluate_model(y, y_pred, y_proba)

# =============================================================================
# 4. ABLATION ANALYSIS
# =============================================================================
def run_ablation_analysis(X, y, feature_groups, analysis_type='leave_out'):
    """
    Run ablation analysis.
    
    analysis_type: 
        'leave_out' - remove each group and measure impact
        'only' - use only each group and measure standalone power
    """
    results = []
    all_features = X.columns.tolist()
    
    # Baseline (all features)
    print("Running BASELINE (all features)...")
    baseline_metrics = run_cv_evaluation(X, y)
    baseline_metrics['experiment'] = 'BASELINE'
    baseline_metrics['features_used'] = len(all_features)
    baseline_metrics['features_removed'] = 0
    results.append(baseline_metrics)
    print(f"  ROC-AUC: {baseline_metrics['roc_auc']:.4f}")
    
    # Run each ablation
    for group_name, group_features in feature_groups.items():
        # Filter to features that actually exist in data
        existing_features = [f for f in group_features if f in all_features]
        
        if not existing_features:
            print(f"Skipping {group_name}: no features found in data")
            continue
        
        if analysis_type == 'leave_out':
            # Remove this group
            features_to_use = [f for f in all_features if f not in existing_features]
            experiment_name = f"No {group_name}"
        else:  # 'only'
            # Use only this group
            features_to_use = existing_features
            experiment_name = f"Only {group_name}"
        
        print(f"Running {experiment_name} ({len(existing_features)} features affected)...")
        
        X_subset = X[features_to_use]
        metrics = run_cv_evaluation(X_subset, y)
        
        metrics['experiment'] = experiment_name
        metrics['features_used'] = len(features_to_use)
        metrics['features_removed'] = len(existing_features) if analysis_type == 'leave_out' else len(all_features) - len(existing_features)
        metrics['delta_roc_auc'] = metrics['roc_auc'] - baseline_metrics['roc_auc']
        metrics['delta_rejected_f1'] = metrics['rejected_f1'] - baseline_metrics['rejected_f1']
        metrics['delta_merged_f1'] = metrics['merged_f1'] - baseline_metrics['merged_f1']
        
        results.append(metrics)
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f} (Δ {metrics['delta_roc_auc']:+.4f})")
    
    return pd.DataFrame(results)

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Load data
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    X, y = load_and_prepare_data('/home/niruthi/ai_code/data/engineered_features.parquet')
    X, label_encoders = encode_features(X)
    print(f"Data shape: {X.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")
    
    # Validate feature groups
    print("\n" + "="*70)
    print("VALIDATING FEATURE GROUPS")
    print("="*70)
    all_grouped_features = []
    for name, features in FEATURE_GROUPS.items():
        existing = [f for f in features if f in X.columns]
        missing = [f for f in features if f not in X.columns]
        all_grouped_features.extend(existing)
        print(f"{name}: {len(existing)}/{len(features)} features found")
        if missing:
            print(f"  Missing: {missing}")
    
    ungrouped = [f for f in X.columns if f not in all_grouped_features]
    if ungrouped:
        print(f"\nWARNING - Ungrouped features: {ungrouped}")
    
    # Run Leave-One-Group-Out Analysis
    print("\n" + "="*70)
    print("LEAVE-ONE-GROUP-OUT ANALYSIS")
    print("="*70)
    leave_out_results = run_ablation_analysis(X, y, FEATURE_GROUPS, 'leave_out')
    
    # Run with combined groups too
    print("\n" + "="*70)
    print("COMBINED GROUPS ANALYSIS")
    print("="*70)
    combined_results = run_ablation_analysis(X, y, COMBINED_GROUPS, 'leave_out')
    # Remove duplicate baseline
    combined_results = combined_results[combined_results['experiment'] != 'BASELINE']
    
    # Merge results
    all_leave_out = pd.concat([leave_out_results, combined_results], ignore_index=True)
    
    # Run Only-This-Group Analysis
    print("\n" + "="*70)
    print("ONLY-THIS-GROUP ANALYSIS (Standalone Predictive Power)")
    print("="*70)
    only_results = run_ablation_analysis(X, y, FEATURE_GROUPS, 'only')
    # Remove baseline from this one
    only_results = only_results[only_results['experiment'] != 'BASELINE']
    
    # =============================================================================
    # 6. DISPLAY RESULTS
    # =============================================================================
    print("\n" + "="*70)
    print("LEAVE-ONE-GROUP-OUT RESULTS")
    print("="*70)
    display_cols = ['experiment', 'features_removed', 'roc_auc', 'delta_roc_auc', 
                    'rejected_f1', 'delta_rejected_f1', 'merged_f1', 'delta_merged_f1']
    print(all_leave_out[display_cols].to_string(index=False, float_format='%.4f'))
    
    print("\n" + "="*70)
    print("ONLY-THIS-GROUP RESULTS (Standalone Power)")
    print("="*70)
    only_display_cols = ['experiment', 'features_used', 'roc_auc', 
                         'rejected_f1', 'merged_f1']
    print(only_results[only_display_cols].to_string(index=False, float_format='%.4f'))
    
    # =============================================================================
    # 7. KEY INSIGHTS
    # =============================================================================
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Most impactful group (biggest drop when removed)
    leave_out_only = all_leave_out[all_leave_out['experiment'] != 'BASELINE']
    most_impactful = leave_out_only.loc[leave_out_only['delta_roc_auc'].idxmin()]
    print(f"\nMost impactful group (biggest drop when removed):")
    print(f"  {most_impactful['experiment']}: Δ ROC-AUC = {most_impactful['delta_roc_auc']:.4f}")
    
    # Least impactful group
    least_impactful = leave_out_only.loc[leave_out_only['delta_roc_auc'].idxmax()]
    print(f"\nLeast impactful group (smallest drop / improvement when removed):")
    print(f"  {least_impactful['experiment']}: Δ ROC-AUC = {least_impactful['delta_roc_auc']:.4f}")
    
    # Best standalone predictor
    best_standalone = only_results.loc[only_results['roc_auc'].idxmax()]
    print(f"\nBest standalone predictor group:")
    print(f"  {best_standalone['experiment']}: ROC-AUC = {best_standalone['roc_auc']:.4f}")
    
    # =============================================================================
    # 8. SAVE RESULTS
    # =============================================================================
    all_leave_out.to_csv('ablation_leave_out_results.csv', index=False)
    only_results.to_csv('ablation_only_results.csv', index=False)
    print("\nResults saved to:")
    print("  - ablation_leave_out_results.csv")
    print("  - ablation_only_results.csv")
