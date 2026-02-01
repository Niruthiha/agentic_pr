# ablation_with_stats.py
# Results saved to: ablation_statistical_results.csv

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_score, recall_score, f1_score)
from scipy.stats import wilcoxon, ttest_rel
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. FEATURE GROUPS (same as before)
# =============================================================================
FEATURE_GROUPS = {
    'agent_identity': [
        'agent', 'agent_prior_merge_rate_global', 'agent_prior_merge_rate_in_repo'
    ],
    'agent_interactions': [
        'agent_x_size_zscore', 'agent_x_has_tests', 'agent_x_has_issue'
    ],
    'pr_size_complexity': [
        'num_files', 'num_unique_extensions', 'total_changes', 
        'net_change', 'add_del_ratio', 'avg_changes_per_file', 
        'num_commits', 'num_unique_authors', 'num_unique_committers',
        'pr_size_zscore', 'pr_file_count_zscore', 'pr_changes_per_file_zscore'
    ],
    'pr_content_type': ['is_test', 'is_doc', 'is_config'],
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
    'temporal': ['day_of_week', 'is_weekend', 'is_holiday_season']
}

COMBINED_GROUPS = {
    'all_agent': FEATURE_GROUPS['agent_identity'] + FEATURE_GROUPS['agent_interactions'],
    'all_user': FEATURE_GROUPS['user_reputation'] + FEATURE_GROUPS['user_history'],
    'all_pr_quality': (FEATURE_GROUPS['pr_size_complexity'] + 
                       FEATURE_GROUPS['pr_content_type'] + 
                       FEATURE_GROUPS['pr_metadata']),
}

# =============================================================================
# 2. DATA LOADING (same as before)
# =============================================================================
def load_and_prepare_data(filepath):
    df = pd.read_parquet(filepath)
    leaky_cols = ['merged_at', 'closed_at', 'state', 'id', 'number', 'user_id', 
                  'repo_id', 'title', 'body', 'html_url', 'repo_url',
                  'created_at', 'user_account_created', 'user', 'month']
    df_clean = df.drop(columns=leaky_cols, errors='ignore')
    redundant_cols = ['additions', 'deletions', 'repo_forks']
    df_clean = df_clean.drop(columns=redundant_cols, errors='ignore')
    X = df_clean.drop(columns=['is_merged'])
    y = df_clean['is_merged']
    return X, y

def check_and_fix_infinities(X):
    """Check for inf values (common in interaction terms with zero denominators)"""
    inf_cols = []
    for col in X.select_dtypes(include=[np.number]).columns:
        n_inf = np.isinf(X[col]).sum()
        n_nan = X[col].isna().sum()
        if n_inf > 0 or n_nan > 0:
            inf_cols.append((col, n_inf, n_nan))
    
    if inf_cols:
        print("\n⚠️  WARNING: Found inf/NaN values (likely from division by zero):")
        for col, n_inf, n_nan in inf_cols:
            print(f"   {col}: {n_inf} inf, {n_nan} NaN")
        
        # Fix: replace inf with large finite values, NaN with median
        X = X.copy()
        for col in X.select_dtypes(include=[np.number]).columns:
            if np.isinf(X[col]).any():
                max_finite = X.loc[np.isfinite(X[col]), col].max()
                min_finite = X.loc[np.isfinite(X[col]), col].min()
                X[col] = X[col].replace([np.inf], max_finite * 2)
                X[col] = X[col].replace([-np.inf], min_finite * 2)
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        print("   → Fixed by capping inf and imputing NaN with median\n")
    else:
        print("✓ No inf/NaN values found in numeric features\n")
    
    return X

def encode_features(X, label_encoders=None):
    X = X.copy()
    bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)
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
                X[col] = X[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    return X, label_encoders

def get_model():
    return RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=10,
        min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)

# =============================================================================
# 3. PER-FOLD EVALUATION (NEW - for statistical testing)
# =============================================================================
def run_cv_with_fold_metrics(X, y, n_splits=10):
    """Run CV and return metrics for EACH fold (not just aggregated)
    
    NOTE: n_splits=10 required for Wilcoxon to achieve p < 0.05
    (With n=5, minimum possible p-value is 0.0625)
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        model = get_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        y_test = y.iloc[test_idx]
        y_pred = model.predict(X.iloc[test_idx])
        y_proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        
        fold_metrics.append({
            'fold': fold,
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
            'rejected_f1': f1_score(y_test, y_pred, pos_label=0),
            'merged_f1': f1_score(y_test, y_pred, pos_label=1),
        })
    
    return pd.DataFrame(fold_metrics)

# =============================================================================
# 4. STATISTICAL TESTS
# =============================================================================
def compute_statistical_tests(baseline_folds, ablation_folds, metric='roc_auc'):
    """
    Compare baseline vs ablation using paired tests.
    Returns p-values and effect size.
    """
    baseline = baseline_folds[metric].values
    ablation = ablation_folds[metric].values
    
    # Mean difference
    mean_diff = ablation.mean() - baseline.mean()
    
    # Paired t-test
    t_stat, t_pval = ttest_rel(baseline, ablation)
    
    # Wilcoxon signed-rank test (non-parametric, better for small n)
    # Handle case where differences might be zero
    differences = ablation - baseline
    if np.all(differences == 0):
        w_stat, w_pval = np.nan, 1.0
    else:
        try:
            w_stat, w_pval = wilcoxon(baseline, ablation)
        except ValueError:
            w_stat, w_pval = np.nan, np.nan
    
    # Cohen's d effect size
    pooled_std = np.sqrt((baseline.std()**2 + ablation.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return {
        'mean_diff': mean_diff,
        'std_diff': differences.std(),
        't_statistic': t_stat,
        't_pvalue': t_pval,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_pvalue': w_pval,
        'cohens_d': cohens_d,
    }

def interpret_significance(p_value):
    """Return significance stars"""
    if pd.isna(p_value):
        return ''
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    elif p_value < 0.1:
        return '†'
    return ''

def interpret_effect_size(d):
    """Interpret Cohen's d"""
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    return 'large'

# =============================================================================
# 5. ABLATION WITH STATISTICS
# =============================================================================
def run_ablation_with_stats(X, y, feature_groups, analysis_type='leave_out'):
    """Run ablation with per-fold metrics and statistical tests"""
    results = []
    all_features = X.columns.tolist()
    
    # Baseline
    print("Running BASELINE...")
    baseline_folds = run_cv_with_fold_metrics(X, y)
    baseline_summary = {
        'experiment': 'BASELINE',
        'features_used': len(all_features),
        'features_removed': 0,
        'roc_auc_mean': baseline_folds['roc_auc'].mean(),
        'roc_auc_std': baseline_folds['roc_auc'].std(),
        'rejected_f1_mean': baseline_folds['rejected_f1'].mean(),
        'merged_f1_mean': baseline_folds['merged_f1'].mean(),
    }
    results.append(baseline_summary)
    print(f"  ROC-AUC: {baseline_summary['roc_auc_mean']:.4f} ± {baseline_summary['roc_auc_std']:.4f}")
    
    # Each ablation
    for group_name, group_features in feature_groups.items():
        existing_features = [f for f in group_features if f in all_features]
        if not existing_features:
            continue
        
        if analysis_type == 'leave_out':
            features_to_use = [f for f in all_features if f not in existing_features]
            experiment_name = f"No {group_name}"
        else:
            features_to_use = existing_features
            experiment_name = f"Only {group_name}"
        
        print(f"Running {experiment_name}...")
        
        X_subset = X[features_to_use]
        ablation_folds = run_cv_with_fold_metrics(X_subset, y)
        
        # Statistical tests
        stats_roc = compute_statistical_tests(baseline_folds, ablation_folds, 'roc_auc')
        stats_rej = compute_statistical_tests(baseline_folds, ablation_folds, 'rejected_f1')
        stats_mer = compute_statistical_tests(baseline_folds, ablation_folds, 'merged_f1')
        
        result = {
            'experiment': experiment_name,
            'features_used': len(features_to_use),
            'features_removed': len(existing_features) if analysis_type == 'leave_out' else len(all_features) - len(existing_features),
            # ROC-AUC
            'roc_auc_mean': ablation_folds['roc_auc'].mean(),
            'roc_auc_std': ablation_folds['roc_auc'].std(),
            'delta_roc_auc': stats_roc['mean_diff'],
            'roc_auc_t_pval': stats_roc['t_pvalue'],
            'roc_auc_wilcox_pval': stats_roc['wilcoxon_pvalue'],
            'roc_auc_cohens_d': stats_roc['cohens_d'],
            # Rejected F1
            'rejected_f1_mean': ablation_folds['rejected_f1'].mean(),
            'delta_rejected_f1': stats_rej['mean_diff'],
            'rejected_f1_wilcox_pval': stats_rej['wilcoxon_pvalue'],
            # Merged F1
            'merged_f1_mean': ablation_folds['merged_f1'].mean(),
            'delta_merged_f1': stats_mer['mean_diff'],
            'merged_f1_wilcox_pval': stats_mer['wilcoxon_pvalue'],
        }
        results.append(result)
        
        sig = interpret_significance(stats_roc['wilcoxon_pvalue'])
        effect = interpret_effect_size(stats_roc['cohens_d'])
        print(f"  ROC-AUC: {result['roc_auc_mean']:.4f} ± {result['roc_auc_std']:.4f} "
              f"(Δ {result['delta_roc_auc']:+.4f}) p={stats_roc['wilcoxon_pvalue']:.4f}{sig} [{effect}]")
    
    return pd.DataFrame(results)

# =============================================================================
# 6. MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    X, y = load_and_prepare_data('/home/niruthi/ai_code/data/engineered_features.parquet')
    X, label_encoders = encode_features(X)
    
    # Check for inf/NaN in interaction terms
    print("\nChecking for inf/NaN values...")
    X = check_and_fix_infinities(X)
    
    print(f"Data shape: {X.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")
    
    # Run ablation with stats
    print("\n" + "="*70)
    print("LEAVE-ONE-GROUP-OUT WITH STATISTICAL TESTS (10-fold CV)")
    print("="*70)
    results_individual = run_ablation_with_stats(X, y, FEATURE_GROUPS, 'leave_out')
    
    print("\n" + "="*70)
    print("COMBINED GROUPS WITH STATISTICAL TESTS")
    print("="*70)
    results_combined = run_ablation_with_stats(X, y, COMBINED_GROUPS, 'leave_out')
    results_combined = results_combined[results_combined['experiment'] != 'BASELINE']
    
    all_results = pd.concat([results_individual, results_combined], ignore_index=True)
    
    # =============================================================================
    # 7. FORMATTED OUTPUT
    # =============================================================================
    print("\n" + "="*70)
    print("SUMMARY TABLE WITH SIGNIFICANCE")
    print("="*70)
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.1")
    print("Effect size: |d|<0.2 negligible, <0.5 small, <0.8 medium, ≥0.8 large\n")
    
    summary = all_results.copy()
    summary['sig'] = summary['roc_auc_wilcox_pval'].apply(interpret_significance)
    summary['effect'] = summary['roc_auc_cohens_d'].apply(interpret_effect_size)
    
    display_cols = ['experiment', 'delta_roc_auc', 'roc_auc_std', 
                    'roc_auc_wilcox_pval', 'sig', 'roc_auc_cohens_d', 'effect']
    print(summary[summary['experiment'] != 'BASELINE'][display_cols].to_string(index=False, float_format='%.4f'))
    
    # =============================================================================
    # 8. KEY FINDINGS
    # =============================================================================
    print("\n" + "="*70)
    print("STATISTICALLY SIGNIFICANT FINDINGS (p < 0.05)")
    print("="*70)
    
    sig_results = summary[(summary['experiment'] != 'BASELINE') & 
                          (summary['roc_auc_wilcox_pval'] < 0.05)]
    
    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            direction = "hurts" if row['delta_roc_auc'] < 0 else "helps"
            print(f"• {row['experiment']}: Δ={row['delta_roc_auc']:+.4f}, "
                  f"p={row['roc_auc_wilcox_pval']:.4f}, d={row['roc_auc_cohens_d']:.2f} ({row['effect']})")
            print(f"  → Removing this group {direction} performance significantly")
    else:
        print("No significant differences found at p < 0.05")
        print("\nMarginal effects (p < 0.1):")
        marginal = summary[(summary['experiment'] != 'BASELINE') & 
                          (summary['roc_auc_wilcox_pval'] < 0.1)]
        for _, row in marginal.iterrows():
            print(f"• {row['experiment']}: Δ={row['delta_roc_auc']:+.4f}, p={row['roc_auc_wilcox_pval']:.4f}")
    
    # =============================================================================
    # 9. SAVE
    # =============================================================================
    all_results.to_csv('ablation_statistical_results.csv', index=False)
    print("\n" + "="*70)
    print("Results saved to: ablation_statistical_results.csv")
    print("="*70)
