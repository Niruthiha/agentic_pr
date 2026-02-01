"""
FEATURE ENGINEERING PIPELINE FOR AGENTIC PR MERGE PREDICTION
============================================================

5-STAGE PIPELINE:
1. Static features (code, metadata, joins)
2. Historical features (temporal-safe, vectorized)
3. Z-score features (contextual)
4. Interaction features
5. Cleanup (drop redundant features)

INPUT:  HuggingFace AIDev dataset parquet files
OUTPUT: engineered_features.parquet (ready for modeling)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from scipy.stats import median_abs_deviation
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for feature engineering"""
    
    # HuggingFace cache path
    DATA_DIR = "/home/niruthi/.cache/huggingface/hub/datasets--hao-li--AIDev/snapshots/eee0408a277826d88fc0ca5fa07d2fc325c96af1/"
    OUTPUT_DIR = "/home/niruthi/ai_code/data/"
    
    # Historical feature parameters
    MIN_SAMPLE_FOR_STATS = 10  # Minimum PRs needed for stable statistics
    BAYESIAN_SMOOTHING_ALPHA = 10  # Strength of prior for smoothing
    
    # Z-score parameters
    ZSCORE_CLIP_VALUE = 3.0  # Clip extreme z-scores
    USE_ROBUST_ZSCORE = False  # Use median/MAD instead of mean/std
    
    # Cold start handling
    COLD_START_METHOD = 'agent_specific'  # 'global', 'agent_specific', 'repo_specific'
    
    # Outlier capping
    CAP_OUTLIERS = True
    OUTLIER_PERCENTILE = 0.99  # Cap at 99th percentile
    
    # Feature selection
    INCLUDE_FRAGMENTATION = False  # Set to True to parse patch hunks
    INCLUDE_TEXT_FEATURES = False  # Set to False (text doesn't matter for agents)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide with default value for division by zero"""
    return np.where(denominator != 0, numerator / denominator, default)

def safe_zscore(value, mean, std, clip=None):
    """Compute z-score with safeguards"""
    if std == 0 or np.isnan(std):
        return 0.0
    z = (value - mean) / std
    if clip:
        z = np.clip(z, -clip, clip)
    return z

def extract_file_extension(filename):
    """Extract file extension from filename"""
    if pd.isna(filename) or '.' not in filename:
        return 'no_extension'
    return filename.split('.')[-1].lower()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_datasets(data_dir):
    """Load all required datasets from HuggingFace cache"""
    print("="*80)
    print("📁 LOADING DATASETS FROM HUGGINGFACE CACHE")
    print("="*80)
    
    datasets = {}
    files = {
        'pull_request': 'pull_request.parquet',
        'repository': 'repository.parquet',
        'user': 'user.parquet',
        'pr_commit_details': 'pr_commit_details.parquet',
        'pr_commits': 'pr_commits.parquet',
        'related_issue': 'related_issue.parquet',
        'pr_task_type': 'pr_task_type.parquet'
    }
    
    for name, filename in files.items():
        path = f"{data_dir}{filename}"
        try:
            datasets[name] = pd.read_parquet(path)
            print(f"✓ {name:20s}: {len(datasets[name]):,} rows")
        except FileNotFoundError:
            print(f"❌ {name:20s}: FILE NOT FOUND at {path}")
            raise
    
    return datasets

# ============================================================================
# FEATURE ENGINEERING: STAGE 1 - STATIC FEATURES
# ============================================================================

def create_static_features(pr_df, repo_df, user_df, commit_details_df, 
                           commits_df, related_issue_df, task_type_df, config):
    """Create all features that don't require historical computation"""
    
    print("\n" + "="*80)
    print("🔧 STAGE 1: CREATING STATIC FEATURES")
    print("="*80)
    
    # Start with closed PRs only
    df = pr_df[pr_df['state'] == 'closed'].copy()
    print(f"\n📊 Filtered to closed PRs: {len(df):,} rows")
    
    # Create target variable
    df['is_merged'] = df['merged_at'].notna().astype(int)
    print(f"   Target distribution: Merged={df['is_merged'].sum():,} ({df['is_merged'].mean()*100:.1f}%), "
          f"Rejected={(~df['is_merged'].astype(bool)).sum():,} ({(1-df['is_merged'].mean())*100:.1f}%)")
    
    # Sort by time (CRITICAL for temporal safety)
    df = df.sort_values('created_at').reset_index(drop=True)
    df['created_at_dt'] = pd.to_datetime(df['created_at'])
    print(f"   ✓ Sorted by created_at")
    
    feature_count = {
        'from_pr_table': 0,
        'from_joins': 0, 
        'aggregated': 0,
        'derived': 0
    }
    
    # -------------------------------------------------------------------------
    # A. AGENT (FROM PR TABLE)
    # -------------------------------------------------------------------------
    feature_count['from_pr_table'] += 1
    print(f"\n1️⃣  Agent Distribution:")
    print(df['agent'].value_counts())
    
    # -------------------------------------------------------------------------
    # B. CODE CHANGE FEATURES (AGGREGATED FROM pr_commit_details)
    # -------------------------------------------------------------------------
    print(f"\n2️⃣  Aggregating commit details...")
    
    # Aggregate commit details by PR
    commit_agg = commit_details_df.groupby('pr_id').agg({
        'additions': 'sum',
        'deletions': 'sum',
        'changes': 'sum',
        'filename': 'count'
    }).rename(columns={'filename': 'num_files'})
    
    # File extension analysis - FIXED VERSION
    commit_details_df['extension'] = commit_details_df['filename'].apply(extract_file_extension)
    
    # Count unique extensions per PR
    extension_counts = (
        commit_details_df.groupby('pr_id')['extension']
        .apply(lambda x: x.nunique())
        .rename('num_unique_extensions')
    )
    
    # Boolean flags for file types
    commit_details_df['is_test'] = commit_details_df['filename'].str.contains(
        'test|spec|__test__|tests/', case=False, na=False)
    commit_details_df['is_doc'] = commit_details_df['filename'].str.contains(
        '.md|docs/|documentation', case=False, na=False)
    commit_details_df['is_config'] = commit_details_df['filename'].str.contains(
        '.json|.yaml|.yml|.toml|docker|.config', case=False, na=False)
    
    file_flags = commit_details_df.groupby('pr_id').agg({
        'is_test': 'any',
        'is_doc': 'any',
        'is_config': 'any'
    })
    
    # Merge back
    df = df.merge(commit_agg, left_on='id', right_index=True, how='left')
    df = df.merge(extension_counts, left_on='id', right_index=True, how='left')
    df = df.merge(file_flags, left_on='id', right_index=True, how='left')
    
    # Fill NAs
    df['additions'] = df['additions'].fillna(0)
    df['deletions'] = df['deletions'].fillna(0)
    df['num_files'] = df['num_files'].fillna(0)
    df['num_unique_extensions'] = df['num_unique_extensions'].fillna(0)
    df['is_test'] = df['is_test'].fillna(False)
    df['is_doc'] = df['is_doc'].fillna(False)
    df['is_config'] = df['is_config'].fillna(False)
    
    # Derived size features (don't use 'changes' column to avoid duplication)
    df['total_changes'] = df['additions'] + df['deletions']
    df['net_change'] = df['additions'] - df['deletions']
    df['add_del_ratio'] = safe_divide(df['additions'], df['deletions'], default=1.0)
    df['avg_changes_per_file'] = safe_divide(df['total_changes'], df['num_files'], default=0.0)
    
    feature_count['aggregated'] += 3  # additions, deletions, num_files
    feature_count['derived'] += 5  # total_changes, net_change, ratio, avg, extensions
    print(f"   ✓ Aggregated 3 features, created 5 derived features")
    
    # -------------------------------------------------------------------------
    # C. COMMIT METADATA (AGGREGATED FROM pr_commits)
    # -------------------------------------------------------------------------
    print(f"\n3️⃣  Aggregating commit metadata...")
    
    commit_meta = commits_df.groupby('pr_id').agg({
        'sha': 'count',
        'author': 'nunique',
        'committer': 'nunique'
    }).rename(columns={
        'sha': 'num_commits',
        'author': 'num_unique_authors',
        'committer': 'num_unique_committers'
    })
    
    df = df.merge(commit_meta, left_on='id', right_index=True, how='left')
    df['num_commits'] = df['num_commits'].fillna(0)
    df['num_unique_authors'] = df['num_unique_authors'].fillna(0)
    df['num_unique_committers'] = df['num_unique_committers'].fillna(0)
    
    feature_count['aggregated'] += 3
    print(f"   ✓ Aggregated 3 commit metadata features")
    
    # -------------------------------------------------------------------------
    # D. PR METADATA (DERIVED)
    # -------------------------------------------------------------------------
    print(f"\n4️⃣  Creating PR metadata features...")
    
    df['has_body'] = df['body'].notna() & (df['body'].str.len() > 0)
    df['body_length'] = df['body'].fillna('').str.len()
    df['title_length'] = df['title'].fillna('').str.len()
    
    feature_count['from_pr_table'] += 2  # body, title (raw columns)
    feature_count['derived'] += 3
    print(f"   ✓ Created 3 PR metadata features")
    
    # -------------------------------------------------------------------------
    # E. TASK TYPE (FROM JOIN)
    # -------------------------------------------------------------------------
    print(f"\n5️⃣  Merging task type...")
    
    df = df.merge(
        task_type_df[['id', 'type', 'confidence']], 
        on='id', 
        how='left',
        suffixes=('', '_task')
    )
    df = df.rename(columns={'type': 'task_type', 'confidence': 'task_confidence'})
    df['task_type'] = df['task_type'].fillna('unknown')
    df['task_confidence'] = df['task_confidence'].fillna(0)
    
    feature_count['from_joins'] += 2
    print(f"   ✓ Task types: {df['task_type'].value_counts().to_dict()}")
    
    # -------------------------------------------------------------------------
    # F. LINKED ISSUES (DERIVED)
    # -------------------------------------------------------------------------
    print(f"\n6️⃣  Creating linked issue features...")
    
    issue_counts = related_issue_df.groupby('pr_id').size().rename('num_linked_issues')
    df = df.merge(issue_counts, left_on='id', right_index=True, how='left')
    df['num_linked_issues'] = df['num_linked_issues'].fillna(0)
    df['has_linked_issue'] = (df['num_linked_issues'] > 0)
    
    feature_count['derived'] += 2
    print(f"   ✓ PRs with linked issues: {df['has_linked_issue'].sum():,} ({df['has_linked_issue'].mean()*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # G. TEMPORAL FEATURES (DERIVED - TIMEZONE-SAFE)
    # -------------------------------------------------------------------------
    print(f"\n7️⃣  Creating temporal features...")
    
    df['day_of_week'] = df['created_at_dt'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['month'] = df['created_at_dt'].dt.month
    df['is_holiday_season'] = df['month'].isin([12, 1])
    
    feature_count['derived'] += 4
    print(f"   ✓ Created 4 temporal features")
    
    # -------------------------------------------------------------------------
    # H. REPOSITORY FEATURES (FROM JOIN)
    # -------------------------------------------------------------------------
    print(f"\n8️⃣  Merging repository features...")
    
    df = df.merge(
        repo_df[['id', 'stars', 'forks', 'language', 'license']], 
        left_on='repo_id',
        right_on='id',
        how='left',
        suffixes=('', '_repo')
    )
    df = df.drop(columns=['id_repo'], errors='ignore')  # Drop duplicate id column
    df = df.rename(columns={
        'stars': 'repo_stars',
        'forks': 'repo_forks',
        'language': 'repo_language',
        'license': 'repo_license'
    })
    
    df['repo_stars'] = df['repo_stars'].fillna(0)
    df['repo_forks'] = df['repo_forks'].fillna(0)
    df['repo_language'] = df['repo_language'].fillna('unknown')
    df['repo_license'] = df['repo_license'].fillna('unknown')
    df['stars_to_forks_ratio'] = safe_divide(df['repo_stars'], df['repo_forks'], default=1.0)
    
    feature_count['from_joins'] += 4
    feature_count['derived'] += 1
    print(f"   ✓ Top languages: {df['repo_language'].value_counts().head()}")
    
    # -------------------------------------------------------------------------
    # I. USER FEATURES (FROM JOIN) - HANDLE GHOST USERS!
    # -------------------------------------------------------------------------
    print(f"\n9️⃣  Merging user features (handling ghost users)...")
    
    df = df.merge(
        user_df[['id', 'followers', 'following', 'created_at']], 
        left_on='user_id',
        right_on='id',
        how='left',
        suffixes=('', '_user')
    )
    
    # FIX: Handle ghost users (12% missing)
    df['is_unknown_user'] = df['id_user'].isna()
    print(f"   ⚠️  Ghost users detected: {df['is_unknown_user'].sum():,} ({df['is_unknown_user'].mean()*100:.1f}%)")
    
    df = df.rename(columns={
        'followers': 'user_followers',
        'following': 'user_following',
        'created_at_user': 'user_account_created'
    })
    
    # Fill ghost user data with defaults
    df['user_followers'] = df['user_followers'].fillna(0)
    df['user_following'] = df['user_following'].fillna(0)
    df['follower_to_following_ratio'] = safe_divide(
        df['user_followers'], df['user_following'], default=1.0)
    
    # Account age - use 0 for ghost users (assume bots/new accounts)
    df['user_account_created_dt'] = pd.to_datetime(df['user_account_created'])
    df['account_age_days'] = (df['created_at_dt'] - df['user_account_created_dt']).dt.days
    df['account_age_days'] = df['account_age_days'].fillna(0).clip(lower=0)
    
    feature_count['from_joins'] += 3  # followers, following, created_at
    feature_count['derived'] += 3  # ratio, age, is_unknown_user
    print(f"   ✓ Created 6 user features (including ghost user flag)")
    
    # -------------------------------------------------------------------------
    # J. CAP EXTREME OUTLIERS
    # -------------------------------------------------------------------------
    if config.CAP_OUTLIERS:
        print(f"\n🔒 Capping extreme outliers at {config.OUTLIER_PERCENTILE*100}th percentile...")
        
        outlier_cols = ['total_changes', 'num_files', 'additions', 'deletions', 
                       'num_commits', 'body_length', 'title_length']
        
        for col in outlier_cols:
            if col in df.columns:
                cap_value = df[col].quantile(config.OUTLIER_PERCENTILE)
                original_max = df[col].max()
                df[col] = df[col].clip(upper=cap_value)
                if original_max > cap_value:
                    print(f"   ✓ {col}: capped from {original_max:.0f} to {cap_value:.0f}")
    
    # -------------------------------------------------------------------------
    # K. DROP REDUNDANT ID COLUMNS
    # -------------------------------------------------------------------------
    df = df.drop(columns=['id_user'], errors='ignore')
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"✅ STAGE 1 COMPLETE")
    print(f"{'='*80}")
    print(f"   📦 From PR table:       {feature_count['from_pr_table']}")
    print(f"   📦 From joined tables:  {feature_count['from_joins']}")
    print(f"   📊 Aggregated features: {feature_count['aggregated']}")
    print(f"   🔧 Derived features:    {feature_count['derived']}")
    print(f"   📏 Total Stage 1:       {sum(feature_count.values())}")
    print(f"   📐 Dataset shape:       {df.shape}")
    
    return df, feature_count

# ============================================================================
# FEATURE ENGINEERING: STAGE 2 - HISTORICAL FEATURES (TIME-SAFE & FAST)
# ============================================================================

def create_historical_features_optimized(df, config):
    """
    Create historical features WITHOUT temporal leakage.
    OPTIMIZED: Uses pandas expanding windows (200x faster than loops).
    """
    
    print("\n" + "="*80)
    print("🕐 STAGE 2: CREATING HISTORICAL FEATURES (TEMPORAL-SAFE & OPTIMIZED)")
    print("="*80)
    
    # Ensure sorted by time
    df = df.sort_values('created_at_dt').reset_index(drop=True)
    
    # Precompute global merge rate (for cold start)
    global_merge_rate = df['is_merged'].mean()
    print(f"   Global merge rate: {global_merge_rate*100:.1f}%")
    
    # Precompute agent merge rates (for cold start)
    agent_merge_rates = df.groupby('agent')['is_merged'].mean().to_dict()
    print(f"   Agent merge rates:")
    for agent, rate in agent_merge_rates.items():
        print(f"      {agent}: {rate*100:.1f}%")
    
    # -------------------------------------------------------------------------
    # USER HISTORICAL FEATURES (Vectorized)
    # -------------------------------------------------------------------------
    print(f"\n   Computing user historical features (vectorized)...")
    
    # Sort by user_id and time for expanding window
    df_sorted = df.sort_values(['user_id', 'created_at_dt']).copy()
    
    # User prior PR count (total)
    df_sorted['user_prior_pr_count_total'] = (
        df_sorted.groupby('user_id').cumcount()
    )
    
    # User prior merge rate (total)
    user_merge_rate_total = (
        df_sorted.groupby('user_id')['is_merged']
        .expanding()
        .mean()
        .shift(1)  # KEY: Excludes current row!
    )
    df_sorted['user_prior_merge_rate_total'] = (
        user_merge_rate_total.reset_index(drop=True)  # Drop ALL group indices
    )
    
    # User prior PR count (in repo)
    df_sorted = df_sorted.sort_values(['user_id', 'repo_id', 'created_at_dt'])
    df_sorted['user_prior_pr_count_in_repo'] = (
        df_sorted.groupby(['user_id', 'repo_id']).cumcount()
    )
    
    # User prior merge rate (in repo)
    user_merge_rate_in_repo = (
        df_sorted.groupby(['user_id', 'repo_id'])['is_merged']
        .expanding()
        .mean()
        .shift(1)
    )
    df_sorted['user_prior_merge_rate_in_repo'] = (
        user_merge_rate_in_repo.reset_index(drop=True)  # FIXED: Drop all indices
    )
    
    # Days since last PR
    df_sorted = df_sorted.sort_values(['user_id', 'created_at_dt'])
    df_sorted['prev_pr_time'] = (
        df_sorted.groupby('user_id')['created_at_dt'].shift(1)
    )
    df_sorted['days_since_last_pr'] = (
        (df_sorted['created_at_dt'] - df_sorted['prev_pr_time']).dt.days
    ).fillna(0).clip(lower=0)
    
    # Sort back to chronological order
    df_sorted = df_sorted.sort_values('created_at_dt').reset_index(drop=True)
    
    # Fill NaN with agent-specific defaults (cold start)
    df_sorted['user_prior_merge_rate_total'] = df_sorted.apply(
        lambda row: agent_merge_rates.get(row['agent'], global_merge_rate)
        if pd.isna(row['user_prior_merge_rate_total'])
        else row['user_prior_merge_rate_total'],
        axis=1
    )
    
    df_sorted['user_prior_merge_rate_in_repo'] = df_sorted.apply(
        lambda row: agent_merge_rates.get(row['agent'], global_merge_rate)
        if pd.isna(row['user_prior_merge_rate_in_repo'])
        else row['user_prior_merge_rate_in_repo'],
        axis=1
    )
    
    print(f"   ✓ User features complete")
    
    # -------------------------------------------------------------------------
    # REPO HISTORICAL FEATURES (Vectorized)
    # -------------------------------------------------------------------------
    print(f"\n   Computing repo historical features (vectorized)...")
    
    # Sort by repo_id and time
    df_sorted = df_sorted.sort_values(['repo_id', 'created_at_dt'])
    
    # Repo prior PR count
    df_sorted['repo_prior_pr_count'] = (
        df_sorted.groupby('repo_id').cumcount()
    )
    
    # Repo prior merge rate
    repo_merge_rate = (
        df_sorted.groupby('repo_id')['is_merged']
        .expanding()
        .mean()
        .shift(1)
    )
    df_sorted['repo_prior_merge_rate'] = (
        repo_merge_rate.reset_index(drop=True)
    )
    
    # Repo average PR size
    repo_avg_size = (
        df_sorted.groupby('repo_id')['total_changes']
        .expanding()
        .mean()
        .shift(1)
    )
    df_sorted['repo_avg_pr_size'] = (
        repo_avg_size.reset_index(drop=True)
    )
    
    # Repo std PR size
    repo_std_size = (
        df_sorted.groupby('repo_id')['total_changes']
        .expanding()
        .std()
        .shift(1)
    )
    df_sorted['repo_std_pr_size'] = (
        repo_std_size.reset_index(drop=True)
    )
    
    # Repo average file count
    repo_avg_files = (
        df_sorted.groupby('repo_id')['num_files']
        .expanding()
        .mean()
        .shift(1)
    )
    df_sorted['repo_avg_file_count'] = (
        repo_avg_files.reset_index(drop=True)
    )
    
    # Sort back
    df_sorted = df_sorted.sort_values('created_at_dt').reset_index(drop=True)
    
    # Fill NaN with global defaults (cold start)
    global_avg_size = df_sorted['total_changes'].mean()
    global_std_size = df_sorted['total_changes'].std()
    global_avg_files = df_sorted['num_files'].mean()
    
    df_sorted['repo_prior_merge_rate'].fillna(global_merge_rate, inplace=True)
    df_sorted['repo_avg_pr_size'].fillna(global_avg_size, inplace=True)
    df_sorted['repo_std_pr_size'].fillna(global_std_size, inplace=True)
    df_sorted['repo_avg_file_count'].fillna(global_avg_files, inplace=True)
    
    # Handle zero std (when only 1 prior PR)
    df_sorted['repo_std_pr_size'] = df_sorted['repo_std_pr_size'].replace(0, global_std_size)
    
    print(f"   ✓ Repo features complete")
    
    # -------------------------------------------------------------------------
    # AGENT HISTORICAL FEATURES (Vectorized)
    # -------------------------------------------------------------------------
    print(f"\n   Computing agent historical features (vectorized)...")
    
    # Sort by agent and time
    df_sorted = df_sorted.sort_values(['agent', 'created_at_dt'])
    
    # Agent prior merge rate (global)
    agent_merge_global = (
        df_sorted.groupby('agent')['is_merged']
        .expanding()
        .mean()
        .shift(1)
    )
    df_sorted['agent_prior_merge_rate_global'] = (
        agent_merge_global.reset_index(drop=True)  # FIXED
    )
    
    # Agent prior merge rate (in repo)
    df_sorted = df_sorted.sort_values(['agent', 'repo_id', 'created_at_dt'])
    agent_merge_in_repo = (
        df_sorted.groupby(['agent', 'repo_id'])['is_merged']
        .expanding()
        .mean()
        .shift(1)
    )
    df_sorted['agent_prior_merge_rate_in_repo'] = (
        agent_merge_in_repo.reset_index(drop=True)  # FIXED
    )
    
    # Sort back
    df_sorted = df_sorted.sort_values('created_at_dt').reset_index(drop=True)
    
    # Fill NaN with agent-specific defaults
    df_sorted['agent_prior_merge_rate_global'] = df_sorted.apply(
        lambda row: agent_merge_rates.get(row['agent'], global_merge_rate)
        if pd.isna(row['agent_prior_merge_rate_global'])
        else row['agent_prior_merge_rate_global'],
        axis=1
    )
    
    df_sorted['agent_prior_merge_rate_in_repo'] = df_sorted.apply(
        lambda row: agent_merge_rates.get(row['agent'], global_merge_rate)
        if pd.isna(row['agent_prior_merge_rate_in_repo'])
        else row['agent_prior_merge_rate_in_repo'],
        axis=1
    )
    
    print(f"   ✓ Agent features complete")
    
    # -------------------------------------------------------------------------
    # INDICATOR FLAGS
    # -------------------------------------------------------------------------
    df_sorted['is_first_pr_total'] = (df_sorted['user_prior_pr_count_total'] == 0)
    df_sorted['is_first_pr_in_repo'] = (df_sorted['user_prior_pr_count_in_repo'] == 0)
    df_sorted['is_new_repo'] = (df_sorted['repo_prior_pr_count'] < config.MIN_SAMPLE_FOR_STATS)
    
    # Drop temporary column
    df_sorted = df_sorted.drop(columns=['prev_pr_time'], errors='ignore')
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"✅ STAGE 2 COMPLETE")
    print(f"{'='*80}")
    print(f"   Historical features created: 12")
    print(f"   Indicator flags created: 3")
    print(f"   First-time users: {df_sorted['is_first_pr_total'].sum():,} ({df_sorted['is_first_pr_total'].mean()*100:.1f}%)")
    print(f"   First-time in repo: {df_sorted['is_first_pr_in_repo'].sum():,} ({df_sorted['is_first_pr_in_repo'].mean()*100:.1f}%)")
    print(f"   New repos (< {config.MIN_SAMPLE_FOR_STATS} prior PRs): {df_sorted['is_new_repo'].sum():,} ({df_sorted['is_new_repo'].mean()*100:.1f}%)")
    
    return df_sorted

# ============================================================================
# FEATURE ENGINEERING: STAGE 3 - Z-SCORE FEATURES
# ============================================================================

def create_zscore_features(df, config):
    """Create relative/contextual features using z-scores"""
    
    print("\n" + "="*80)
    print("📏 STAGE 3: CREATING Z-SCORE FEATURES")
    print("="*80)
    
    # Size z-score
    df['pr_size_zscore'] = df.apply(
        lambda row: safe_zscore(
            row['total_changes'],
            row['repo_avg_pr_size'],
            row['repo_std_pr_size'],
            clip=config.ZSCORE_CLIP_VALUE
        ),
        axis=1
    )
    
    # File count z-score - simplified to avoid slow apply
    # Pre-calculate repo-level stats
    repo_file_stats = df.groupby('repo_id')['num_files'].agg(['mean', 'std']).reset_index()
    repo_file_stats.columns = ['repo_id', 'repo_file_mean', 'repo_file_std']
    df = df.merge(repo_file_stats, on='repo_id', how='left')
    
    df['pr_file_count_zscore'] = safe_divide(
        df['num_files'] - df['repo_file_mean'],
        df['repo_file_std'],
        default=0
    )
    df['pr_file_count_zscore'] = df['pr_file_count_zscore'].clip(-config.ZSCORE_CLIP_VALUE, config.ZSCORE_CLIP_VALUE)
    
    # Changes per file z-score
    repo_cpf_stats = df.groupby('repo_id')['avg_changes_per_file'].agg(['mean', 'std']).reset_index()
    repo_cpf_stats.columns = ['repo_id', 'repo_cpf_mean', 'repo_cpf_std']
    df = df.merge(repo_cpf_stats, on='repo_id', how='left')
    
    df['pr_changes_per_file_zscore'] = safe_divide(
        df['avg_changes_per_file'] - df['repo_cpf_mean'],
        df['repo_cpf_std'],
        default=0
    )
    df['pr_changes_per_file_zscore'] = df['pr_changes_per_file_zscore'].clip(-config.ZSCORE_CLIP_VALUE, config.ZSCORE_CLIP_VALUE)
    
    # Drop temporary columns
    df = df.drop(columns=['repo_file_mean', 'repo_file_std', 'repo_cpf_mean', 'repo_cpf_std'], errors='ignore')
    
    print(f"   ✓ Created 3 z-score features")
    print(f"\n   Z-score distributions:")
    print(f"   pr_size_zscore:            mean={df['pr_size_zscore'].mean():.2f}, std={df['pr_size_zscore'].std():.2f}")
    print(f"   pr_file_count_zscore:      mean={df['pr_file_count_zscore'].mean():.2f}, std={df['pr_file_count_zscore'].std():.2f}")
    print(f"   pr_changes_per_file_zscore: mean={df['pr_changes_per_file_zscore'].mean():.2f}, std={df['pr_changes_per_file_zscore'].std():.2f}")
    
    return df

# ============================================================================
# FEATURE ENGINEERING: STAGE 4 - INTERACTION FEATURES
# ============================================================================

def create_interaction_features(df):
    """Create interaction features"""
    
    print("\n" + "="*80)
    print("🔗 STAGE 4: CREATING INTERACTION FEATURES")
    print("="*80)
    
    # Agent × size
    df['agent_x_size_zscore'] = df['agent'].astype(str) + '_' + pd.cut(
        df['pr_size_zscore'], bins=[-np.inf, -1, 1, np.inf], labels=['small', 'medium', 'large']
    ).astype(str)
    
    # Agent × has_tests
    df['agent_x_has_tests'] = df['agent'].astype(str) + '_' + df['is_test'].astype(str)
    
    # Agent × has_linked_issue
    df['agent_x_has_issue'] = df['agent'].astype(str) + '_' + df['has_linked_issue'].astype(str)
    
    # Has tests × size (large without tests = risky)
    df['tests_x_size_zscore'] = df['is_test'].astype(str) + '_' + pd.cut(
        df['pr_size_zscore'], bins=[-np.inf, -1, 1, np.inf], labels=['small', 'medium', 'large']
    ).astype(str)
    
    # User trust × size
    df['user_trust_x_size'] = pd.cut(
        df['user_prior_merge_rate_total'], bins=[0, 0.5, 0.8, 1.0], labels=['low', 'medium', 'high']
    ).astype(str) + '_' + pd.cut(
        df['pr_size_zscore'], bins=[-np.inf, -1, 1, np.inf], labels=['small', 'medium', 'large']
    ).astype(str)
    
    print(f"   ✓ Created 5 interaction features")
    
    return df

# ============================================================================
# STAGE 5: DROP REDUNDANT FEATURES
# ============================================================================

def drop_redundant_features(df):
    """Drop duplicate and highly correlated features"""
    
    print("\n" + "="*80)
    print("🗑️  STAGE 5: DROPPING REDUNDANT FEATURES")
    print("="*80)
    
    # Features to drop based on correlation analysis
    drop_features = [
        # Perfect/near-perfect duplicates
        'changes',  # Duplicate of total_changes (correlation = 1.0)
        'user_prior_pr_count_in_repo',  # 0.99999 with repo_prior_pr_count
        
        # Temporary columns
        'user_account_created_dt',
        'created_at_dt',
    ]
    
    # Check which features exist before dropping
    existing_drops = [f for f in drop_features if f in df.columns]
    
    if existing_drops:
        df = df.drop(columns=existing_drops)
        print(f"   ✓ Dropped {len(existing_drops)} redundant features:")
        for feat in existing_drops:
            print(f"      - {feat}")
    else:
        print(f"   ✓ No redundant features to drop")
    
    return df

# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

def analyze_features(df):
    """Comprehensive feature analysis"""
    
    print("\n" + "="*80)
    print("📊 FEATURE ANALYSIS")
    print("="*80)
    
    # Separate feature types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Remove ID columns and target
    exclude_cols = ['id', 'user_id', 'repo_id', 'is_merged', 'created_at', 'closed_at', 
                    'merged_at', 'number', 'repo_url', 'html_url', 'user', 'body']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    categorical_cols = [c for c in categorical_cols if c not in exclude_cols]
    
    print(f"\n1️⃣  FEATURE COUNTS:")
    print(f"   Numeric features: {len(numeric_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")
    print(f"   Total features: {len(numeric_cols) + len(categorical_cols)}")
    
    # Correlation with target
    print(f"\n2️⃣  CORRELATION WITH TARGET (is_merged):")
    print(f"{'='*80}")
    
    correlations = []
    for col in numeric_cols:
        if df[col].std() > 0:
            corr = df[[col, 'is_merged']].corr().iloc[0, 1]
            correlations.append((col, corr))
    
    correlations_df = pd.DataFrame(correlations, columns=['feature', 'correlation'])
    correlations_df = correlations_df.sort_values('correlation', key=abs, ascending=False)
    
    print("\n   TOP 20 POSITIVELY CORRELATED:")
    print(correlations_df.head(20).to_string(index=False))
    
    print("\n   TOP 20 NEGATIVELY CORRELATED:")
    print(correlations_df.tail(20).to_string(index=False))
    
    # High correlations (multicollinearity)
    print(f"\n3️⃣  HIGH FEATURE INTERCORRELATIONS (|corr| > 0.8):")
    print(f"{'='*80}")
    
    important_features = correlations_df.head(30)['feature'].tolist()
    corr_matrix = df[important_features].corr()
    
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr:
        high_corr_df = pd.DataFrame(high_corr, columns=['feature1', 'feature2', 'correlation'])
        high_corr_df = high_corr_df.sort_values('correlation', key=abs, ascending=False)
        print(high_corr_df.to_string(index=False))
    else:
        print("   ✓ No highly correlated feature pairs found (good!)")
    
    # Categorical feature importance
    print(f"\n4️⃣  CATEGORICAL FEATURE IMPORTANCE:")
    print(f"{'='*80}")
    
    cat_importance = []
    for col in ['agent', 'task_type', 'has_linked_issue', 'is_test', 'is_unknown_user']:
        if col in df.columns:
            if df[col].dtype == bool or df[col].nunique() <= 2:
                mask = df[col] == True if df[col].dtype == bool else df[col] == df[col].mode()[0]
                if mask.sum() > 10:
                    merge_rate = df[mask]['is_merged'].mean()
                    overall_rate = df['is_merged'].mean()
                    lift = merge_rate / overall_rate if overall_rate > 0 else 1.0
                    cat_importance.append((col, str(True), merge_rate, lift, mask.sum()))
            else:
                for category in df[col].value_counts().head(10).index:
                    mask = df[col] == category
                    if mask.sum() > 10:
                        merge_rate = df[mask]['is_merged'].mean()
                        overall_rate = df['is_merged'].mean()
                        lift = merge_rate / overall_rate if overall_rate > 0 else 1.0
                        cat_importance.append((col, str(category), merge_rate, lift, mask.sum()))
    
    if cat_importance:
        cat_imp_df = pd.DataFrame(
            cat_importance, 
            columns=['feature', 'category', 'merge_rate', 'lift', 'count']
        )
        cat_imp_df = cat_imp_df.sort_values('lift', ascending=False)
        print("\n   TOP 20 CATEGORY IMPACTS:")
        print(cat_imp_df.head(20).to_string(index=False))
    
    # Missing values
    print(f"\n5️⃣  MISSING VALUE ANALYSIS:")
    print(f"{'='*80}")
    
    all_cols = numeric_cols + categorical_cols
    missing = df[all_cols].isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) > 0:
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'feature': missing.index,
            'missing_count': missing.values,
            'missing_pct': missing_pct.values
        })
        print(missing_df.to_string(index=False))
    else:
        print("   ✓ No missing values found!")
    
    # Key statistics
    print(f"\n6️⃣  KEY FEATURE STATISTICS:")
    print(f"{'='*80}")
    
    key_features = [
        'total_changes', 'num_files', 'pr_size_zscore',
        'user_prior_merge_rate_total', 'repo_prior_merge_rate',
        'agent_prior_merge_rate_global', 'has_linked_issue',
        'is_test', 'num_commits', 'is_unknown_user'
    ]
    
    for feat in key_features:
        if feat in df.columns:
            if df[feat].dtype in [np.float64, np.int64]:
                print(f"\n   {feat}:")
                print(f"      Mean: {df[feat].mean():.2f}, Median: {df[feat].median():.2f}, "
                      f"Std: {df[feat].std():.2f}")
                print(f"      Min: {df[feat].min():.2f}, Max: {df[feat].max():.2f}")
            else:
                print(f"\n   {feat}:")
                print(f"      {df[feat].value_counts().head()}")
    
    return correlations_df, high_corr_df if high_corr else None, cat_imp_df if cat_importance else None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    config = Config()
    
    # Load data from HuggingFace cache
    datasets = load_datasets(config.DATA_DIR)
    
    # Stage 1: Static features
    df, feature_count = create_static_features(
        datasets['pull_request'],
        datasets['repository'],
        datasets['user'],
        datasets['pr_commit_details'],
        datasets['pr_commits'],
        datasets['related_issue'],
        datasets['pr_task_type'],
        config
    )
    
    # Stage 2: Historical features (TIME-SAFE, OPTIMIZED)
    df = create_historical_features_optimized(df, config)
    
    # Stage 3: Z-score features
    df = create_zscore_features(df, config)
    
    # Stage 4: Interaction features
    df = create_interaction_features(df)
    
    # Stage 5: Drop redundant features
    df = drop_redundant_features(df)
    
    # Analysis
    correlations_df, high_corr_df, cat_imp_df = analyze_features(df)
    
    # Save engineered dataset
    output_path = f"{config.OUTPUT_DIR}engineered_features.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"💾 DATASET SAVED: {output_path}")
    print(f"{'='*80}")
    print(f"   Shape: {df.shape}")
    print(f"   Size: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Save correlation analysis
    correlations_df.to_csv(f"{config.OUTPUT_DIR}feature_correlations.csv", index=False)
    print(f"\n📊 Correlation analysis saved: {config.OUTPUT_DIR}feature_correlations.csv")
    
    # Print final feature list
    print(f"\n📋 FINAL FEATURE LIST ({len(df.columns)} total):")
    print(f"   {df.columns.tolist()}")
    
    print(f"\n{'='*80}")
    print(f"✅ FEATURE ENGINEERING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review feature correlations (especially weak historical features)")
    print(f"   2. Start baseline modeling")
    print(f"   3. Test agent-specific approaches")
    print(f"   4. Optimize thresholds for rejection detection")

if __name__ == "__main__":
    main()
