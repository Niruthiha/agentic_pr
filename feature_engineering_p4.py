"""
STEP 2: COMPUTE HISTORICAL AGGREGATES
=====================================
Run this AFTER the GraphQL fetch completes
"""

import pandas as pd
import numpy as np

# Load the fetched data
df = pd.read_parquet('data_with_raw_ci_reviews.parquet')
print(f"Loaded: {len(df)} rows")
print(f"ci_passed coverage: {df['ci_passed'].notna().mean():.1%}")
print(f"review_count coverage: {df['review_count'].notna().mean():.1%}")

# Sort by time (CRITICAL for point-in-time calculations)
df = df.sort_values('created_at').reset_index(drop=True)

# --- 1. Agent Historical CI Pass Rate ---
df['agent_historical_ci_pass_rate'] = (
    df.groupby('agent')['ci_passed']
    .transform(lambda x: x.expanding().mean().shift(1))
)

# --- 2. User Historical CI Pass Rate ---
df['user_historical_ci_pass_rate'] = (
    df.groupby('user_id')['ci_passed']
    .transform(lambda x: x.expanding().mean().shift(1))
)

# --- 3. Repo Historical CI Fail Rate ---
df['repo_historical_ci_pass_rate'] = (
    df.groupby('repo_id')['ci_passed']
    .transform(lambda x: x.expanding().mean().shift(1))
)
df['repo_avg_ci_fail_rate'] = 1 - df['repo_historical_ci_pass_rate']

# --- 4. Agent Avg Review Rounds ---
df['agent_avg_review_rounds'] = (
    df.groupby('agent')['review_count']
    .transform(lambda x: x.expanding().mean().shift(1))
)

# --- 5. Repo Avg Time to Merge ---
df['time_to_merge_hours'] = np.where(
    df['is_merged'] == True,
    (pd.to_datetime(df['merged_at']) - pd.to_datetime(df['created_at'])).dt.total_seconds() / 3600,
    np.nan
)
df['repo_avg_time_to_merge'] = (
    df.groupby('repo_id')['time_to_merge_hours']
    .transform(lambda x: x.expanding().mean().shift(1))
)

# --- Fill NaNs ---
df['agent_historical_ci_pass_rate'] = df['agent_historical_ci_pass_rate'].fillna(0.5)
df['user_historical_ci_pass_rate'] = df['user_historical_ci_pass_rate'].fillna(0.5)
df['repo_avg_ci_fail_rate'] = df['repo_avg_ci_fail_rate'].fillna(0.5)
df['agent_avg_review_rounds'] = df['agent_avg_review_rounds'].fillna(df['review_count'].median())
df['repo_avg_time_to_merge'] = df['repo_avg_time_to_merge'].fillna(df['time_to_merge_hours'].median())

# --- Cleanup ---
df = df.drop(columns=['repo_historical_ci_pass_rate', 'time_to_merge_hours'], errors='ignore')

# --- Save ---
df.to_parquet('/home/niruthi/ai_code/data/engineered_features_v2.parquet', index=False)
print("\n✅ Saved: engineered_features_v2.parquet")

# --- Verify ---
new_features = ['agent_historical_ci_pass_rate', 'user_historical_ci_pass_rate', 
                'repo_avg_ci_fail_rate', 'agent_avg_review_rounds', 'repo_avg_time_to_merge']
print("\nNew Features Summary:")
print(df[new_features].describe().round(3))
