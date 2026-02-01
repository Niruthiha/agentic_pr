import pandas as pd
import numpy as np

# 1. Load Data
df = pd.read_parquet('/home/niruthi/ai_code/data/engineered_features.parquet')

# 2. Ensure Time Sorting (CRITICAL)
if 'created_at' in df.columns:
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('created_at')

# 3. Calculate "Time to Merge" for each PR (in hours)
# If merged_at is null (rejected), we can't use it for the *repo's* average speed.
# We interpret "Time to Merge" as "How fast does this repo accept code?"
df['merged_at'] = pd.to_datetime(df['merged_at'])
df['pr_duration_hours'] = (df['merged_at'] - df['created_at']).dt.total_seconds() / 3600

# 4. Calculate Historical Average (Expanding Window)
# Group by Repo -> Expanding Mean -> Shift(1) (Prevent Leakage)
print("Calculating repo_avg_time_to_merge...")
df['repo_avg_time_to_merge'] = (
    df.groupby('repo_id')['pr_duration_hours']
    .expanding()
    .mean()
    .shift(1) # <--- The "No Peeking" Shift
    .reset_index(level=0, drop=True)
)

# Fill NaNs (First PR in a repo has no history)
# We fill with the global average to be safe
global_avg_time = df['pr_duration_hours'].mean()
df['repo_avg_time_to_merge'] = df['repo_avg_time_to_merge'].fillna(global_avg_time)

print("Done! Feature added.")
