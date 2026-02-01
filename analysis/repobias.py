#It compares agents only when they are working in the Same Repo AND on the Same Task Type

"""
Task Type & Repository Agent Analysis
=====================================
Analyzes whether agent performance differences are due to:
1. Task selection bias (agents pick different task types)
2. Repo selection bias (agents target different repos)
3. Actual capability differences (within same repo+task)

Author: Analysis for AIDev PR merge prediction study
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("="*70)
print("LOADING DATA")
print("="*70)

# Load main PR data
df = pd.read_parquet('/home/niruthi/ai_code/data/engineered_features.parquet')

# Check if task_type exists
if 'task_type' not in df.columns:
    # Try loading from separate task type file
    try:
        task_df = pd.read_parquet('hf://datasets/hao-li/AIDev/pr_task_type.parquet')
        df = df.merge(task_df[['id', 'task_type']], on='id', how='left')
        print("Loaded task_type from separate file")
    except:
        print("WARNING: task_type not found. Using placeholder.")
        df['task_type'] = 'unknown'

print(f"Total PRs: {len(df):,}")
print(f"Agents: {df['agent'].unique()}")
print(f"Task types: {df['task_type'].nunique()} unique")

# =============================================================================
# 2. TASK TYPE DISTRIBUTION BY AGENT
# =============================================================================
print("\n" + "="*70)
print("TASK TYPE DISTRIBUTION BY AGENT")
print("="*70)

# Cross-tabulation
task_agent_crosstab = pd.crosstab(df['agent'], df['task_type'], margins=True)
print("\nRaw counts:")
print(task_agent_crosstab)

# Percentage within each agent
task_agent_pct = pd.crosstab(df['agent'], df['task_type'], normalize='index') * 100
print("\nPercentage by agent (row %):")
print(task_agent_pct.round(1))

# Chi-square test for independence
contingency = pd.crosstab(df['agent'], df['task_type'])
chi2, p_value, dof, expected = chi2_contingency(contingency)

print(f"\n--- Chi-Square Test for Task Selection Independence ---")
print(f"Chi-square statistic: {chi2:.2f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value:.2e}")
if p_value < 0.05:
    print("Result: SIGNIFICANT - Agents select different task types (selection bias exists)")
else:
    print("Result: NOT SIGNIFICANT - Agents select similar task types")

# =============================================================================
# 3. MERGE RATE BY AGENT AND TASK TYPE
# =============================================================================
print("\n" + "="*70)
print("MERGE RATE BY AGENT AND TASK TYPE")
print("="*70)

# Calculate merge rate for each agent-task combination
merge_by_agent_task = df.groupby(['agent', 'task_type']).agg(
    total_prs=('is_merged', 'count'),
    merged_prs=('is_merged', 'sum'),
    merge_rate=('is_merged', 'mean')
).reset_index()

# Pivot for display
merge_rate_pivot = merge_by_agent_task.pivot(
    index='agent', 
    columns='task_type', 
    values='merge_rate'
) * 100

print("\nMerge Rate (%) by Agent and Task Type:")
print(merge_rate_pivot.round(1))

# Count pivot (to see sample sizes)
count_pivot = merge_by_agent_task.pivot(
    index='agent', 
    columns='task_type', 
    values='total_prs'
)
print("\nSample Size by Agent and Task Type:")
print(count_pivot)

# =============================================================================
# 4. REPO OVERLAP ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("REPO OVERLAP ANALYSIS")
print("="*70)

# Get unique repos per agent
repos_by_agent = df.groupby('agent')['repo_id'].apply(set).to_dict()

agents = list(repos_by_agent.keys())
print("\nRepos per agent:")
for agent in agents:
    print(f"  {agent}: {len(repos_by_agent[agent]):,} repos")

# Pairwise overlap
print("\nPairwise repo overlap:")
for i, agent1 in enumerate(agents):
    for agent2 in agents[i+1:]:
        overlap = repos_by_agent[agent1] & repos_by_agent[agent2]
        total = repos_by_agent[agent1] | repos_by_agent[agent2]
        jaccard = len(overlap) / len(total) if total else 0
        print(f"  {agent1} ∩ {agent2}: {len(overlap)} repos (Jaccard: {jaccard:.3f})")

# Find repos with multiple agents
df['repo_agents'] = df.groupby('repo_id')['agent'].transform('nunique')
multi_agent_repos = df[df['repo_agents'] >= 2]['repo_id'].unique()
print(f"\nRepos with 2+ agents: {len(multi_agent_repos):,}")

# =============================================================================
# 5. WITHIN-REPO AGENT COMPARISON
# =============================================================================
print("\n" + "="*70)
print("WITHIN-REPO AGENT COMPARISON")
print("="*70)

# Filter to multi-agent repos
df_multi = df[df['repo_id'].isin(multi_agent_repos)].copy()
print(f"PRs in multi-agent repos: {len(df_multi):,}")

# Merge rate by agent in shared repos only
within_repo_rates = df_multi.groupby('agent').agg(
    total_prs=('is_merged', 'count'),
    merged_prs=('is_merged', 'sum'),
    merge_rate=('is_merged', 'mean')
).round(4)

print("\nMerge rates in SHARED repos only:")
print(within_repo_rates)

# Compare to overall rates
overall_rates = df.groupby('agent')['is_merged'].mean()
print("\nComparison (Shared vs Overall):")
for agent in agents:
    if agent in within_repo_rates.index and agent in overall_rates.index:
        shared = within_repo_rates.loc[agent, 'merge_rate']
        overall = overall_rates[agent]
        diff = shared - overall
        print(f"  {agent}: Shared={shared:.3f}, Overall={overall:.3f}, Δ={diff:+.3f}")

# =============================================================================
# 6. WITHIN-REPO + WITHIN-TASK COMPARISON (STRICTEST CONTROL)
# =============================================================================
print("\n" + "="*70)
print("WITHIN-REPO + WITHIN-TASK COMPARISON (Strictest Control)")
print("="*70)

# Find (repo, task_type) pairs with multiple agents
df_multi['repo_task'] = df_multi['repo_id'].astype(str) + '_' + df_multi['task_type'].astype(str)
df_multi['repo_task_agents'] = df_multi.groupby('repo_task')['agent'].transform('nunique')

# Filter to repo-task combinations with 2+ agents
df_strict = df_multi[df_multi['repo_task_agents'] >= 2].copy()
print(f"PRs in (repo, task) pairs with 2+ agents: {len(df_strict):,}")
print(f"Unique (repo, task) pairs: {df_strict['repo_task'].nunique():,}")

if len(df_strict) > 0:
    # Merge rate by agent under strictest control
    strict_rates = df_strict.groupby('agent').agg(
        total_prs=('is_merged', 'count'),
        merged_prs=('is_merged', 'sum'),
        merge_rate=('is_merged', 'mean')
    ).round(4)
    
    print("\nMerge rates with REPO + TASK controlled:")
    print(strict_rates)
    
    # Statistical test: are agent differences significant?
    if strict_rates.shape[0] >= 2:
        # Create contingency table for chi-square
        strict_contingency = df_strict.groupby('agent')['is_merged'].agg(['sum', 'count'])
        strict_contingency['not_merged'] = strict_contingency['count'] - strict_contingency['sum']
        strict_contingency = strict_contingency[['sum', 'not_merged']]
        strict_contingency.columns = ['merged', 'not_merged']
        
        chi2_strict, p_strict, dof_strict, _ = chi2_contingency(strict_contingency)
        print(f"\n--- Chi-Square Test (Repo+Task Controlled) ---")
        print(f"Chi-square: {chi2_strict:.2f}, p-value: {p_strict:.4f}")
        if p_strict < 0.05:
            print("Result: SIGNIFICANT - Agent differences persist even with controls")
        else:
            print("Result: NOT SIGNIFICANT - No agent differences when controlling for repo+task")
else:
    print("Not enough data for strict comparison")

# =============================================================================
# 7. PAIRWISE AGENT COMPARISONS (WITHIN SAME REPO+TASK)
# =============================================================================
print("\n" + "="*70)
print("PAIRWISE AGENT COMPARISONS (Same Repo + Same Task)")
print("="*70)

if len(df_strict) > 0:
    pairwise_results = []
    
    for i, agent1 in enumerate(agents):
        for agent2 in agents[i+1:]:
            # Find repo-task pairs where both agents have PRs
            agent1_repo_tasks = set(df_strict[df_strict['agent'] == agent1]['repo_task'])
            agent2_repo_tasks = set(df_strict[df_strict['agent'] == agent2]['repo_task'])
            shared_repo_tasks = agent1_repo_tasks & agent2_repo_tasks
            
            if len(shared_repo_tasks) >= 5:  # Minimum 5 shared contexts
                # Get PRs from shared contexts
                df_pair = df_strict[
                    (df_strict['repo_task'].isin(shared_repo_tasks)) &
                    (df_strict['agent'].isin([agent1, agent2]))
                ]
                
                # Calculate rates
                rates = df_pair.groupby('agent')['is_merged'].agg(['mean', 'count'])
                
                if agent1 in rates.index and agent2 in rates.index:
                    rate1, n1 = rates.loc[agent1, 'mean'], rates.loc[agent1, 'count']
                    rate2, n2 = rates.loc[agent2, 'mean'], rates.loc[agent2, 'count']
                    
                    # Fisher exact test (better for small samples)
                    merged1 = int(rate1 * n1)
                    merged2 = int(rate2 * n2)
                    table = [[merged1, int(n1 - merged1)], 
                             [merged2, int(n2 - merged2)]]
                    
                    try:
                        odds_ratio, p_val = fisher_exact(table)
                    except:
                        odds_ratio, p_val = np.nan, np.nan
                    
                    pairwise_results.append({
                        'agent1': agent1,
                        'agent2': agent2,
                        'shared_contexts': len(shared_repo_tasks),
                        'n1': int(n1),
                        'n2': int(n2),
                        'rate1': rate1,
                        'rate2': rate2,
                        'diff': rate1 - rate2,
                        'odds_ratio': odds_ratio,
                        'p_value': p_val,
                        'significant': p_val < 0.05 if not np.isnan(p_val) else False
                    })
    
    if pairwise_results:
        pairwise_df = pd.DataFrame(pairwise_results)
        print("\nPairwise comparisons (controlling for repo + task type):")
        print(pairwise_df.to_string(index=False, float_format='%.3f'))
        
        # Summary
        n_sig = pairwise_df['significant'].sum()
        n_total = len(pairwise_df)
        print(f"\nSignificant differences: {n_sig}/{n_total} pairs")
    else:
        print("Not enough shared repo-task pairs for pairwise comparison")

# =============================================================================
# 8. TASK SELECTION STRATEGY ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("TASK SELECTION STRATEGY ANALYSIS")
print("="*70)

# Calculate task difficulty (inverse of overall merge rate for that task)
task_difficulty = df.groupby('task_type').agg(
    total_prs=('is_merged', 'count'),
    merge_rate=('is_merged', 'mean')
).sort_values('merge_rate', ascending=False)

task_difficulty['difficulty'] = 1 - task_difficulty['merge_rate']
print("\nTask types by difficulty (harder = lower merge rate):")
print(task_difficulty.round(3))

# Calculate weighted average difficulty per agent
agent_difficulty = []
for agent in agents:
    agent_tasks = df[df['agent'] == agent]['task_type'].value_counts(normalize=True)
    weighted_diff = 0
    for task, pct in agent_tasks.items():
        if task in task_difficulty.index:
            weighted_diff += pct * task_difficulty.loc[task, 'difficulty']
    agent_difficulty.append({
        'agent': agent,
        'weighted_difficulty': weighted_diff,
        'total_prs': len(df[df['agent'] == agent])
    })

agent_diff_df = pd.DataFrame(agent_difficulty).sort_values('weighted_difficulty', ascending=False)
print("\nWeighted task difficulty by agent (higher = harder tasks):")
print(agent_diff_df.round(3))

# Correlation: does targeting harder tasks correlate with lower merge rate?
agent_merge_rates = df.groupby('agent')['is_merged'].mean()
agent_diff_df['merge_rate'] = agent_diff_df['agent'].map(agent_merge_rates)

corr = agent_diff_df['weighted_difficulty'].corr(agent_diff_df['merge_rate'])
print(f"\nCorrelation (task difficulty vs merge rate): {corr:.3f}")
if corr < -0.5:
    print("Strong negative correlation: Agents targeting harder tasks have lower merge rates")
elif corr < 0:
    print("Weak negative correlation: Some evidence of task difficulty effect")
else:
    print("No negative correlation: Task difficulty doesn't explain merge rate differences")

# =============================================================================
# 9. SUMMARY AND CONCLUSIONS
# =============================================================================
print("\n" + "="*70)
print("SUMMARY AND CONCLUSIONS")
print("="*70)

print("""
KEY FINDINGS:
=============

1. TASK SELECTION BIAS
   - Do agents select different task types? (Chi-square test above)
   - If significant: agents have different task selection strategies
   
2. REPO SELECTION BIAS  
   - How much repo overlap exists between agents?
   - Low overlap = hard to compare directly
   
3. WITHIN-REPO COMPARISON
   - When comparing in shared repos: do agent differences persist?
   - If differences shrink: repo selection was a confound
   
4. WITHIN-REPO+TASK COMPARISON (Strictest)
   - When comparing same repo + same task type:
   - If differences persist: actual capability differences
   - If differences disappear: pure selection bias

5. TASK DIFFICULTY TARGETING
   - Do some agents target harder tasks?
   - This would explain lower merge rates without implying lower quality
""")

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================
results = {
    'task_agent_distribution': task_agent_pct.to_dict(),
    'task_selection_chi2': {'chi2': chi2, 'p_value': p_value, 'significant': p_value < 0.05},
    'multi_agent_repos': len(multi_agent_repos),
    'within_repo_rates': within_repo_rates.to_dict() if 'within_repo_rates' in dir() else None,
    'agent_task_difficulty': agent_diff_df.to_dict() if 'agent_diff_df' in dir() else None,
}

print("\nAnalysis complete!")
print("Results can be used to refine paper claims about agent identity effects.")
