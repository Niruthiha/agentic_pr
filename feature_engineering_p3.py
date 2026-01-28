import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
GRAPHQL_URL = "https://api.github.com/graphql"

def fetch_pr_graphql(args):
    """Fetch everything in ONE API call using GraphQL"""
    idx, owner, repo, pr_number = args
    
    query = """
    query($owner: String!, $repo: String!, $pr: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $pr) {
          commits(last: 1) {
            nodes {
              commit {
                statusCheckRollup {
                  state
                }
              }
            }
          }
          reviewThreads {
            totalCount
          }
          reviews {
            totalCount
          }
        }
      }
    }
    """
    
    variables = {"owner": owner, "repo": repo, "pr": pr_number}
    
    try:
        resp = requests.post(
            GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers=HEADERS,
            timeout=10
        )
        
        if resp.status_code != 200:
            return idx, np.nan, np.nan
        
        data = resp.json()
        pr_data = data.get('data', {}).get('repository', {}).get('pullRequest')
        
        if not pr_data:
            return idx, np.nan, np.nan
        
        # CI Status
        ci_passed = np.nan
        commits = pr_data.get('commits', {}).get('nodes', [])
        if commits:
            rollup = commits[0].get('commit', {}).get('statusCheckRollup')
            if rollup:
                state = rollup.get('state')
                if state == 'SUCCESS': ci_passed = 1
                elif state in ['FAILURE', 'ERROR']: ci_passed = 0
        
        # Review count
        reviews = pr_data.get('reviews', {}).get('totalCount', 0)
        threads = pr_data.get('reviewThreads', {}).get('totalCount', 0)
        review_count = reviews + threads
        
        return idx, ci_passed, review_count
    
    except Exception as e:
        return idx, np.nan, np.nan


# Load and prepare
df = pd.read_parquet('/home/niruthi/ai_code/data/engineered_features.parquet')
df['ci_passed'] = np.nan
df['review_count'] = np.nan

# Parse owner/repo
tasks = []
for idx, row in df.iterrows():
    parts = row['repo_url'].replace("https://api.github.com/repos/", "").split("/")
    if len(parts) >= 2:
        tasks.append((idx, parts[0], parts[1], row['number']))

# Run parallel
MAX_WORKERS = 15  # GraphQL is more efficient

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(fetch_pr_graphql, task): task for task in tasks}
    
    for future in tqdm(as_completed(futures), total=len(tasks)):
        idx, ci, rev = future.result()
        df.at[idx, 'ci_passed'] = ci
        df.at[idx, 'review_count'] = rev

df.to_parquet('data_with_raw_ci_reviews.parquet', index=False)
