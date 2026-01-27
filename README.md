# Agentic PR the Prediction Model

## Data Processing and Feature Engineering Methodology

A granular definition of every column in this dataset is available at data_dictionary.md

To ensure reproducibility, this project utilizes a strict "Point-in-Time" feature engineering pipeline that prevents data leakage by calculating historical features only using data available prior to each Pull Request's creation.


### 1. Input Data Sources: https://huggingface.co/datasets/hao-li/AIDev/blob/main/data_table.md
We aggregated data from 7 raw tables (SQL/Parquet inputs) sourced from the GitHub REST API:

* **Primary Table:** `pull_request` (33.6k rows) - The core unit of analysis.
* **Context Tables:** `repository` (2.8k rows), `user` (1.8k rows).
* **Detail Tables:** `pr_commit_details` (711k rows), `pr_commits` (88k rows), `related_issue`, `pr_task_type`.

### Preprocessing Filters
* **Status Filter:** Excluded Open PRs as the target variable is undefined.
* **Chronological Sorting:** Sorted by `created_at` to establish a valid temporal sequence for historical windowing.
* **Final Dataset:** 31,284 Closed PRs (76.8% Merged / 23.2% Rejected).

### 2. Feature Engineering Taxonomy
The final dataset contains 73 features derived via a 5-stage pipeline.

#### A. Static and Content Features (Extracted directly)
* **Source:** Raw text and diff statistics.
* **Logic:**
    * **Complexity:** `total_changes`, `num_files`, `num_commits`, and `entropy` (count of unique file extensions).
    * **Classification:** `is_test` (regex matching on filenames), `is_doc`, `is_config`.
    * **Text Quality:** `body_length`, `title_length`, `has_linked_issue`.

#### B. Contextual Features (Enriched)
* **Source:** User and Repository metadata.
* **Logic:**
    * **Repo Health:** `repo_stars`, `repo_language`, `stars_to_forks_ratio`.
    * **User Reputation:** `user_followers`, `account_age_days`, `follower_to_following_ratio`.
    * **Identity:** `is_unknown_user` (Ghost user detection) and `agent` (AI Model ID).

#### C. Historical "Time-Travel" Features (Core Logic)
To capture reputation without look-ahead bias, we computed cumulative statistics strictly prior to the current PR's `created_at` timestamp.
* **User Track Record:** `user_prior_merge_rate`, `days_since_last_pr`, `is_first_pr`.
* **Repo Norms:** `repo_prior_merge_rate`, `repo_avg_pr_size` (Rolling average).
* **Agent History:** `agent_prior_merge_rate_global` (Performance across all repositories).

#### D. Interaction and Normalized Features
* **Z-Scores:** PR size normalized by the repository's history (e.g., `pr_size_zscore`). This normalization ensures that a 500-line PR is treated contextually—normal for a monorepo but significant for a micro-library.
* **Interactions:** `agent_x_size_zscore` (Captures agent performance on large vs. small PRs) and `agent_x_has_tests`.

### 3. Data Transformations
* **Outlier Capping:** Numeric features were capped at the 99.0th percentile to prevent skew (e.g., `total_changes` capped at approximately 30,000 lines).
* **Missing Values:** Categorical imputation applied for missing task types; Ghost users (deleted accounts) were flagged explicitly.
