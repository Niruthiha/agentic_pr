# Data Dictionary

The dataset consists of **80 features** categorized into four primary groups: **PR Quality & Content**, **User & Agent Context**, **Repo & Environment**, and **Historical CI & Review Signals**.

---

## Group 1: PR Quality & Content (22 Features)
*Metrics describing the code changes, text description, and complexity of the Pull Request.*

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `num_files` | Numeric | Total number of files changed in the PR. |
| `total_changes` | Numeric | Sum of additions and deletions. |
| `additions` | Numeric | Lines of code added. |
| `deletions` | Numeric | Lines of code deleted. |
| `net_change` | Numeric | Additions minus deletions. |
| `add_del_ratio` | Numeric | Ratio of additions to deletions (Additions / Deletions). |
| `avg_changes_per_file` | Numeric | Average lines changed per file (`total_changes` / `num_files`). |
| `num_commits` | Numeric | Number of commits in the PR. |
| `num_unique_extensions` | Numeric | Count of unique file extensions touched (measure of breadth). |
| `num_unique_authors` | Numeric | Number of distinct authors in the commit history. |
| `num_unique_committers` | Numeric | Number of distinct committers in the commit history. |
| `is_test` | Boolean | True if the PR touches files containing "test" in the path. |
| `is_doc` | Boolean | True if the PR touches files containing "doc" or "readme" in the path. |
| `is_config` | Boolean | True if the PR touches configuration files (e.g., .yaml, .json). |
| `body_length` | Numeric | Character count of the PR description body. |
| `title_length` | Numeric | Character count of the PR title. |
| `has_body` | Boolean | True if the PR has a non-empty description. |
| `has_linked_issue` | Boolean | True if the PR links to a GitHub Issue. |
| `num_linked_issues` | Numeric | Count of detected issue links. |
| `task_type` | Categorical | Classifier output of PR intent (e.g., feat, fix, chore, docs). |
| `task_confidence` | Numeric | Confidence score of the task type classifier. |
| `tests_x_size_zscore` | Numeric | Interaction term: Presence of tests weighted by normalized PR size. |

---

## Group 2: User & Agent Context (24 Features)
*Metrics describing the author (Human or AI), their history, and their reputation.*

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `agent` | Categorical | The AI model identity (e.g., OpenAI_Codex, Devin, Copilot). |
| `user_followers` | Numeric | Number of followers the PR author has. |
| `user_following` | Numeric | Number of accounts the PR author follows. |
| `follower_to_following_ratio` | Numeric | Ratio of followers to following. |
| `account_age_days` | Numeric | Days since the user account was created. |
| `is_unknown_user` | Boolean | True if the user ID is null (Ghost User/Deleted Account). |
| `user_prior_pr_count_total` | Numeric | Total PRs submitted by this user across all repos (point-in-time). |
| `user_prior_merge_rate_total` | Numeric | User's global merge rate prior to this PR. |
| `user_prior_merge_rate_in_repo` | Numeric | User's merge rate specifically within this repository. |
| `days_since_last_pr` | Numeric | Days elapsed since the user's previous PR. |
| `is_first_pr_total` | Boolean | True if this is the user's first recorded PR in the dataset. |
| `is_first_pr_in_repo` | Boolean | True if this is the user's first PR in this specific repository. |
| `agent_prior_merge_rate_global` | Numeric | The specific AI agent's global merge rate prior to this PR. |
| `agent_prior_merge_rate_in_repo` | Numeric | The specific AI agent's merge rate in this repository. |
| `agent_x_size_zscore` | Categorical | Interaction: Agent ID combined with normalized PR size. |
| `agent_x_has_tests` | Categorical | Interaction: Agent ID combined with presence of tests. |
| `agent_x_has_issue` | Categorical | Interaction: Agent ID combined with linked issue presence. |
| `user_trust_x_size` | Categorical | Interaction: User reputation score combined with PR size. |

---

## Group 3: Repo & Environment (27 Features)
*Metrics describing the repository context, temporal factors, and statistical norms.*

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `repo_stars` | Numeric | Number of stars the repository has. |
| `repo_forks` | Numeric | Number of forks the repository has. |
| `repo_language` | Categorical | Primary programming language of the repository. |
| `repo_license` | Categorical | License type (e.g., MIT, Apache-2.0). |
| `stars_to_forks_ratio` | Numeric | Ratio of stars to forks (measure of repo hype vs. usage). |
| `repo_prior_pr_count` | Numeric | Total number of PRs in this repo prior to the current one. |
| `repo_prior_merge_rate` | Numeric | Historical merge rate of the repository. |
| `repo_avg_pr_size` | Numeric | Rolling average of `total_changes` for this repository. |
| `repo_std_pr_size` | Numeric | Rolling standard deviation of `total_changes`. |
| `repo_avg_file_count` | Numeric | Rolling average of `num_files` for this repository. |
| `is_new_repo` | Boolean | True if the repository has fewer than 10 prior PRs. |
| `pr_size_zscore` | Numeric | Z-Score of current PR size relative to repo history. |
| `pr_file_count_zscore` | Numeric | Z-Score of current file count relative to repo history. |
| `pr_changes_per_file_zscore` | Numeric | Z-Score of changes-per-file relative to repo history. |
| `created_at` | DateTime | Timestamp of PR creation. |
| `day_of_week` | Categorical | Day of the week (0=Monday, 6=Sunday). |
| `hour_of_day` | Numeric | Hour of PR submission (0-23). |
| `is_weekend` | Boolean | True if submitted on Saturday or Sunday. |
| `month` | Numeric | Month of the year (1-12). |
| `is_holiday_season` | Boolean | True if month is December or January. |

---

## Group 4: Historical CI & Review Signals (7 Features)
*Point-in-time historical metrics derived from GitHub API data. These capture past CI/review patterns without leaking information about the current PR.*

| Feature Name | Type | Description | Safe for Modeling? |
| :--- | :--- | :--- | :--- |
| `ci_passed` | Binary | Whether this PR's CI checks passed (1) or failed (0). | No (Leaky) |
| `review_count` | Numeric | Total number of review comments on this PR. | No (Leaky) |
| `agent_historical_ci_pass_rate` | Numeric | Agent's historical CI pass rate across all prior PRs. | Yes |
| `user_historical_ci_pass_rate` | Numeric | User's historical CI pass rate across all prior PRs. | Yes |
| `repo_avg_ci_fail_rate` | Numeric | Repository's historical CI failure rate (proxy for strictness). | Yes |
| `agent_avg_review_rounds` | Numeric | Agent's average review comment count on prior PRs. | Yes |
| `repo_avg_time_to_merge` | Numeric | Repository's average time-to-merge in hours (for merged PRs). | Yes |

### Important Notes on Group 4

- **Leaky Features (`ci_passed`, `review_count`)**: These are raw values for the current PR, fetched from the GitHub API. They exist after the PR is submitted, so using them would cause data leakage. They are used only to compute the historical aggregates.

- **Safe Features (5 total)**: These are point-in-time historical aggregates computed using `expanding().mean().shift(1)`, meaning they only include data from PRs before the current one. These are valid for the Early Warning System.

---

## Target Variable

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `is_merged` | Boolean | Target variable: True if merged, False if rejected or closed. |

---

## Summary

| Group | Features | Safe for Modeling |
| :--- | :---: | :---: |
| PR Quality & Content | 22 | All |
| User & Agent Context | 24 | All |
| Repo & Environment | 27 | Except leaky columns |
| Historical CI & Review | 7 | Only 5 (exclude `ci_passed`, `review_count`) |
| **Total** | **80** | **59 after cleaning** |

Leaky columns to drop:  
`merged_at`, `closed_at`, `state`, `id`, `number`, `user_id`, `repo_id`, `title`, `body`, `html_url`, `repo_url`, `created_at`, `user_account_created`, `user`, `month`, `ci_passed`, `review_count`
