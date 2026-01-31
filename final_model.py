"""
Save only the best model (unoptimized XGBoost)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import joblib

# Load data
df = pd.read_parquet('/home/niruthi/ai_code/data/engineered_features_v2.parquet')

leaky_cols = ['merged_at', 'closed_at', 'state', 'id', 'number', 'user_id', 'repo_id',
              'title', 'body', 'html_url', 'repo_url', 'created_at', 'user_account_created',
              'user', 'month', 'ci_passed', 'review_count']
redundant_cols = ['additions', 'deletions', 'repo_forks']
df_clean = df.drop(columns=leaky_cols + redundant_cols, errors='ignore')

X = df_clean.drop(columns=['is_merged'])
y = df_clean['is_merged']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df_clean['agent']
)

# Preprocessor
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
num_cols = [c for c in X.select_dtypes(include=['number']).columns if c not in bool_cols]

preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', 'passthrough', num_cols),
    ('bool', 'passthrough', bool_cols)
], verbose_feature_names_out=False)

X_train_prep = preprocessor.fit_transform(X_train)

# Train best model
print("Training XGBoost (best model)...")
xgb_best = XGBClassifier(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=8,
    scale_pos_weight=0.5, 
    random_state=42, 
    n_jobs=-1
)
xgb_best.fit(X_train_prep, y_train)

# Save
joblib.dump(xgb_best, 'xgb_best.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')

print("✓ Saved: xgb_best.joblib")
print("✓ Saved: preprocessor.joblib")

# Print feature info for app.py
print(f"\n--- Features ({len(X.columns)}) ---")
print("Categorical:", cat_cols)
print("Numeric:", num_cols)
print("Boolean:", bool_cols)
