# model_training.py

import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ========================
# 1. Load Training Data
# ========================
train = pd.read_csv("train.csv")

# ========================
# 2. Drop Leakage Columns
# ========================
# Remove any columns related to post-transplant survival time
for col in ["efs_time", "efs_time_bin"]:
    if col in train.columns:
        train.drop(columns=[col], inplace=True)

# ========================
# 3. Features & Target
# ========================
target = "efs"
X = train.drop(columns=[target])
y = train[target]

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

# ========================
# 4. Preprocessor
# ========================
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", num_imputer, num_cols),
    ("cat", Pipeline(steps=[("imputer", cat_imputer), ("ohe", ohe)]), cat_cols)
])

# Fit preprocessor
X_p = preprocessor.fit_transform(X)

# ========================
# 5. Models
# ========================
rf = RandomForestClassifier(
    n_estimators=100, max_depth=12, min_samples_split=8,
    n_jobs=-1, random_state=42
)
xgb = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    use_label_encoder=False, eval_metric="logloss"
)
log_reg = LogisticRegression(max_iter=200, solver="liblinear")

voting_clf = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("log", log_reg)],
    voting="soft"
)

# ========================
# 6. Train Voting Classifier
# ========================
voting_clf.fit(X_p, y)

# ========================
# 7. Save Model & Preprocessor
# ========================
joblib.dump(voting_clf, "voting_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(num_cols, "num_cols.pkl")
joblib.dump(cat_cols, "cat_cols.pkl")

print("Model and preprocessor saved successfully.")
