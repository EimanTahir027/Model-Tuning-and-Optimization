import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------
df = pd.read_csv("Telco-Customer-Churn.csv")

print("Dataset Info:\n")
print(df.info())
print("\nClass Distribution:\n")
print(df["Churn"].value_counts())
print("\nSample Data:\n", df.head())

# ---------------------------------------------------------------
# 2. Stateless cleaning (safe before the split: no statistics learned)
# ---------------------------------------------------------------
# Blank strings become NaN. The median is NOT computed here;
# the imputer inside the pipeline will do it on training data only.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# customerID is a unique identifier, not a predictive feature
df = df.drop(columns=["customerID"])

# Target: fixed mapping, no fitting involved
y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Churn"])

numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numerical_features]

# ---------------------------------------------------------------
# 3. Split FIRST
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------
# 4. Preprocessing + model in one Pipeline
#    Everything is fitted on training data only. Inside cross-validation,
#    it is refitted on each training fold separately.
# ---------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_features,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)


def make_pipeline():
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )


# ---------------------------------------------------------------
# 5. Initial model
# ---------------------------------------------------------------
initial_pipe = make_pipeline()
initial_pipe.fit(X_train, y_train)

y_pred = initial_pipe.predict(X_test)
accuracy_initial = accuracy_score(y_test, y_pred)

print(f"\nInitial Model Accuracy: {accuracy_initial:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------------
# 6. Hyperparameter tuning (parameters are prefixed with 'model__')
# ---------------------------------------------------------------
param_dist = {
    "model__n_estimators": np.arange(50, 200, 10),
    "model__max_depth": [None, 5, 10, 15],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 4],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

random_search = RandomizedSearchCV(
    estimator=make_pipeline(),
    param_distributions=param_dist,
    n_iter=20,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print(f"Best Parameters (RandomizedSearchCV): {best_params}")

best_model = random_search.best_estimator_

# ---------------------------------------------------------------
# 7. Final evaluation on the untouched test set
# ---------------------------------------------------------------
y_pred_tuned = best_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"\nTuned Model Accuracy: {accuracy_tuned:.4f}")
print(
    "\nClassification Report (Tuned Model):\n",
    classification_report(y_test, y_pred_tuned),
)

# ---------------------------------------------------------------
# 8. Cross-validation on TRAINING data only; the pipeline is cloned
#    and refitted inside every fold, so there is no leakage.
# ---------------------------------------------------------------
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy")

print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")
