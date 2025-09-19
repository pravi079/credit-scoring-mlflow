from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(X):
    """Build preprocessing pipeline for credit scoring dataset."""

    # Numeric columns
    numeric_features = [
        "age", "balance", "day", "duration",
        "campaign", "pdays", "previous"
    ]

    # Categorical columns
    categorical_features = [
        "job", "marital", "education", "default",
        "housing", "loan", "contact", "month", "poutcome"
    ]

    # Numeric pipeline: impute missing with median, then scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: impute missing with most_frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor
