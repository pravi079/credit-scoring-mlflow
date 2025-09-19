import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from data import load_data
from features import build_preprocessor
from pathlib import Path

MODEL_PATH = Path("models/credit_scoring_pipeline.pkl")

def train():
    X_train, X_test, y_train, y_test = load_data()
    preprocessor = build_preprocessor(X_train)

    # MLflow experiment
    mlflow.set_experiment("credit_scoring")

    with mlflow.start_run(run_name="logreg_training"):

        # Define model
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])

        # Train
        model.fit(X_train, y_train)

        # Log params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # save locally too
        joblib.dump(model, MODEL_PATH)
        print(f"✅ Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train()
