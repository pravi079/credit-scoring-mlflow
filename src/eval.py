import joblib
import mlflow
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from data import load_data
from pathlib import Path

MODEL_PATH = Path("models/credit_scoring_pipeline.pkl")

# MLflow experiment
mlflow.set_experiment("credit_scoring")

def evaluate():
    X_train, X_test, y_train, y_test = load_data()
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # print("📊 Classification Report:")
    # print(classification_report(y_test, y_pred))
    # Metrics
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="yes")  # target is 'yes'/'no'


    print("📊 Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    # MLflow logging
    with mlflow.start_run(run_name="logreg_eval"):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
 
if __name__ == "__main__":
    evaluate()

