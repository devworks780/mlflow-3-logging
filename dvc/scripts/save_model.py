import os

import mlflow
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import yaml

load_dotenv()

with open("dvc/models/fitted_model.pkl", "rb") as fd:
    model = joblib.load(fd)
with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

X_val = pd.read_csv("dvc/split/X_val.csv")
y_val = pd.read_csv("dvc/split/y_val.csv").squeeze()

validation_data = X_val.copy()
validation_data[params["target_col"]] = y_val.to_numpy()
validation_dataset = mlflow.data.from_pandas(
    validation_data,
    source="dvc/split",
    targets=params["target_col"],
    name="validation_dataset",
)

EXPERIMENT_NAME = params["experiment_name"]
RUN_NAME = params["run_name"]
REGISTRY_MODEL_NAME = params["registry_model_name"]
TRACKING_URI = params["tracking_uri"]


os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

pip_requirements = "requirements.txt"
prediction = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, 1]
_, err1, err2, _ = confusion_matrix(
    y_val, prediction, labels=[0, 1], normalize="all"
).ravel()
metrics = {
    "err1": err1,
    "err2": err2,
    "auc": roc_auc_score(y_val, probabilities),
    "precision": precision_score(y_val, prediction, zero_division=0),
    "recall": recall_score(y_val, prediction, zero_division=0),
    "f1": f1_score(y_val, prediction, zero_division=0),
    "logloss": log_loss(y_val, probabilities, labels=[0, 1]),
}
signature = mlflow.models.infer_signature(X_val, prediction)
input_example = X_val[:10]
metadata = {"model_type": "monthly"}

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    mlflow.log_input(validation_dataset, context="validation")
    mlflow.log_metrics(metrics)

    mlflow.log_artifact("dvc/data", artifact_path="models_artifacts")

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="models",
        registered_model_name=REGISTRY_MODEL_NAME,
        pip_requirements=pip_requirements,
        signature=signature,
        input_example=input_example,
        metadata=metadata,
        await_registration_for=60,
        serialization_format="cloudpickle",
    )
