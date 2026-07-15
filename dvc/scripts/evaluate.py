# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os


# оценка качества модели
def evaluate_model():
    # прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    # загрузите результат прошлого шага: fitted_model.pkl
    with open("dvc/models/fitted_model.pkl", "rb") as fd:
        pipeline = joblib.load(fd)

    # реализуйте основную логику шага с использованием прочтённых
    # гиперпараметров
    n_splits = params["n_splits"]
    metrics = params["metrics"]

    X_val = pd.read_csv("dvc/split/X_val.csv")
    y_val = pd.read_csv("dvc/split/y_val.csv").squeeze()

    cv_strategy = StratifiedKFold(n_splits=n_splits)
    cv_res = cross_validate(
        pipeline,
        X_val,
        y_val,
        cv=cv_strategy,
        n_jobs=-1,
        scoring=metrics,
    )

    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3)
    # сохраните результата кросс-валидации в cv_res.json
    os.makedirs("dvc/cv_results", exist_ok=True)
    with open("dvc/cv_results/cv_res.json", "w") as fd:
        json.dump(cv_res, fd)


if __name__ == "__main__":
    evaluate_model()
