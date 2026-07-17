# scripts/fit.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
import yaml
import os
import joblib


# обучение модели
def fit_model():
    # Прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    # загрузите результат предыдущего шага: initial_data.csv
    data = pd.read_csv("dvc/data/initial_data.csv")

    # реализуйте основную логику шага с использованием гиперпараметров
    target_col = params["target_col"]
    one_hot_drop = params["one_hot_drop"]
    auto_class_weights = params["auto_class_weights"]
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    train_data, val_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[target_col],
    )

    y_train = train_data[target_col]
    y_val = val_data[target_col]

    train_data = train_data.drop(columns=[target_col])
    val_data = val_data.drop(columns=[target_col])

    # Удаляем признак end_date (утечка информации)
    train_data = train_data.drop(columns=["end_date"], errors="ignore")
    val_data = val_data.drop(columns=["end_date"], errors="ignore")

    os.makedirs("dvc/split", exist_ok=True)
    train_data.to_csv("dvc/split/X_train.csv", index=False)
    val_data.to_csv("dvc/split/X_val.csv", index=False)
    y_train.to_csv("dvc/split/y_train.csv", index=False)
    y_val.to_csv("dvc/split/y_val.csv", index=False)

    cat_features = train_data.select_dtypes(include="object")

    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[
        potential_binary_features[potential_binary_features].index
    ]
    other_cat_features = cat_features[
        potential_binary_features[~potential_binary_features].index
    ]
    num_features = train_data.select_dtypes(["float"])

    preprocessor = ColumnTransformer(
        [
            (
                "binary",
                OneHotEncoder(drop=one_hot_drop),
                binary_cat_features.columns.tolist(),
            ),
            (
                "cat",
                CatBoostEncoder(return_df=False),
                other_cat_features.columns.tolist(),
            ),
            ("num", StandardScaler(), num_features.columns.tolist()),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = CatBoostClassifier(auto_class_weights=auto_class_weights)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(train_data, y_train)

    threshold = params.get("classification_threshold", 0.6)
    pipeline.named_steps["model"].set_probability_threshold(threshold)

    # сохраните обученную модель в models/fitted_model.pkl
    os.makedirs(
        "dvc/models", exist_ok=True
    )  # создание директории, если её ещё нет
    with open("dvc/models/fitted_model.pkl", "wb") as fd:
        joblib.dump(pipeline, fd)


if __name__ == "__main__":
    fit_model()
