# scripts/fit.py

# поправь импорты при необходимости, например,
# удали дублирующиеся импорты, добавь недостающие
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import (
    SplineTransformer,
    QuantileTransformer,
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml
import os
import joblib


# обучение модели
def data_preprocessing():
    # Прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    is_preprocessing = params.get("is_preprocessing", False)
    if not is_preprocessing:
        print(
            "Data preprocessing is disabled in params.yaml. Skipping this step."
        )
        return
    # загрузите результат предыдущего шага: initial_data.csv
    data = pd.read_csv("dvc/data/initial_data.csv")

    preprocessing_data = preprocess_data_local(data)
    # сохраните обученную модель в models/fitted_model.pkl
    os.makedirs("dvc/data", exist_ok=True)
    preprocessing_data.to_csv("dvc/data/preprocessing_data.csv", index=False)


def preprocess_data_local(data):
    cat_columns = ["type", "payment_method", "internet_service", "gender"]
    num_columns = ["monthly_charges", "total_charges"]

    encoder_oh = OneHotEncoder(
        categories="auto",
        handle_unknown="ignore",
        max_categories=10,
        sparse_output=False,
        drop="first",
    )

    n_knots = 3
    degree_spline = 4
    n_quantiles = 100
    degree = 3
    n_bins = 5
    encode = "ordinal"
    strategy = "uniform"
    subsample = None

    encoder_spl = SplineTransformer(n_knots=n_knots, degree=degree_spline)
    encoder_q = QuantileTransformer(n_quantiles=n_quantiles)
    encoder_rb = RobustScaler()
    encoder_pol = PolynomialFeatures(degree=degree)
    encoder_kbd = KBinsDiscretizer(
        n_bins=n_bins, encode=encode, strategy=strategy, subsample=subsample
    )

    numeric_transformer = ColumnTransformer(
        transformers=[
            ("spl", encoder_spl, num_columns),
            ("q", encoder_q, num_columns),
            ("rb", encoder_rb, num_columns),
            ("pol", encoder_pol, num_columns),
            ("kbd", encoder_kbd, num_columns),
        ]
    )

    categorical_transformer = Pipeline(steps=[("encoder", encoder_oh)])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_columns),
            ("cat", categorical_transformer, cat_columns),
        ],
        n_jobs=-1,
    )

    preprocessing_data = data.copy()
    preprocessing_data[num_columns] = preprocessing_data[num_columns].fillna(0)
    encoded_features = preprocessor.fit_transform(
        preprocessing_data
    )  # ваш код здесь #

    transformed_df = pd.DataFrame(
        encoded_features, columns=preprocessor.get_feature_names_out()
    )  # ваш код здесь #

    # # добавь и удаление исходных колонок, чтобы не было дублирования
    preprocessing_data = pd.concat(
        [preprocessing_data, transformed_df], axis=1
    )
    preprocessing_data.drop(columns=num_columns + cat_columns, inplace=True)
    return preprocessing_data


if __name__ == "__main__":
    data_preprocessing()
