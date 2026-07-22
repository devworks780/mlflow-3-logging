# scripts/data_extend.py

# поправь импорты при необходимости, например,
# удали дублирующиеся импорты, добавь недостающие
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from autofeat import AutoFeatClassifier
from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)


# обучение модели
def data_extend():
    # Прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    data = pd.read_csv("dvc/data/preprocessing_data.csv")
    is_extend = params.get("is_extend", False)
    if not is_extend:
        print("Data extend is disabled in params.yaml. Skipping this step.")
        data.to_csv("dvc/data/extended_data.csv", index=False)
        return

    extended_data = extend_data_local(data)
    # сохраните обученную модель в models/fitted_model.pkl
    os.makedirs("dvc/data", exist_ok=True)
    extended_data.to_csv("dvc/data/extended_data.csv", index=False)
    return extended_data


def extend_data_local(dat):
    data = dat.copy()
    # print(data.head())
    print(data.dtypes)

    cat_features = [
        "paperless_billing",
        #       "payment_method",
        #       "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        #      "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "multiple_lines",
    ]
    num_features = ["monthly_charges", "total_charges"]

    features = cat_features + num_features
    target = "target"
    split_column = "begin_date"
    test_size = 0.2

    data = data.sort_values(by=[split_column])
    y = data[target]
    X = data.drop(columns=[target])
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].fillna(0)

    # AutoFeat с преобразованиями log и sqrt не должен получать признаки,
    # содержащие отрицательные значения.
    finite_num_cols = num_cols[np.isfinite(X[num_cols].to_numpy()).all(axis=0)]
    non_constant_num_cols = finite_num_cols[X[finite_num_cols].nunique() > 1]
    non_negative_num_cols = non_constant_num_cols[
        (X[non_constant_num_cols] >= 0).all()
    ].tolist()
    excluded_num_cols = num_cols.difference(non_negative_num_cols).tolist()

    if not non_negative_num_cols:
        raise ValueError(
            "Не найдено числовых признаков без отрицательных значений"
        )

    print(
        "Числовые признаки, переданные в AutoFeat:",
        non_negative_num_cols,
    )
    print(
        "Исключены отрицательные, константные или бесконечные признаки:",
        excluded_num_cols,
    )
    print("---" * 20)
    # print(data[num_cols].isnull().count())
    # missing_capitals = data[data[num_cols].isna().any(axis=1)][num_cols]
    # print(missing_capitals)

    print("Пропуски в X_train:", X[num_cols].isna().sum().sum())
    print("Пропуски в y_train:", y.isna().sum())

    X[num_cols].to_csv("dvc/data/X.csv", index=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False,
    )

    # transformations = ["1/", "exp", "log", "abs", "sqrt", "^2", "^3", "1+", "1-", "sin", "cos", "exp-", "2^"]
    # Все эти преобразования определены и для нулевых значений.
    # 1/ и log здесь использовать нельзя: в выбранных колонках есть нули.
    transformations = ("abs", "sqrt", "^2")

    afc = AutoFeatClassifier(
        # categorical_cols=cat_features,
        feateng_cols=non_negative_num_cols,
        transformations=transformations,
        feateng_steps=1,
        n_jobs=1,
    )

    X_train_features = afc.fit_transform(
        X_train[non_negative_num_cols].copy(),
        y_train.copy().fillna(0),
    )
    X_test_features = afc.transform(X_test[non_negative_num_cols].copy())

    # fit_transform/transform возвращают как исходные числовые признаки,
    # так и созданные AutoFeat. Добавляем к dat только действительно новые.
    new_feature_columns = [
        column
        for column in X_train_features.columns
        if column not in non_negative_num_cols
    ]

    X_train_new_features = X_train_features[new_feature_columns].copy()
    X_test_new_features = X_test_features[new_feature_columns].copy()
    X_train_new_features.index = X_train.index
    X_test_new_features.index = X_test.index

    new_features = pd.concat([X_train_new_features, X_test_new_features]).loc[
        data.index
    ]

    return pd.concat([data, new_features], axis=1)


if __name__ == "__main__":
    extended_data = data_extend()
    if extended_data is not None:
        print("Extended data shape:", extended_data.shape)
