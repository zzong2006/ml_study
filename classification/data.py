import pandas as pd
from dataclasses import dataclass

from loguru import logger


@dataclass
class ClassificationDataset:
    X: pd.DataFrame
    y: list[int]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]


def load_data(
    label_column: str = "허위매물여부",
    unused_columns: list[str] = ["ID", "전용면적", "총주차대수", "총층"],
    preprocess: bool = True,
    is_train: bool = True,
    train_dataset: ClassificationDataset = None,
) -> ClassificationDataset:
    if train_dataset is None and not is_train and preprocess:
        raise ValueError("train_dataset must be provided for normalization when test dataset is provided")

    df = pd.read_csv(
        "data/train.csv" if is_train else "data/test.csv",
        dtype={
            "매물확인방식": "category",
            "보증금": "float",
            "월세": "int",
            "전용면적": "float",
            "해당층": "float",
            "총층": "float",
            "방향": "category",
            "방수": "category",
            "욕실수": "float",
            "주차가능여부": "category",
            "총주차대수": "float",
            "관리비": "int",
            "중개사무소": "category",
            "제공플랫폼": "category",
            "게재일": "category",
        },
    )
    df = df.drop(columns=unused_columns, errors="ignore")
    # make sure no nan in label column
    if is_train:
        assert df[label_column].isna().sum() == 0
        y = df[label_column].to_list()
        X = df.drop(columns=[label_column])
    else:
        X = df
        y = []
    logger.debug(f"X.shape: {X.shape}")

    # `게재일`의 day 값을 삭제 (2024-01-01 -> 2024-01), convert to category
    X["게재일"] = X["게재일"].apply(func=lambda x: x[:7]).astype("category")
    categorical_columns = X.select_dtypes(include=["category"]).columns

    # count unique values in each categorical column
    logger.debug("categorical columns:")
    for column in categorical_columns:
        logger.debug(f"{column}: {X[column].nunique()}")

    # count nan in each column
    logger.debug("### `nan` count in each column:")
    for column in X.columns:
        logger.debug(f"{column}: {X[column].isna().sum()}")

    if preprocess:
        numeric_columns = X.select_dtypes(include=["number"]).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
        # normalize numeric columns
        if is_train:
            numeric_mean = X[numeric_columns].mean()
            numeric_std = X[numeric_columns].std()
        else:
            numeric_mean = train_dataset.numeric_mean
            numeric_std = train_dataset.numeric_std
        X[numeric_columns] = (X[numeric_columns] - numeric_mean) / numeric_std
        X = pd.get_dummies(X, drop_first=True, columns=categorical_columns)

    logger.debug(X.head())

    return ClassificationDataset(X, y, numeric_mean, numeric_std)

