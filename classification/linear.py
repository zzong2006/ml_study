import pandas as pd
from dataclasses import dataclass

from sklearn import linear_model
from sklearn.compose import ColumnTransformer

from loguru import logger
from data import load_data


def main():
    train_dataset = load_data(is_train=True)
    test_dataset = load_data(is_train=False, train_dataset=train_dataset)

    model = linear_model.LassoCV()
    estimator = model.fit(train_dataset.X, train_dataset.y)

    coef2column = {
        coef: column
        for coef, column in zip(estimator.coef_, train_dataset.X.columns)
        if coef != 0
    }
    for idx, (coef, column) in enumerate(
        sorted(coef2column.items(), key=lambda x: abs(x[0]), reverse=True)
    ):
        print(f"{idx + 1}. {column}: {coef}")

    # Predict on the test dataset
    print(estimator.predict(test_dataset.X))


if __name__ == "__main__":
    main()
