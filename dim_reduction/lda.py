from dataclasses import dataclass
from loguru import logger
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@dataclass
class DatasetForLDA:
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray

def get_dataset(verbose=False):
    # Load the Iris dataset
    logger.info("Loading dataset...")
    sample_size = 10

    data = load_iris()
    # ignore class except 0 and 1
    data.data = data.data[data.target != 2]
    data.target = data.target[data.target != 2]

    # split data
    train_X, test_X, train_y, test_y = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    dataset = DatasetForLDA(train_X, train_y, test_X, test_y)

    if verbose:
        logger.info("Total samples: {}".format(len(train_X) + len(test_X)))
        logger.info("Sample features (first {} rows):\n{}".format(sample_size, train_X[:sample_size]))
        logger.info("Sample labels (first {} entries):\n{}".format(sample_size, train_y[:sample_size]))
    return dataset

def lda_from_scratch(dataset: DatasetForLDA):
    class_1_x = dataset.train_X[dataset.train_y == 0]
    class_2_x = dataset.train_X[dataset.train_y == 1]

    mu_1 = np.mean(class_1_x, axis=0)
    mu_2 = np.mean(class_2_x, axis=0)

    # lda assumption: covariance matrix is the same for both classes
    var_mat_1 = np.cov(class_1_x.T)
    class_1_x_centered = class_1_x - mu_1
    np.testing.assert_allclose(var_mat_1, class_1_x_centered.T @ class_1_x_centered)
    var_mat_2 = np.cov(class_2_x.T)
    
    pass

def train_lda_from_sklearn(dataset: DatasetForLDA):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataset.train_X, dataset.train_y, test_size=0.3, random_state=42)

    # Initialize the LDA model
    lda = LinearDiscriminantAnalysis()

    # Fit the model on the training data
    lda.fit(X_train, y_train)

    # Predict on the test data
    y_pred = lda.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of LDA on Iris dataset: {accuracy:.2f}")


if __name__ == "__main__":
    dataset = get_dataset(verbose=True)
    lda_from_scratch(dataset)
    # train_lda_from_sklearn(dataset.train_X, dataset.train_y)
