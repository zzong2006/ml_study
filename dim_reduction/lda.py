from loguru import logger
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_dataset(verbose=False):
    # Load the Iris dataset
    logger.info("Loading dataset...")
    sample_size = 10

    data = load_iris()
    X: np.ndarray = data.data
    y: np.ndarray = data.target
    if verbose:
        logger.info(f"Sample features (first {sample_size} rows):\n{X[:sample_size]}")
        logger.info(f"Sample labels (first {sample_size} entries):\n{y[:sample_size]}")
    return X, y


def train_lda_from_sklearn(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
    X, y = get_dataset(verbose=True)
    train_lda_from_sklearn(X, y)