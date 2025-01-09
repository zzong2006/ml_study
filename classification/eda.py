import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from loguru import logger

from data import load_data


def main():
    dataset = load_data()

    # 총주차대수 별 허위매물여부를 scatter plot으로 그리기
    target_column = "월세"
    parking_count = dataset.X[target_column]
    # remove outliers (by z-score)
    z_score = stats.zscore(parking_count)

    plt.scatter(np.log(parking_count), dataset.y)
    plt.show()


if __name__ == "__main__":
    main()
