import numpy as np

def practice_array():
    Y = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    )
    print(Y.shape)

    for k in range(Y.shape[1]):
        print(f"Y[:, {k}] = {Y[:, k]}")  # column vector
        print(f"Y[{k}, :] = {Y[k, :]}")  # row vector

if __name__ == "__main__":
    practice_array()
