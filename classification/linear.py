import pandas as pd

from sklearn import linear_model


def load_data():
    df = pd.read_csv("data/train.csv")
    return df

def main():
    
    df = load_data()
    print(df.head())


if __name__ == "__main__":
    main()
