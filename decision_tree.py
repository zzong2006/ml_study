"""
- implementation
    - https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/algorithms/decisiontree/decision_tree.py
    - https://www.geeksforgeeks.org/decision-tree-implementation-python/
- explanation: https://ratsgo.github.io/machine%20learning/2017/03/26/tree/

"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

class CustomDecisionTree:
    def __init__(self) -> None:
        pass

    def train(self):
        pass

train_data = np.loadtxt("data/data.txt", delimiter=",")
train_y = np.loadtxt("data/targets.txt")

if __name__ == "__main__":
    print(train_data)
    print(train_y)