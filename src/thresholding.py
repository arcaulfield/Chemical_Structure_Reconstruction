from src.data_processing import clean_mordred_data
from src.multi_output_regressor import run_multi_output_regressor
from sklearn.feature_selection import VarianceThreshold
import numpy as np


def find_threshold():
    X, y = clean_mordred_data()

    run_multi_output_regressor(X, y)
    n = 0.01

    for i in range(1000):
        print("Iteration " + str(i))
        sel = VarianceThreshold(threshold=n)
        train_x = sel.fit_transform(X)
        print("Removed descriptors with variance less than " + str(n) + ". X has size: " + str(train_x.shape[1]))
        run_multi_output_regressor(train_x, y)
        n = n + 0.001


if __name__ == '__main__':
    find_threshold()