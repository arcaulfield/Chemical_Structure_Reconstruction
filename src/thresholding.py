from src.data_processing import clean_mordred_data
from src.multi_output_regressor import run_multi_output_regressor
from sklearn.feature_selection import VarianceThreshold
from src.config import result_path
import numpy as np
import pandas as pd
import os


def find_threshold():
    X, y = clean_mordred_data()

    x_output = run_multi_output_regressor(X, y)
    y_output = np.array([X.shape[1]])

    n = 0.172

    for i in range(2000):
        print("Iteration " + str(i))
        sel = VarianceThreshold(threshold=n)
        train_x = sel.fit_transform(X)
        print("Removed descriptors with variance less than " + str(n) + ". X has size: " + str(train_x.shape[1]))
        x_output = np.vstack((x_output, run_multi_output_regressor(train_x, y)))
        y_output = np.vstack((y_output, train_x.shape[1]))
        n = n + 0.001

    pd.DataFrame(x_output).to_csv(os.path.join(result_path, "x_output.csv"))
    pd.DataFrame(y_output).to_csv(os.path.join(result_path, "y_output.csv"))


if __name__ == '__main__':
    find_threshold()