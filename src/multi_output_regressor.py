import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from src.data_processing import clean_mordred_data


def run_multi_output_regressor(X, y):
    total_acc = np.zeros(shape=(y.shape[1]))

    kf = KFold(n_splits=5)
    for i, (train_index, valid_index) in enumerate(kf.split(X)):
        x_train, x_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        # Train classifier
        lr = LinearRegression()
        mor = MultiOutputRegressor(lr)
        mor.fit(x_train, y_train)
        y_pred = np.rint(mor.predict(x_valid))

        acc = accuracy_score(y_valid, y_pred)
        print(f"Iteration {i+1}: L1 = {acc}")
        total_acc = total_acc + acc

    print(f"Average accuracy = {total_acc/kf.get_n_splits()}")
    return total_acc


def accuracy_score(y_valid, y_pred):
    dist = np.linalg.norm(y_valid - y_pred, ord=1, axis=0)
    return dist


if __name__ == '__main__':
    X, y = clean_mordred_data()
    run_multi_output_regressor(X, y)