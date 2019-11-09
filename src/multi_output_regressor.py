import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


def run(X, y):

    total_acc = 0

    kf = KFold(n_splits=5)
    for i, (train_index, valid_index) in enumerate(kf.split(X)):
        x_train, x_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        # Train classifier
        lr = LinearRegression()
        mor = MultiOutputRegressor(lr)
        mor.fit(x_train, y_train)
        y_pred = mor.predict(x_valid)

        acc = accuracy_score(y_valid, y_pred)
        print(f"Iteration {i+1}: Accuracy = {acc * 100}%")
        total_acc = total_acc + acc

    print(f"Average accuracy = {total_acc/kf.get_n_splits()}")


def accuracy_score(y_valid, y_pred):
    dist = np.linalg.norm(y_valid - y_pred, axis=1)
    return dist


