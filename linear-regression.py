import sklearn
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.n_features = None
        self.n_samples = None
        self.weights = None
        self.bias = 0
    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        for _ in range(self.epochs):
            self.single_pass(X, y)


    def predict(self, X):
        y_hat = np.dot(X, self.weights) + self.bias
        return y_hat

    def single_pass(self, X, y):

        y_hat = np.apply_along_axis(self.predict, 1, X)
        print("cost: " + str(self.mse(y_hat, y)))
        dms = []
        for i in range(self.n_features):
            dms.append(np.mean(2 * np.dot(X.T ,y_hat - y)))
        dms = np.array(dms)
        self.weights -= self.lr * dms
        self.bias -= self.lr * np.mean((y_hat - y) * 2)

    def mse(self, y_hat, y):
        return np.mean(np.square(y_hat - y))



if __name__ == "__main__":
    x, y = make_regression(n_samples=100, n_features=1, random_state=1, noise=20)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=132)
    model = LinearRegression(epochs=100)
    model.fit(X_train, y_train)
    # y_hat = model.evaluate(X_test[0])
    # print(model.mse(y_hat, y_test[0]))
    y_pred_line = model.predict(x)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(x, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()

