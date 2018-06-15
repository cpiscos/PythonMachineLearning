import numpy as np

class LogitReg():
    def __init__(self, X, y, alpha, n_iter):
        self.mu = []
        self.std = []
        self.X = self.transform(X)
        self.y = y.reshape(y.size, 1)
        self.alpha = alpha
        self.n_iter = n_iter


    def fit(self):
        self.w_ = np.random.randn(1, self.X.shape[1])
        self.cost = []
        for _ in range(self.n_iter):
            hyp = np.dot(self.X, self.w_.T)
            self.log_func = 1 / (1 + np.exp(-hyp))
            self.w_ -= np.transpose(self.alpha * self.X.T.dot((self.log_func - self.y))/self.X.shape[0])
            cost_func = (-self.y.T.dot(np.log(self.log_func)) - (1-self.y).T.dot(np.log(1-self.log_func))) / self.y.shape[0]
            self.cost.append(cost_func.reshape(1))

    def predict(self, array):
        Y = np.array(array).reshape(1, 4)
        pred = self.transform(Y).dot(self.w_.T)

    def transform(self, X):
        X = np.concatenate((X, np.square(X)), axis=1)
        if len(self.mu) == 0:
            for j in range(X.shape[1]):
                self.mu.append(X[:,j].mean())
                self.std.append(X[:,j].std())

        for j in range(X.shape[1]):
            X[:,j] = (X[:,j] - self.mu[j]) / self.std[j]
        return np.concatenate((np.ones((X.shape[0], 1), dtype=float), X), axis=1)