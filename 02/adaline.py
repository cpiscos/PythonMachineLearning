import numpy as np


class Adaline(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.1, size=1 + X.shape[1])

        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            print('y: ', y)
            print('net_input: ', net_input)
            errors = (y-net_input)
            print('errors; ', errors)
            print('weights: ', self.w_[1:])
            print('delta', self.eta * (X.T.dot(errors)))
            self.w_[1:] += self.eta * (X.T.dot(errors))
            print('X: ', X)
            print('X shape: ', X.shape)
            print('X.T: ', X.T)
            print('X.T.dot(errors)', X.T.dot(errors))
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2
            self.cost_.append(cost)
        return self

    def predict(self, X):
        a = np.where(self.net_input(X) >= 0, 1, -1)
        print('prediction', a)
        return a

    def net_input(self, X):
        b = np.dot(X, self.w_[1:]) + self.w_[0]
        # print('sample:', X)
        print('dot: ', np.dot(X, self.w_[1:]))
        print('b: ', b)
        return b
