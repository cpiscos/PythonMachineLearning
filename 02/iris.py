import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adaline import *

# Import data and change setosa to -1 & vice-versa
df = pd.read_csv('../Resources/iris.data', header=None)
y = df.iloc[:5, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

# Extract sepal length[0] and petal length[2]
X = df.iloc[:5, [0, 2]].values
# plot
# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# # plt.legend(loc='upper left')
# plt.show()
#
ada = Adaline(eta=0.0001, n_iter=1)
ada.fit(X, y)
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
# print(ada.w_)
# print('sample set: ', X.shape[1])

# a = np.arange(10).reshape(5, 2)
# b = np.arange(5)
# print(a)
# print(b)
# print(a.T.dot(b))
# print(a[:, 1].T.dot(b))
# print(a[:, 0].T.dot(b))
