import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logit_regression import *

# Import data and change setosa to -1 & vice-versa
df = pd.read_csv('../Resources/iris.data', header=None)
y = df.iloc[:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)

X = df.iloc[:100,:4].values

iris = LogitReg(X, y, alpha=0.01, n_iter=500)
iris.fit()
iris.predict([5.6,3.0,4.5,1.5])
plt.plot(range(1, len(iris.cost) + 1), iris.cost, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
