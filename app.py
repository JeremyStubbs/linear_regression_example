import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

X = np.array([[1],[2],[3]])
y = np.array([1,2,4])
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
print(reg.coef_)
print(reg.intercept_)
plt.scatter(X, y,color='g')
plt.plot(X, reg.predict(X),color='k')
plt.show()
