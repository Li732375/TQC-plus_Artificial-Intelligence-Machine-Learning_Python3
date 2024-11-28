from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
# TODO
data = datasets.load_diabetes()

#get x
# TODO 
X = data.data
y = data.target
#print(X)
#print(y)

#Total number of examples
# TODO 
lm1 = LinearRegression()
lm1.fit(X, y)

from sklearn.metrics import mean_squared_error, r2_score
print('Total number of examples')
print('MSE=', round(mean_squared_error(y, lm1.predict(X)), 4))
print('R-squared=', round(r2_score(y, lm1.predict(X)), 4))

#3:1 100
from sklearn.model_selection import train_test_split
xTrain2, xTest2, yTrain2, yTest2 = train_test_split(X, y, train_size = 0.75, 
                                                    random_state = 100)
lm2 = LinearRegression()
lm1.fit(xTrain2, yTrain2)
# TODO 

print('Split 3:1')
print('train MSE=', round(mean_squared_error(yTrain2, lm1.predict(xTrain2)), 4))
print('test MSE=', round(mean_squared_error(yTest2, lm1.predict(xTest2)), 4))
print('train R-squared=', round(r2_score(yTrain2, lm1.predict(xTrain2)), 4))
print('test R-squared=', round(r2_score(yTest2, lm1.predict(xTest2)), 4))
