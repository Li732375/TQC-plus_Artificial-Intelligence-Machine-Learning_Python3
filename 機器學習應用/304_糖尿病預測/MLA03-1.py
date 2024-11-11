from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
data = datasets.load_diabetes()
#get x
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
#Total number of examples
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y, y_pred)
r2 = model.score(X, y)
print('Total number of examples')
print(f'MSE= {MSE:.4f}')
print(f'R-squared= {r2:.4f}')
#3:1 100
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=100)
model2=LinearRegression()
model2.fit(X_train, y_train)
train_pred = model2.predict(X_train)
test_pred = model2.predict(X_test)
train_MSE = mean_squared_error(y_train, train_pred)
test_MSE = mean_squared_error(y_test, test_pred)
print('Split 3:1')
print(f'train MSE= {train_MSE:.4f}')
print(f'test MSE= {test_MSE:.4f}')