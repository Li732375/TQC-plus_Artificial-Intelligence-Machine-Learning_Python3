#from sklearn import datasets
#from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
# TODO
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
# TODO
# MEDV即預測目標向量
# TODO
X = pd.DataFrame(boston.data, columns = boston.feature_names) #有13個feature
y = pd.DataFrame(boston.target, columns = ['MEDV'])
print(X)
print(y)

#分出20%的資料作為test set
# TODO
from sklearn.model_selection import train_test_split
tr_x, te_x, tr_y, te_y = train_test_split(X, y, test_size = 0.2, 
                                          random_state = 1)

#Fit linear model 配適線性模型
lm = linear_model.LinearRegression().fit(tr_x, tr_y)
p_y = lm.predict(te_x)

# TODO
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(te_y, p_y)
mse = mean_squared_error(te_y, p_y)

print('MAE:', round(mae, 4))
print('MSE:', round(mse, 4))
print('RMSE:', round(mse ** 0.5, 4))

#  ([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90 , 4.98]])
X_new = [[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 
          15.30, 396.90 , 4.98]]
prediction = lm.predict(X_new)[0][0]
print(prediction)
print('X_new', round(prediction, 4))
