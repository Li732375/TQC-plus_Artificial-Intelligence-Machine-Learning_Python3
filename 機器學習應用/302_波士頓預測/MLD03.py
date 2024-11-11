#from sklearn import datasets
#from sklearn.model_selection import cross_val_predict
#from sklearn import linear_model
# TODO
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data.T, ['CRIM','ZN','INDUS','CHAS','NOX','RM' ,'AGE','DIS','RAD','TAX', 'PTRATIO','B','LSTAT']) #有13個feature
# TODO
# MEDV即預測目標向量
# TODO
X = df[['CRIM','ZN','INDUS','CHAS','NOX','RM' ,'AGE','DIS','RAD','TAX', 'PTRATIO','B','LSTAT']]
y = df['MEDV']

#分出20%的資料作為test set
# TODO


#Fit linear model 配適線性模型


# TODO
print('MAE:'              )
print('MSE:'              )
print('RMSE:'             )

#  ([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90 , 4.98]])
prediction = lm.predict(X_new)
print(         )
