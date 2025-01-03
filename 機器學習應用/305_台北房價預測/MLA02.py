# #############################################################################
# 本題參數設定，請勿更改
seed = 0   # 亂數種子數  
# #############################################################################
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd 

# TODO

# 讀取台北市房價資料集
# TODO
data = pd.read_csv('Taipei_house.csv')
print(data)
print(data.columns)

# 對"行政區"進行 one-hot encoding
# TODO
data = pd.get_dummies(data, columns = ['行政區'])
print(data)

# 處理"車位類別"
# TODO
print(data.loc[:10, '車位類別'])
data['車位類別'] = [0 if i == '無' else 1 for i in data['車位類別']]
print(data.loc[:10, '車位類別'])

print(data['用途'])


# 計算 Adjusted R-squared
def adj_R2(r2, n, k):
    """ 函式描述：計算 Adjusted R-squared
    參數：
        r2:R-squared 數值
        n: 樣本數
        k: 特徵數

    回傳：
        Adjusted R-squared
    """
    return r2-(k-1)/(n-k)*(1-r2)

# TODO
from sklearn.model_selection import train_test_split

# 切分訓練集(80%)、測試集(20%)
features= ['土地面積', '建物總面積', '屋齡', '樓層', '總樓層', '用途', 
           '房數', '廳數', '衛數', '電梯', '車位類別', 
           '行政區_信義區', '行政區_大安區', '行政區_文山區','行政區_松山區']
target = '總價'  
# TODO
X_tr, X_te, y_tr, y_te = train_test_split(data[features], data[target], 
                                          train_size = 0.8, 
                                          random_state = seed)

# 複迴歸(參數皆為預設值)
# #########################################################################
# '行政區_信義區', '行政區_大安區', '行政區_文山區','行政區_松山區' 四個特徵是經過
# one-hot encoding 後產生，若欄位名稱不同可自行修改之。
# #########################################################################
from sklearn import linear_model
# TODO
cl1 = linear_model.LinearRegression()

# 脊迴歸(Ridge regression)，除以下參數設定外，其餘為預設值
# #########################################################################
# alpha=10
# #########################################################################
# TODO
cl2 = linear_model.Ridge(alpha=10)

# 多項式迴歸，除以下參數設定外，其餘為預設值
# #########################################################################
# degree=2
# #########################################################################
from sklearn.preprocessing import PolynomialFeatures
# TODO
p = PolynomialFeatures(degree=2)
cl3 = linear_model.LinearRegression()

# 多項式迴歸 + Lasso迴歸，除以下參數設定外，其餘為預設值
# #########################################################################
# alpha=10
# #########################################################################
# TODO
cl4 = linear_model.Lasso(alpha=10)

from sklearn.metrics import r2_score, mean_squared_error

evaluation = pd.DataFrame(columns = ['adj. R2 (train)', 'RMSE (test)', 
                                     'adj. R2 (test)'])
#print(evaluation)
for m in [cl1, cl2, cl3, cl4]:
    if m == cl3:
        X_tr, X_te = p.fit_transform(X_tr), p.fit_transform(X_te)
        
    m.fit(X_tr, y_tr)
    evaluation.loc[len(evaluation)] = [adj_R2(r2_score(y_tr, m.predict(X_tr)), 
                                                       X_tr.shape[0],
                                                       X_tr.shape[1]),
                                       mean_squared_error(y_te, m.predict(X_te))** 0.5,
                                       adj_R2(r2_score(y_te, m.predict(X_te)), 
                                                       X_te.shape[0],
                                                       X_te.shape[1])]
    
print(evaluation)

print('對訓練集的最大 Adjusted R-squared: %.4f' % max(evaluation['adj. R2 (train)']))
print('對測試集的最小 RMSE:%d' % round(min(evaluation['RMSE (test)'])))
print('兩個模型對測試集的最大 Adjusted R-squared: %.4f' % 
      max(evaluation.loc[:1, 'adj. R2 (test)']))

''' 預測 '''
# 利用所有資料重新擬合模型，並進行預測
# TODO
print('對測試集的最大 Adjusted R-squared: %.4f' % 
      max(evaluation['adj. R2 (test)']))

max_r2 = max(evaluation['adj. R2 (test)'])
print(evaluation[evaluation['adj. R2 (test)'] == max_r2])
print(evaluation[evaluation['adj. R2 (test)'] == max_r2].index)
print('對測試集的最大 Adjusted R-squared 模型: %d' % 
      evaluation[evaluation['adj. R2 (test)'] == max_r2].index[0])

cl5 = linear_model.Lasso(alpha=10)
cl5.fit(p.fit_transform(data[features]), data[target])
p = cl5.predict(p.transform([[36,99,32,4,4,0,3,2,1,0,0,0,0,0,1]]))
print(p)
print(type(p))
print(*p) # 解開最外層 []

print('predict price ', round(*p)) # 要求先行解開 []
print('predict price ', int(p)) # 僅單一值會自行提取 []
