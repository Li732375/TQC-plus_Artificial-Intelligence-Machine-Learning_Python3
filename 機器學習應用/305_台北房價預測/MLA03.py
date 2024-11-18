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
#print(data)
df = data.copy()
#print(df.columns)
#print(df.loc[:15, '行政區'])
#print(df['車位類別'])
# 對"行政區"進行 one-hot encoding
# TODO
df = pd.get_dummies(df, columns = ['行政區'])
#print(df.columns)
# =============================================================================
# 原先欄位 '行政區' 下的資料類別會分別成為新的欄位名稱，依據資料原先數值設定為 1，其
# 餘為 0。
# =============================================================================

# 處理"車位類別"
# TODO
df['車位類別'] = df['車位類別'].apply(lambda x: 0 if x == '無' else 1)
#print(df['車位類別'])

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


# 切分訓練集(80%)、測試集(20%)
features= ['土地面積', '建物總面積', '屋齡', '樓層', '總樓層', '用途', 
           '房數', '廳數', '衛數', '電梯', '車位類別', 
           '行政區_信義區', '行政區_大安區', '行政區_文山區','行政區_松山區']
target = '總價'  
# TODO
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], 
                                                    test_size = 0.2, 
                                                    random_state = seed)
# 這裡切分完後物件類別不再是 Dataframe，而是 numpy.ndarray

# 複迴歸(參數皆為預設值)
# #########################################################################
# '行政區_信義區', '行政區_大安區', '行政區_文山區','行政區_松山區' 四個特徵是經過
# one-hot encoding 後產生，若欄位名稱不同可自行修改之。
# #########################################################################
from sklearn import linear_model
# TODO
ml1 = linear_model.LinearRegression()

# 脊迴歸(Ridge regression)，除以下參數設定外，其餘為預設值
# #########################################################################
# alpha=10
# #########################################################################
# TODO
ml2 = linear_model.Ridge(alpha = 10)

# 多項式迴歸，除以下參數設定外，其餘為預設值
# #########################################################################
# degree=2
# #########################################################################
from sklearn.preprocessing import PolynomialFeatures
# TODO
p = PolynomialFeatures(degree = 2)
ml3 = linear_model.LinearRegression()

# 多項式迴歸 + Lasso迴歸，除以下參數設定外，其餘為預設值
# #########################################################################
# alpha=10
# #########################################################################
from sklearn.linear_model import Lasso
# TODO
ml4 = Lasso(alpha = 10)


evaluation = pd.DataFrame(columns = ['model', 'adj. R2 (train)', 'RMSE (test)', 
                                     'adj. R2 (test)'])

from sklearn.metrics import mean_squared_error

for m in [ml1, ml2, ml3, ml4]:
    if m == ml3:
        X_train, X_test = p.fit_transform(X_train), p.fit_transform(X_test)
    
    m.fit(X_train, y_train)
    evaluation.loc[len(evaluation)] = [m, 
                                       round(adj_R2(m.score(X_train, y_train), 
                                                    len(X_train), 
                                                    X_train.shape[1]), 4),
                                       int(mean_squared_error(y_test, 
                                                              m.predict(X_test)) ** 0.5), 
                                       round(adj_R2(m.score(X_test, y_test), 
                                                    len(X_test), 
                                                    X_test.shape[1]), 4)]

print('對訓練集的最大 Adjusted R-squared: %.4f' % max(evaluation['adj. R2 (train)']))
print('對測試集的最小 RMSE:%d' % min(evaluation['RMSE (test)']))
print('兩個模型對測試集的最大 Adjusted R-squared: %.4f' % 
      max(evaluation.loc[:1, 'adj. R2 (test)']))

''' 預測 '''
# 利用所有資料重新擬合模型，並進行預測
# TODO
print('Max Adjusted R-squared:', max(evaluation['adj. R2 (test)']))

#features= ['土地面積', '建物總面積', '屋齡', '樓層', '總樓層', '用途', 
#           '房數', '廳數', '衛數', '電梯', '車位類別', 
#           '行政區_信義區', '行政區_大安區', '行政區_文山區','行政區_松山區']

# TODO
#print(df['電梯'])
ml4 = Lasso(alpha = 10)
ml4.fit(p.fit_transform(df[features]), df[target])
pre = ml4.predict(p.transform([[36,99,32,4,4,0,3,2,1,0,0,0,0,0,1]]))
print('predict price', int(pre))

