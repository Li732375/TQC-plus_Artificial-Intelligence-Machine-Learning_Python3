# #############################################################################
# 本題參數設定，請勿更改
seed = 0   # 亂數種子數  
# #############################################################################
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 讀取台北市房價資料集
data = pd.read_csv('Taipei_house.csv')

# 對"行政區"進行 One-Hot encoding
data = pd.get_dummies(data, columns = ['行政區'])

# 處理"車位類別"：將"無"修改為0，其餘為1
data['車位類別'] = data['車位類別'].apply(lambda x: 0 if x == '無' else 1)

# 定義計算 Adjusted R-squared 函數
def adj_R2(r2, n, k):
    return r2 - (k - 1) / (n - k) * (1 - r2)

# 選取特徵和目標變數
features = ['土地面積', '建物總面積', '屋齡', '樓層', '總樓層', '用途', 
            '房數', '廳數', '衛數', '電梯', '車位類別', 
            '行政區_信義區', '行政區_大安區', '行政區_文山區','行政區_松山區']
target = '總價'

X = data[features]
y = data[target]

# 切分訓練集(80%)和測試集(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = seed)

# 初始化效能評估數據框
evaluation = pd.DataFrame(columns = ['Model', 'RMSE (train)', 'RMSE (test)', 
                                   'adj. R2 (train)', 'adj. R2 (test)'])

# 複迴歸
lm = LinearRegression()
lm.fit(X_train, y_train)

evaluation = pd.concat([evaluation, pd.DataFrame({
    'Model': ['Linear Regression'],
    'RMSE (train)': [np.sqrt(mean_squared_error(y_train, lm.predict(X_train)))],
    'RMSE (test)': [np.sqrt(mean_squared_error(y_test, lm.predict(X_test)))],
    'adj. R2 (train)': [adj_R2(lm.score(X_train, y_train), X_train.shape[0], X_train.shape[1])],
    'adj. R2 (test)': [adj_R2(lm.score(X_test, y_test), X_test.shape[0], X_test.shape[1])]
})], ignore_index = True)

# 脊迴歸（Ridge Regression）
ridge = Ridge(alpha = 10)
ridge.fit(X_train, y_train)

evaluation = pd.concat([evaluation, pd.DataFrame({
    'Model': ['Ridge Regression'],
    'RMSE (train)': [np.sqrt(mean_squared_error(y_train, ridge.predict(X_train)))],
    'RMSE (test)': [np.sqrt(mean_squared_error(y_test, ridge.predict(X_test)))],
    'adj. R2 (train)': [adj_R2(ridge.score(X_train, y_train), X_train.shape[0], X_train.shape[1])],
    'adj. R2 (test)': [adj_R2(ridge.score(X_test, y_test), X_test.shape[0], X_test.shape[1])]
})], ignore_index = True)

# 多項式迴歸
poly = PolynomialFeatures(degree = 2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

evaluation = pd.concat([evaluation, pd.DataFrame({
    'Model': ['Polynomial Regression'],
    'RMSE (train)': [(mean_squared_error(y_train, poly_reg.predict(X_train_poly))) **0.5],
    'RMSE (test)': [(mean_squared_error(y_test, poly_reg.predict(X_test_poly))) **0.5],
    'adj. R2 (train)': [adj_R2(poly_reg.score(X_train_poly, y_train), X_train_poly.shape[0], X_train_poly.shape[1])],
    'adj. R2 (test)': [adj_R2(poly_reg.score(X_test_poly, y_test), X_test_poly.shape[0], X_test_poly.shape[1])]
})], ignore_index = True)

# 多項式迴歸 + Lasso
lasso_poly = Lasso(alpha = 10)
lasso_poly.fit(X_train_poly, y_train)

evaluation = pd.concat([evaluation, pd.DataFrame({
    'Model': ['Polynomial + Lasso Regression'],
    'RMSE (train)': [np.sqrt(mean_squared_error(y_train, lasso_poly.predict(X_train_poly)))],
    'RMSE (test)': [mean_squared_error(y_test, lasso_poly.predict(X_test_poly)) **0.5],
    'adj. R2 (train)': [adj_R2(lasso_poly.score(X_train_poly, y_train), X_train_poly.shape[0], X_train_poly.shape[1])],
    'adj. R2 (test)': [adj_R2(lasso_poly.score(X_test_poly, y_test), X_test_poly.shape[0], X_test_poly.shape[1])]
})], ignore_index = True)

# 印出結果
#print(f"訓練集 Adjusted R-squared: \n{evaluation['adj. R2 (train)']}")
print(f"對訓練集的最大 Adjusted R-squared: {max(evaluation['adj. R2 (train)']):.4f}")
#print(f"測試集 RMSE: \n{evaluation['RMSE (test)']}")
print(f"對測試集的最小 RMSE: {int(min(evaluation['RMSE (test)']))}")
# =============================================================================
# 或者
# print(f"對測試集的最小 RMSE: {min(evaluation['RMSE (test)']):0.f}")
# 
# 都會在容許範圍內
# =============================================================================
print(f"複迴歸與脊迴歸對測試集的最大 Adjusted R-squared: {max(evaluation.loc[evaluation['Model'].isin(['Linear Regression', 'Ridge Regression']), 'adj. R2 (test)']):.4f}")

# 預測房價
best_model = evaluation.loc[evaluation['adj. R2 (test)'].idxmax(), 'Model']
if best_model == 'Polynomial Regression':
    X_full = poly.fit_transform(X)
    model = poly_reg
elif best_model == 'Polynomial + Lasso Regression':
    X_full = poly.fit_transform(X)
    model = lasso_poly
elif best_model == 'Ridge Regression':
    X_full = X
    model = ridge
else:
    X_full = X
    model = lm

model.fit(X_full, y)

new_house = pd.DataFrame({
    '土地面積': [36], '建物總面積': [99], '屋齡': [32], '樓層': [4], 
    '總樓層': [4], '用途': [0], '房數': [3], '廳數': [2], '衛數': [1], 
    '電梯': [0], '車位類別': [0], '行政區_信義區': [0], '行政區_大安區': [0],
    '行政區_文山區': [0], '行政區_松山區': [1]
})

if best_model in ['Polynomial Regression', 'Polynomial + Lasso Regression']:
    new_house_poly = poly.transform(new_house)
    price_pred = model.predict(new_house_poly)
else:
    price_pred = model.predict(new_house)

print(f'預測房價: {int(price_pred)} 萬元')
