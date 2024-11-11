from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 讀取糖尿病資料集
diabetes = datasets.load_diabetes()

# X 和 Y
X = diabetes.data
y = diabetes.target

# A. 不分割資料集的情況下
# 建立線性回歸模型
lm = LinearRegression()
lm.fit(X, y)

# 預測結果
y_pred = lm.predict(X)

# 計算均方誤差和決定係數
mse_total = mean_squared_error(y, y_pred)
r2_total = r2_score(y, y_pred)

print('Total number of examples')
print(f'MSE={mse_total:.4f}')
print(f'R-squared={r2_total:.4f}')

# B. 在分割成訓練資料集及測試資料集，其比率為3:1，並設定亂數種子為100的情況下
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                                    random_state = 100)

# 建立線性回歸模型
lm2 = LinearRegression()
lm2.fit(X_train, y_train)

# 預測訓練和測試資料集
y_train_pred = lm2.predict(X_train)
y_test_pred = lm2.predict(X_test)

# 計算均方誤差和決定係數
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print('Split 3:1')
print(f'train MSE={mse_train:.4f}')
print(f'test MSE={mse_test:.4f}')
print(f'train R-squared={r2_train:.4f}')
print(f'test R-squared={r2_test:.4f}')
