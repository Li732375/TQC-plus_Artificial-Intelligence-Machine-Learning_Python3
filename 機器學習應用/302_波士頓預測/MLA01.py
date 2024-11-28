import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

# 1. 載入波士頓房價資料集
boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)

# 新增目標變數 MEDV 到資料集
df['MEDV'] = boston.target

# 2. 選擇特徵和目標變數
X = df[['CRIM','ZN','INDUS','CHAS','NOX','RM' ,'AGE','DIS','RAD','TAX', 
        'PTRATIO','B','LSTAT']]
y = df['MEDV']

# 3. 分割訓練集和測試集，測試集占20%，random_state=1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 1)

# 4. 建立並訓練線性迴歸模型
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# 5. 預測測試集的房價
y_pred = lm.predict(X_test)

# 計算平均絕對誤差（MAE）、均方誤差（MSE）、均方根誤差（RMSE）
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 印出結果（四捨五入至小數點後四位）
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')

# 6. 根據輸入資料進行房價預測
X_new = pd.DataFrame([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 
                   296.0, 15.30, 396.90, 4.98]], columns = X.columns)

# 輸出預測結果（四捨五入至小數點後四位）
print(f"預測房價: {lm.predict(X_new)[0]:.4f}")
