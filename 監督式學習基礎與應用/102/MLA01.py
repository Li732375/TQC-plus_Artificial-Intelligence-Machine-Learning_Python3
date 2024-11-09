import pandas as pd
#import numpy as np
from sklearn import preprocessing, linear_model
#from sklearn.metrics import confusion_matrix

# 原始資料
titanic = pd.read_csv("titanic.csv")
print('raw data')
#print(titanic.head())  # 輸出前幾筆資料作為檢查

# 將年齡的空值填入年齡的中位數
median_age = titanic['Age'].median()  # 計算年齡中位數
titanic['Age'] = titanic['Age'].fillna(median_age) # 修正寫法，且不可擅自改成取整數，會影響答案
#titanic['Age'].fillna(median_age, inplace = True)  # 用中位數填補 NA
# =============================================================================
# FutureWarning: A value is trying to be set on a copy of a DataFrame or Series 
# through chained assignment using an inplace method.
# The behavior will change in pandas 3.0. This inplace method will never work 
# because the intermediate object on which we are setting values always behaves 
# as a copy.
# 
# For example, when doing 'df[col].method(value, inplace=True)', try using 
# 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) 
# instead, to perform the operation inplace on the original object.
#
#   titanic['Age'].fillna(median_age, inplace = True)
# =============================================================================

print(f"年齡中位數 = {median_age:.0f}")

# 更新後資料
print('new data')
#print(titanic.head())  # 再次檢查更新後的資料

# 轉換欄位值成為數值
label_encoder = preprocessing.LabelEncoder()
titanic['PClass_N'] = label_encoder.fit_transform(titanic['PClass']) # 將性別欄位轉為數值

# 建立模型
features = ['PClass_N', 'Age', 'SexCode']  # 選擇用於訓練的特徵欄位
X = titanic[features]
y = titanic['Survived']

log_reg = linear_model.LogisticRegression()
log_reg.fit(X, y)

#print('截距=', log_reg.intercept_)
print(f'截距= {log_reg.intercept_[0]:.4f}') # 這裡確定會與答案(1.9966)不符
print(f"迴歸係數= {log_reg.coef_[0][features.index('SexCode')]:.4f}") # 這裡確定會與答案(2.3834)有緩衝誤差

# =============================================================================
# # 混淆矩陣(Confusion Matrix)，計算準確度
# y_pred = log_reg.predict(X)
# conf_matrix = confusion_matrix(y, y_pred)
# print('Confusion Matrix')
# print(conf_matrix)
# =============================================================================

# 使用 .score() 方法計算準確率
print(f"模型準確率 = {log_reg.score(X, y):.4f} %")