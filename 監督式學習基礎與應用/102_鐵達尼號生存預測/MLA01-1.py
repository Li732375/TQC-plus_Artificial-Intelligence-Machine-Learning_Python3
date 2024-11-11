import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model

# 原始資料
titanic = pd.read_csv("titanic.csv")
df = titanic.copy()

# 將年齡的空值填入年齡的中位數
df.info()
age_median = df.Age.median()
df['Age'] = df['Age'].fillna(age_median) # 更新寫法
#df.fillna({'Age':age_median}, inplace = True) # 更新寫法二
#df.Age.fillna(age_median, inplace = True) # 原先寫法
# =============================================================================
# FutureWarning: A value is trying to be set on a copy of a DataFrame or Series 
# through chained assignment using an inplace method.
# 
# The behavior will change in pandas 3.0. This inplace method will never work 
# because the intermediate object on which we are setting values always behaves 
# as a copy.
# 
# For example, when doing 'df[col].method(value, inplace=True)', try using 
# 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) 
# instead, to perform the operation inplace on the original object.
# 
#   df.Age.fillna(age_median, inplace = True)
# =============================================================================
df.info()
print(f"年齡中位數= {age_median}")

# 轉換欄位值成為數值
le = preprocessing.LabelEncoder()
df.PClass = le.fit_transform(df.PClass)

X = df.loc[:, ['PClass', 'Age', 'SexCode']]
y = df.loc[:, 'Survived']

# 建立模型
model = linear_model.LogisticRegression()
model.fit(X, y)
intercept = model.intercept_
coef = model.coef_
print(f'截距= {intercept[0]:.4f}') # 這裡確定會與答案(1.9966)不符
print(f'迴歸係數= {coef[0][2]:.4f}') # 這裡確定會與答案(2.3834)有緩衝誤差


# 混淆矩陣(Confusion Matrix)，計算準確度
accuracy = model.score(X, y)
print(f'accuracy: {accuracy:.4f}')