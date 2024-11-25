import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model

# 原始資料
titanic = pd.read_csv("titanic.csv")
print(titanic.columns)
# TODO
df = titanic.copy()

# 將年齡的空值填入年齡的中位數
# TODO
age_med = df['Age'].median()
print("年齡中位數=", round(age_med))
# TODO
print(df['Age'].isna().sum())
#df['Age'] = round(df['Age'].fillna(age_med)) # 不可取整數，會影響答案
df['Age'] = df['Age'].fillna(age_med)
print(df['Age'].isna().sum())


# 更新後資料
print('new data')
# TODO
print(df[['PClass', 'Age', 'SexCode']])

# 轉換欄位值成為數值
label_encoder = preprocessing.LabelEncoder()
df['PClass_N'] = label_encoder.fit_transform(df[['PClass']])
# TODO


# 建立模型
# TODO
cl = linear_model.LogisticRegression()
cl.fit(df[['PClass_N', 'Age', 'SexCode']], df[['Survived']])

# =============================================================================
# print('截距=', cl.intercept_)
# print('迴歸係數=', cl.coef_)
# =============================================================================
print('截距=', round(cl.intercept_[0], 4))
print('迴歸係數=', round(cl.coef_[0][-1], 4))


# 混淆矩陣(Confusion Matrix)，計算準確度
print('Confusion Matrix', round(cl.score(df[['PClass_N', 'Age', 'SexCode']], 
                                         df[['Survived']]), 4))
# TODO




