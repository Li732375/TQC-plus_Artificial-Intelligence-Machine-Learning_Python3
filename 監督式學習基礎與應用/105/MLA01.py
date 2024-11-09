import pandas as pd
#import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

NBApoints_data= pd.read_csv("NBApoints.csv")
#TODO
from sklearn import preprocessing

label_encoder_conver = preprocessing.LabelEncoder()
Pos_encoder_value = label_encoder_conver.fit_transform(NBApoints_data['Pos'])
#print(Pos_encoder_value)
#print("\n")

label_encoder_conver = preprocessing.LabelEncoder()
Tm_encoder_value = label_encoder_conver.fit_transform(NBApoints_data['Tm'])
#print(Tm_encoder_value)

# 準備訓練數據
encoded_features = [Pos_encoder_value, NBApoints_data['Age'], 
                    Tm_encoder_value]
feature_names = ['Pos_encoded', 'Age', 'Tm_encoded']
train_X = pd.DataFrame(encoded_features, feature_names).T # 轉置

# 建立線性迴歸模型並進行訓練
NBApoints_linear_model = LinearRegression()
NBApoints_linear_model.fit(train_X, NBApoints_data["3P"]) # 題目說明未提及要預測的欄位 "3P"

# 預測三分球得分
test_X = pd.DataFrame([5, 28, 10], feature_names).T
NBApoints_linear_model_predict_result = NBApoints_linear_model.predict(test_X)
print(f"三分球得球數= {NBApoints_linear_model_predict_result[0]:.4f}")

# 計算 R-squared 和 MSE 值
from sklearn.metrics import mean_squared_error, r2_score

NBApoints_linear_model_predict_result = NBApoints_linear_model.predict(train_X)

r_squared = r2_score(NBApoints_data["3P"], 
                     NBApoints_linear_model_predict_result)
print(f"R_squared值 = {r_squared:.4f}")

mse = mean_squared_error(NBApoints_data["3P"], 
                         NBApoints_linear_model_predict_result)
print(f"Mean Squared Error (MSE) = {mse:.4f}")

# 計算 P 值是否小於 0.05
f_statistic, p_value = f_regression(train_X, NBApoints_data["3P"])
print(f"f_regresstion P值= {['Y' if p < 0.05 else 'N' for p in p_value]}")
