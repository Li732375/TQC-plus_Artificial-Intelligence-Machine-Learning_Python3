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
from sklearn.metrics import r2_score

NBApoints_linear_model_predict_result = NBApoints_linear_model.predict(train_X)

r_squared = r2_score(NBApoints_data["3P"], 
                     NBApoints_linear_model_predict_result)
print(f"R_squared值 = {r_squared:.4f}")

# 計算 P 值是否小於 0.05
f_statistic, p_value = f_regression(train_X, NBApoints_data["3P"])
# =============================================================================
# f_regression 返回兩個值：
# F 值: 每個特徵的 F 統計量，數值越大代表特徵與目標變數的相關性越強。
# p 值: 每個特徵對應的 p 值，表示檢驗結果的顯著性，p 值越小越有統計學意義（通常認為 p 值 < 0.05 為顯著相關）。
# =============================================================================
print(f"f_regresstion P值= {['Y' if p < 0.05 else 'N' for p in p_value]}")
