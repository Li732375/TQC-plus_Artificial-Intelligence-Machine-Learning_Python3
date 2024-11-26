import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

NBApoints_data = pd.read_csv("NBApoints.csv")
#TODO
print(NBApoints_data)
print(NBApoints_data.columns)
print(NBApoints_data[['3P', '3PA', '3P%']])

df = NBApoints_data.copy()[['Pos', 'Age', 'Tm', '3P']]
print(df)

from sklearn import preprocessing
label_encoder_conver = preprocessing.LabelEncoder()
Pos_encoder_value = label_encoder_conver.fit_transform(df['Pos'])
print(Pos_encoder_value)
print("\n")

label_encoder_conver = preprocessing.LabelEncoder()
Tm_encoder_value = label_encoder_conver.fit_transform(df['Tm'])
print(Tm_encoder_value)
print("\n")

print([Pos_encoder_value, df['Age'], Tm_encoder_value])
train_X = pd.DataFrame([Pos_encoder_value, 
                        df['Age'], 
                        Tm_encoder_value], ['Pos', 'Age', 'Tm']).T
print("\n")
print(train_X)
                        
NBApoints_linear_model = LinearRegression()
NBApoints_linear_model.fit(train_X, df['3P'])

NBApoints_linear_model_predict_result = NBApoints_linear_model.predict([[5, 
                                                                         28, 
                                                                         10]])
print("三分球得球數=", round(NBApoints_linear_model_predict_result[0], 4))

from sklearn.metrics import r2_score
r_squared = r2_score(df['3P'], 
                     NBApoints_linear_model.predict(train_X))
print("R_squared值=", round(r_squared, 4))
                     
ans = f_regression(train_X, df['3P'])
print(ans)
print("f_regresstion\n")
print("P值=", ['Y' if i < 0.05 else 'N' for i in ans[1]])
print("\n")

