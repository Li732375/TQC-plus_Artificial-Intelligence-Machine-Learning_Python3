import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

NBApoints_data= pd.read_csv("NBApoints.csv")
#TODO


label_encoder_conver = preprocessing.LabelEncoder()
Pos_encoder_value =
print(Pos_encoder_value)
print("\n")

label_encoder_conver = preprocessing.LabelEncoder()
Tm_encoder_value = 
print(Tm_encoder_value)

train_X = pd.DataFrame(         ,       ).T
                        
NBApoints_linear_model = LinearRegression()
NBApoints_linear_model.fit(train_X, NBApoints_data["3P"])

NBApoints_linear_model_predict_result=
print("三分球得球數=",NBApoints_linear_model_predict_result)

r_squared =
print("R_squared值=",r_squared)

print("f_regresstion\n")
print("P值="              )
print("\n")
