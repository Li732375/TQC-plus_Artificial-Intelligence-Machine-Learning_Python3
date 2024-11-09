import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt

input_file = 'cardata.txt'

# Reading the data
import pandas as pd

data = pd.read_csv(input_file)
print(data.columns)        

data = pd.read_csv(input_file, header = None)
df = data.copy()
print(df)
# TODO


# Convert string data to numerical data將字串資料轉換為數值資料
# TODO
en_df = pd.DataFrame()
encoders = []

for i in range(len(data.columns)):
    encoder = preprocessing.LabelEncoder()
# =============================================================================
#     fit(df[i])：只學習如何將 df[i] 中的類別值轉換成數字，但不實際執行轉換。
#     fit_transform(df[i])：同時學習編碼規則並轉換資料。
# =============================================================================
    en_df[i] = encoder.fit_transform(df[i])
    encoders.append(encoder)  # Storing encoder for each column

print(en_df)

X = en_df.iloc[:, :-1]
y = en_df.iloc[:, -1]

# Build a Random Forest classifier建立隨機森林分類器
# TODO
cl = RandomForestClassifier(n_estimators = 200, max_depth = 8, 
                            random_state = 7)
cl.fit(X, y)

# Cross validation交叉驗證
from sklearn import model_selection
# TODO
acc = model_selection.cross_val_score(cl, X, y, cv = 3)
print(acc)

print("Accuracy of the classifier=", round(acc.mean() * 100, 2), "%")

# Testing encoding on single data instance測試單個資料實例上的編碼
input_data = pd.DataFrame(['high', 'low', '2', 'more', 'med', 'high']).T
# TODO
# 因 encode 每次輸入為特定欄位下的 "所有類別，因此需反轉成相應欄位數供轉換。
for col in range(input_data.shape[1]):
    input_data[col] = encoders[col].transform(input_data[col])
    
test = cl.predict(input_data)
print("Output class=", test)

# Predict and print output for a particular datapoint
# TODO
print("Output class=", encoders[-1].inverse_transform(test))

########################
# Validation curves 驗證曲線
from sklearn.model_selection import validation_curve

# TODO
parameter_grid = np.linspace(25, 200, 8).astype(int)

train_scores, validation_scores = validation_curve(cl, X, y, 
        param_name = "n_estimators", param_range = parameter_grid, cv = 5)
print("##### VALIDATION CURVES #####")
print("\nParam: n_estimators\nTraining scores:\n", round(train_scores[0][0], 4))
print("\nParam: n_estimators\nValidation scores:\n", round(validation_scores[-1][0], 4))
