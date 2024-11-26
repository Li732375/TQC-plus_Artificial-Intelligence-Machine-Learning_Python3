import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt

input_file = 'cardata.txt'

# Reading the data
import pandas as pd

data = pd.read_csv(input_file)
print(data[:5])

data = pd.read_csv(input_file, header = None)
df = data.copy()

# TODO


# Convert string data to numerical data將字串資料轉換為數值資料
# TODO
encoders = []
en_df = pd.DataFrame()

for i in range(len(df.columns)):
    en = preprocessing.LabelEncoder()
    en_df[i] = en.fit_transform(df[i])
    encoders.append(en)

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
accs = model_selection.cross_val_score(cl, X, y, cv = 3)
print(accs)

print("Accuracy of the classifier=", round(accs.mean() * 100, 2), "%")

# Testing encoding on single data instance測試單個資料實例上的編碼
input_data = ['high', 'low', '2', 'more', 'med', 'high']
# TODO
input_data1 = pd.DataFrame(input_data).T

for i in range(input_data1.shape[1]):
    input_data1[i] = encoders[i].transform(input_data1[i])
print(input_data1)

# Predict and print output for a particular datapoint
# TODO
print(cl.predict(input_data1))
print("Output class=", encoders[-1].inverse_transform(cl.predict(input_data1)))

# =============================================================================
# # 作法二
# input_data2 = input_data
# print(input_data2)
# 
# for i in range(len(input_data2)): 
#     input_data2[i] = encoders[i].transform([input_data2[i]])[0]
# print(input_data2)
# 
# # Predict and print output for a particular datapoint
# # TODO
# print(cl.predict([input_data2]))
# print("Output class=", encoders[-1].inverse_transform(cl.predict([input_data2])))
# =============================================================================

########################
# Validation curves 驗證曲線
    
# TODO
parameter_grid = np.linspace(25, 200, 8).astype(int)

train_scores, validation_scores = model_selection.validation_curve(cl, X, y, 
                                                   "n_estimators", 
                                                   parameter_grid, 
                                                   cv = 5)
print("##### VALIDATION CURVES #####")
print("\nParam: n_estimators\nTraining scores:\n", 
      round(train_scores[0][0], 4))
print("\nParam: n_estimators\nValidation scores:\n", 
      round(validation_scores[-1][0], 4))



