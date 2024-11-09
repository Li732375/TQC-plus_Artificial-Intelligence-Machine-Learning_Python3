import numpy as np
# Reading the data
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('cardata.txt', header=None)
df = data.copy()

# Convert string data to numerical data將字串資料轉換為數值資料
from sklearn.preprocessing import LabelEncoder
le_en = []
for i in range(df.shape[1]):
    le =  LabelEncoder().fit(df[i])
    le_en.append(le)
    df[i] = le_en[-1].transform(df[i])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Build a Random Forest classifier建立隨機森林分類器
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=7)
model.fit(X, y)

# Cross validation交叉驗證
from sklearn.model_selection import cross_validate
cv_dic = cross_validate(model, X, y, cv = 3)
score = (cv_dic.get('test_score')).mean()
print(f"Accuracy of the classifier= {score*100:.2f} %")

# Testing encoding on single data instance測試單個資料實例上的編碼
input_data = pd.DataFrame(['high', 'low', '2', 'more', 'med', 'high']).T
for i in range(input_data.shape[1]):
    input_data[i] = le_en[i].transform(input_data[i])
# Predict and print output for a particular datapoint
input_pred = model.predict(input_data)
label = le_en[-1].inverse_transform(input_pred)
print(f"Output class= {label[0]}")

# Validation curves 驗證曲線
parameter_grid = np.linspace(25, 200, 8).astype(int)
from sklearn.model_selection import validation_curve
train_scores, validation_scores = validation_curve(model, X, y, 
                                                   param_name = 'n_estimators', 
                                                   param_range = parameter_grid, 
                                                   cv = 5) # 更新寫法
# =============================================================================
# 原寫法：
# train_scores, validation_scores = validation_curve(model, X, y, 
#         'n_estimators', parameter_grid, cv=5)
# =============================================================================
print("##### VALIDATION CURVES #####")
print("\nParam: n_estimators\nTraining scores:\n", train_scores)
print("\nParam: n_estimators\nValidation scores:\n", validation_scores)