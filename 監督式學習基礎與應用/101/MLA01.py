import numpy as np

input_file = 'wine.csv'

X = []
y = []

# 讀取資料並分割成特徵 (X) 和目標變數 (y)
data = np.genfromtxt(input_file, delimiter = ',')
X = data[:, 1:] # 取第1到第13欄作為特徵
y = data[:, 0].astype(int) # 取第 0 欄作為目標變數，將目標變數轉換為整數

# =============================================================================
# # 讀取資料作法二：
# import pandas as pd
# 
# data = pd.read_csv('wine.csv', header = None)
# df = data.copy() # 避免直接在原始資料上進行修改
# 
# X = df.iloc[:, 1:]
# y = df.iloc[:, 0]
# =============================================================================

# =============================================================================
# # 讀取資料作法三：
# import csv
# 
# # 使用 csv 模組讀取資料
# with open(input_file, 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         y.append(int(row[0]))　# 目標變數，並轉換為整數
#         X.append([float(value) for value in row[1:14]]) # 特徵變數 (化學成分)
# 
# X = np.array(X)
# y = np.array(y)
# =============================================================================


from sklearn import model_selection

# 分割資料集，取75%作為訓練集，25%作為測試集
X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X, y, test_size = 0.25, random_state = 5)

from sklearn.tree import DecisionTreeClassifier

# 初始化並訓練決策樹分類器
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# 計算分類器的精確度
print(f"Accuracy of the classifier = {classifier.score(X_test, y_test) * 100:.2f} %")

# =============================================================================
# # 計算分類器的精確度作法二：
# y_pred = classifier.predict(X_test) # 得出測試集預測結果
# 
# accuracy = np.round(100 * np.mean(y_pred == y_test), 2)
# print("Accuracy of the classifier =", accuracy, " %")
# =============================================================================

# =============================================================================
# # 計算分類器的精確度作法三：
# y_pred = classifier.predict(X_test) # 得出測試集預測結果
# 
# from sklearn.metrics import accuracy_score
# 
# print(f"Accuracy of the classifier = {accuracy_score(y_test, y_pred) * 100:.2f} %")
# =============================================================================


# 測試新的數據點
X_test1 = [[1.51, 1.73, 1.98, 20.15, 85, 2.2, 1.92, .32, 1.48, 2.94, 1, 3.57, 172]]
X_test2 = [[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, .28, 2.29, 5.64, 1.04, 3.92, 1065]]
X_test3 = [[13.71, 5.65, 2.45, 20.5, 95, 1.68, .61, .52, 1.06, 7.7, .64, 1.74, 720]]

print("Prediction for X_test1:", classifier.predict(X_test1))
print("Prediction for X_test2:", classifier.predict(X_test2))
print("Prediction for X_test3:", classifier.predict(X_test3))
