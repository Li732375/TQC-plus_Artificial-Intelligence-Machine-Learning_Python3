import numpy as np

input_file = 'wine.csv'

import pandas as pd

data = pd.read_csv(input_file)
print(data.columns) # 確認是否有欄位標題

data = pd.read_csv(input_file, header = None)
df = data.copy()

X = df[df.columns[1:]]
y = df.iloc[:, 0]


# TODO


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 5)

from sklearn.tree import DecisionTreeClassifier


# TODO
cl = DecisionTreeClassifier()
cl.fit(X_train, y_train)

# compute accuracy of the classifier計算分類器的精確度
accuracy = cl.score(X_test, y_test)
print("Accuracy of the classifier =", round(accuracy * 100, 2) , "%")

X_test1 =[[1.51, 1.73, 1.98, 20.15, 85, 2.2, 1.92, .32, 1.48, 2.94, 1, 3.57, 172]]
X_test2 = [[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, .28, 2.29, 5.64, 1.04, 3.92, 1065]]
X_test3 = [[13.71, 5.65, 2.45, 20.5, 95, 1.68, .61, .52, 1.06, 7.7, .64, 1.74, 720]]


# TODO
print(f"{cl.predict(X_test1)}")
print(f"{cl.predict(X_test2)}")
print(f"{cl.predict(X_test3)}")
