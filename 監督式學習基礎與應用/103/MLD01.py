import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt

input_file = 'cardata.txt'

# Reading the data
X = []
y = []
# TODO


# Convert string data to numerical data將字串資料轉換為數值資料
# TODO



# Build a Random Forest classifier建立隨機森林分類器
# TODO



# Cross validation交叉驗證
from sklearn import model_selection
# TODO



print("Accuracy of the classifier=" +        + "%")

# Testing encoding on single data instance測試單個資料實例上的編碼
input_data = ['high', 'low', '2', 'more', 'med', 'high']
# TODO



# Predict and print output for a particular datapoint
# TODO
print("Output class="                     )

########################
# Validation curves 驗證曲線

# TODO


train_scores, validation_scores = validation_curve(classifier, X, y, 
        "n_estimators", parameter_grid, cv=5)
print("##### VALIDATION CURVES #####")
print("\nParam: n_estimators\nTraining scores:\n", train_scores)
print("\nParam: n_estimators\nValidation scores:\n", validation_scores)



