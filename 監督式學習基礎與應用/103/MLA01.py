import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt

input_file = 'cardata.txt'

# Reading the data
X = []
y = []
# TODO
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line.strip().split(',')
        X.append(data[:-1])
        y.append(data[-1])

# Convert string data to numerical data將字串資料轉換為數值資料
# TODO
label_encoder = []
X_encoded = np.empty((len(X), len(X[0])))
for i, item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform([x[i] for x in X])

label_encoder_y = preprocessing.LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Build a Random Forest classifier建立隨機森林分類器
# TODO
classifier = RandomForestClassifier(n_estimators = 200, max_depth = 8, 
                                    random_state = 7)
classifier.fit(X_encoded, y_encoded)

# Cross validation交叉驗證
from sklearn import model_selection
# TODO
accuracy = model_selection.cross_val_score(classifier, X_encoded, y_encoded, 
                                           cv = 3)
print(f"Accuracy of the classifier= {accuracy.mean() * 100:.2f} %")

# Testing encoding on single data instance測試單個資料實例上的編碼
input_data = ['high', 'low', '2', 'more', 'med', 'high']
# TODO
input_data_encoded = [-1] * len(input_data)
for i, item in enumerate(input_data):
    input_data_encoded[i] = label_encoder[i].transform([item])[0]


# Predict and print output for a particular datapoint
# TODO
predicted_class = classifier.predict([input_data_encoded])[0]
print(f"Output class= {label_encoder_y.inverse_transform([predicted_class])[0]}")


########################
# Validation curves 驗證曲線

# TODO
from sklearn.model_selection import validation_curve

parameter_grid = np.linspace(25, 200, 8).astype(int)
train_scores, validation_scores = validation_curve(classifier, X_encoded, 
                                                   y_encoded, 
                                                   param_name = "n_estimators", 
                                                   param_range = parameter_grid, 
                                                   cv = 5)
print("##### VALIDATION CURVES #####")
print(f"\nParam: n_estimators\nTraining scores: {train_scores[0][0]:.5f}") # 小數點後第四位無條件捨去
print(f"\nParam: n_estimators\nValidation scores: {validation_scores[-1][0]:.5f}") # 小數點後第四位無條件捨去



