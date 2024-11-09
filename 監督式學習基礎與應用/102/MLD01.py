import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model

# 原始資料
titanic = pd.read_csv("titanic.csv")
print('raw data')
# TODO

# 將年齡的空值填入年齡的中位數
# TODO

print("年齡中位數="        )
# TODO

# 更新後資料
print('new data')
# TODO

# 轉換欄位值成為數值
label_encoder = preprocessing.LabelEncoder()
encoded_class = label_encoder.fit_transform(        )
# TODO


# 建立模型
# TODO

print('截距='          )
print('迴歸係數='       )


# 混淆矩陣(Confusion Matrix)，計算準確度
print('Confusion Matrix')
# TODO




