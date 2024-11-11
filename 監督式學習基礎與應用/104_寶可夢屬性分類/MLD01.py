import pandas as pd

# 載入寶可夢資料集
# TODO

# 處理遺漏值
features = ['Attack', 'Defense']
# TODO

# 取出目標寶可夢的 Type1 與兩個特徵欄位
# TODO

# 編碼 Type1
from sklearn.preprocessing import LabelEncoder
# TODO

# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO


# 建立線性支援向量分類器，除以下參數設定外，其餘為預設值
# #############################################################################
# C=0.1, dual=False, class_weight='balanced'
# #############################################################################
from sklearn.svm import LinearSVC
# TODO

# 計算分類錯誤的數量
# TODO

# 計算準確度(accuracy)
from sklearn.metrics import accuracy_score
print('Accuracy: '           )

# 計算有加權的 F1-score (weighted)
from sklearn.metrics import f1_score
# TODO
print('F1-score: '          )

# 預測未知寶可夢的 Type1
# TODO


