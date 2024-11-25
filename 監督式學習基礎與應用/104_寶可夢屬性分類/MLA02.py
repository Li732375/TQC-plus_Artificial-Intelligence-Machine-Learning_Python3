import pandas as pd

# 載入寶可夢資料集
# TODO
data = pd.read_csv('pokemon.csv')
print(data)

df = data.copy()

# 處理遺漏值
features = ['Attack', 'Defense']
# TODO
print(df[features].isna().sum())
df = df.dropna(subset = features)

print(df[features][:5])

# 取出目標寶可夢的 Type1 與兩個特徵欄位
# TODO
df = df[df['Type1'].isin(['Normal', 'Fighting', 'Ghost'])]

X = df[features].values
y = df['Type1'].values

# 編碼 Type1
from sklearn.preprocessing import LabelEncoder
# TODO
en = LabelEncoder()
en_y = en.fit_transform(y)

# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
s = StandardScaler()
s_X = s.fit_transform(X)

# 建立線性支援向量分類器，除以下參數設定外，其餘為預設值
# #############################################################################
# C=0.1, dual=False, class_weight='balanced'
# #############################################################################
from sklearn.svm import LinearSVC
# TODO
cl = LinearSVC(C = 0.1, dual = False, class_weight = 'balanced')
cl.fit(s_X, en_y)

# 計算分類錯誤的數量
# TODO
p_y = cl.predict(s_X)
print('mis', (p_y != en_y).sum())

# 計算準確度(accuracy)
from sklearn.metrics import accuracy_score
print('Accuracy: ', round(accuracy_score(en_y, p_y), 4))

# 計算有加權的 F1-score (weighted)
from sklearn.metrics import f1_score
# TODO
print('F1-score: ', round(f1_score(en_y, p_y, average = 'weighted'), 4))

# 預測未知寶可夢的 Type1
# TODO
test = pd.DataFrame([100, 75]).T
s_test = s.transform(test)
print(cl.predict(s_test))
print(en.inverse_transform(cl.predict(s_test)))
