import pandas as pd

# 載入寶可夢資料集
# TODO
data = pd.read_csv('pokemon.csv')
df = data.copy()
print(df)
print(df.columns)
# 處理遺漏值
features = ['Attack', 'Defense']
# TODO
df = df.dropna(subset = features)
print(df)
# 取出目標寶可夢的 Type1 與兩個特徵欄位
# TODO
df = df[['Attack', 'Defense', 'Type1']]
print(df)
# 編碼 Type1
from sklearn.preprocessing import LabelEncoder
# TODO
l = LabelEncoder()
y = l.fit_transform(df['Type1'])  # 若採用這寫法，會導致答案受影響

# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
s = StandardScaler()
X = s.fit_transform(df[features])  # 若採用這寫法，會導致答案受影響

# 建立線性支援向量分類器，除以下參數設定外，其餘為預設值
# #############################################################################
# C=0.1, dual=False, class_weight='balanced'
# #############################################################################
from sklearn.svm import LinearSVC
# TODO
m = LinearSVC(C=0.1, dual=False, class_weight='balanced')
# 計算分類錯誤的數量
# TODO
df = df[df['Type1'].isin(['Normal', 'Fighting', 'Ghost'])]
print(df)

# =============================================================================
# 這裡才執行，才會符合正解
# X = s.fit_transform(df[features])
# y = l.fit_transform(df['Type1'])
# =============================================================================
m.fit(X, y)

print((y != m.predict(X)).sum())

# 計算準確度(accuracy)
from sklearn.metrics import accuracy_score
print('Accuracy: ', accuracy_score(y, m.predict(X)).round(4))

# 計算有加權的 F1-score (weighted)
from sklearn.metrics import f1_score
# TODO
print('F1-score: ', f1_score(y, m.predict(X), 
                             average = 'weighted').round(4))

# 預測未知寶可夢的 Type1
# TODO
print(l.inverse_transform(m.predict(s.transform([[100,70]]))))

