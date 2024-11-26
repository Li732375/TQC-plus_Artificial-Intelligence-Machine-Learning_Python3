import pandas as pd

# 載入寶可夢資料集
# TODO
data = pd.read_csv('pokemon.csv')
print(data)
print(data.columns)

df = data.copy()
# 處理遺漏值
features = ['Attack', 'Defense']
# TODO
print(df[features].isna().sum())
df = df.dropna(subset = features)
#df = df[features].dropna() # 若寫成這樣，則只會有 features 指定的欄位會保留
print(df[features].isna().sum())

# 取出目標寶可夢的 Type1 與兩個特徵欄位
# TODO
df = df[['Type1', 'Attack', 'Defense']]
print(df)

# 編碼 Type1
from sklearn.preprocessing import LabelEncoder
# TODO
en = LabelEncoder()
df['en_T'] = en.fit_transform(df['Type1'])
print(df)

# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
s = StandardScaler()
df[features] = s.fit_transform(df[features])
print(df)

# 建立線性支援向量分類器，除以下參數設定外，其餘為預設值
# #############################################################################
# C=0.1, dual=False, class_weight='balanced'
# #############################################################################
from sklearn.svm import LinearSVC
# TODO
cl = LinearSVC(C=0.1, dual=False, class_weight='balanced')
# 計算分類錯誤的數量
# TODO
df = df[df['Type1'].isin(['Normal', 'Fighting', 'Ghost'])]
print(df)

cl.fit(df[features], df['en_T'])

print(cl.predict(df[features]) != df['en_T'])
# =============================================================================
# num = len(cl.predict(df[features]) != df['en_T'])
# print(num) # 只會取得比對表 numpy.ndarray 的長度
# num = len([cl.predict(df[features]) != df['en_T']])
# print(num) # 只會取得比對表的長度
# =============================================================================
num = (cl.predict(df[features]) != df['en_T']).sum()
print(num)

# 計算準確度(accuracy)
from sklearn.metrics import accuracy_score
print('Accuracy: ', round(accuracy_score(df['en_T'], cl.predict(df[features])), 4))

# 計算有加權的 F1-score (weighted)
from sklearn.metrics import f1_score
# TODO
print('F1-score: ', 
      round(f1_score(df['en_T'], cl.predict(df[features]), average = 'weighted'), 4))

# 預測未知寶可夢的 Type1
# TODO
print('test', en.inverse_transform(cl.predict([[100, 75]])))
