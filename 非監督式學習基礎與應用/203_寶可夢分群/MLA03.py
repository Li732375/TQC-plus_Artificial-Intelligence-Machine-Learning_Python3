import pandas as pd
# 載入寶可夢資料
# TODO
data = pd.read_json('pokemon.json')
# 取出目標欄位
# TODO
print(data)
df = data.copy()

# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
s = StandardScaler()
df[df.columns[:-1]] = s.fit_transform(df[df.columns[:-1]])
print(df)

# 利用 Hierarchical Clustering 進行分群，除以下參數設定外，其餘為預設值
# #############################################################################
# n_clusters=4, affinity='euclidean', linkage='ward'
# #############################################################################
from sklearn.cluster import AgglomerativeClustering
# TODO
cl = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cl.fit(df[df.columns[:-1]])

# 計算每一群的個數
# TODO
print(cl.n_clusters_)
print(cl.labels_)

min_count = min(i for i in pd.Series(cl.labels_).value_counts())
print('min =', min_count)

max_count = max(i for i in pd.Series(cl.labels_).value_counts())
print('max =', max_count)


# 找到 Speed 有遺漏值的兩隻寶可夢，並填入組內平均
# TODO
print(df.isna().sum())
print(df[df['Speed'].isna()])
print(df[df.isna().any(axis = 1)])
print(data[data['Speed'].isna()])

df['cluster'] = cl.labels_
print(df.iloc[[19, 63]])

group_mean = df.groupby('cluster')['Speed'].mean().round()
print(group_mean)

df.loc[[19, 63], 'Speed'] = df.loc[[19, 63], 'cluster'].map(group_mean)
print(df.iloc[[19, 63]])
