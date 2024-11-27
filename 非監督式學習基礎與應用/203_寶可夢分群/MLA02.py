import pandas as pd
# 載入寶可夢資料
# TODO
data = pd.read_json('pokemon.json')
df = data.copy()

# 取出目標欄位
# TODO
X = df[df.columns[:-1]]
print(X)

# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
sta = StandardScaler()
s_X = sta.fit_transform(X)

# 利用 Hierarchical Clustering 進行分群，除以下參數設定外，其餘為預設值
# #############################################################################
# n_clusters=4, affinity='euclidean', linkage='ward'
# #############################################################################
from sklearn.cluster import AgglomerativeClustering
# TODO
cl = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', 
                             linkage = 'ward')
cl.fit(s_X)

# 計算每一群的個數
# TODO
print(cl.labels_)
cluster_count = pd.Series(cl.labels_).value_counts()
print(cluster_count)

min_c = min(c for c in cluster_count)
max_c = max(c for c in cluster_count)
print('Min =', min_c, 'Max =', max_c)

# 找到 Speed 有遺漏值的兩隻寶可夢，並填入組內平均
# TODO
print(df[df['Speed'].isna()])


df['cluster'] = cl.labels_
clusters_mean = df.groupby('cluster')['Speed'].mean()
print(clusters_mean)
print(type(clusters_mean))

nan_df_index = df[df['Speed'].isna()].index
print(df.loc[nan_df_index])

print('看起來是\n', df['cluster'].map(round(clusters_mean)))
print('實際上是\n', df['cluster'].map(round(clusters_mean))[nan_df_index])
df.iloc[nan_df_index, 5] = df['cluster'].map(round(clusters_mean))
print(df.iloc[nan_df_index])
# =============================================================================
# # or
# df.loc[nan_df_index, 'Speed'] = df['cluster'].map(round(clusters_mean))
# print(df.loc[nan_df_index])
# =============================================================================

print(df[:5])

