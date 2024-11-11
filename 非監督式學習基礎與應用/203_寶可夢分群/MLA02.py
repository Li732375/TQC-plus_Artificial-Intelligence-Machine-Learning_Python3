import pandas as pd
# 載入寶可夢資料
# TODO
data = pd.read_json('pokemon.json')

# 取出目標欄位
# TODO
columns = ["HP", "Attack", "Defense", "SpecialAtk", "SpecialDef"]
features = data[columns]

# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 利用 Hierarchical Clustering 進行分群，除以下參數設定外，其餘為預設值
# #############################################################################
# n_clusters=4, affinity='euclidean', linkage='ward'
# #############################################################################
from sklearn.cluster import AgglomerativeClustering
# TODO
cluster_model = AgglomerativeClustering(n_clusters = 4, metric = 'euclidean', 
                                        linkage = 'ward')
# =============================================================================
# cluster_model = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', 
#                                         linkage = 'ward')
# 在新版本 AgglomerativeClustering 中，affinity 這個參數已經被移除或更名。新的 
# sklearn 版本中，affinity 已經被替換成 metric
# =============================================================================
cluster_model.fit(scaled_features)  # 計算分群結果

# 計算每一群的個數
# TODO
cluster_counts = pd.Series(cluster_model.labels_).value_counts()

min_cluster_size = cluster_counts.min()  # 最小群元素個數
print(f"最小群元素個數: {min_cluster_size}")

max_cluster_size = cluster_counts.max()  # 最大群元素個數
print(f"最大群元素個數: {max_cluster_size}")

data['Cluster'] = cluster_model.labels_  # 將分群結果加入資料中
mean_speed_per_cluster = data.groupby('Cluster')['Speed'].mean() # 先分群，後計算每一群 Speed 的平均值
#print(f"{round(mean_speed_per_cluster)}") # 分類群集的 Speed 平均值


# 找到 Speed 有遺漏值的兩隻寶可夢，並填入組內平均
# TODO
print(data[data['Speed'].isna()]) # 有遺漏值的兩隻寶可夢
nan_index = data[data['Speed'].isna()].index # 取得索引值
#print(nan_index)

print('\nFill after')
data.loc[data['Speed'].isna(), 'Speed'] = data['Cluster'].map(round(mean_speed_per_cluster)) # 填補 Speed 遺漏值
print(data.loc[nan_index, 'Speed']) # 有遺漏值的兩隻寶可夢 Speed 