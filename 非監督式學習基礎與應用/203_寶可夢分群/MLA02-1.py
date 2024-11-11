import pandas as pd
# 載入寶可夢資料
data = pd.read_json('pokemon.json')
df = data.copy()

# 取出目標欄位
X = df.iloc[:, :5]
# 特徵標準化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)

# 利用 Hierarchical Clustering 進行分群，除以下參數設定外，其餘為預設值
# #############################################################################
# n_clusters=4, affinity='euclidean', linkage='ward'
# #############################################################################
from sklearn.cluster import AgglomerativeClustering
clt = AgglomerativeClustering(n_clusters = 4, metric = 'euclidean', 
                              linkage = 'ward') # 更新寫法
# =============================================================================
# cluster_model = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', 
#                                         linkage = 'ward')
# 在新版本 AgglomerativeClustering 中，affinity 這個參數已經被移除或更名。新的 
# sklearn 版本中，affinity 已經被替換成 metric
# =============================================================================
clt.fit(X)

# 計算每一群的個數
labels = pd.Series(clt.labels_, name = 'labels')
print(f"最小群個數:{min(labels.value_counts())}")
print(f"最大群個數:{max(labels.value_counts())}")

df = pd.concat([df, labels], axis=1)
bylabels_speed_mean = df.groupby('labels')['Speed'].mean()
print(f"{round(bylabels_speed_mean)}")

# 找到 Speed 有遺漏值的兩隻寶可夢，並填入組內平均
print(df[df['Speed'].isna()==True])