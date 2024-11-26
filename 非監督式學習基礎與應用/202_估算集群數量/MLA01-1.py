import numpy as np
# TODO
import pandas as pd

input_file = ('data_perf.txt')
# Load data 載入資料
# TODO
data = pd.read_csv(input_file, header = None)
print(data.head())

df = data.copy()

# Find the best epsilon 
eps_grid = np.linspace(0.3, 1.2, num = 10)
silhouette_scores = []

# TODO
from sklearn.cluster import DBSCAN # 大小寫有差異
from sklearn.metrics import silhouette_score

for eps in eps_grid:
    # Train DBSCAN clustering model 訓練DBSCAN分群模型
    # ################
    # min_samples = 5
    # ################
    cl = DBSCAN(eps, min_samples = 5)
    
    # Extract labels 提取標籤
    labels = cl.fit_predict(df)

    # Extract performance metric 提取性能指標
    silhouette_score_v = silhouette_score(df, labels)

    print("Epsilon:", eps, " --> silhouette score:", silhouette_score_v)

    # TODO
    silhouette_scores.append([eps, silhouette_score_v, cl.labels_])
    
# Best params
max_sil = max(s[1] for s in silhouette_scores)
best_eps = [s[0] for s in silhouette_scores if s[1] == max_sil][0]
# =============================================================================
# [expression for item in iterable if condition]
# expression 是對每個元素的操作，通常是對 item 進行某些處理。
# iterable 是要遍歷的對象（例如列表、集合、字典等）。
# if condition 是過濾條件，表示只有當 condition 為 True 時才會執行 expression。
#     
# for s in silhouette_scores 會遍歷 silhouette_scores 中的每個元素（每個元素是像 
# [eps, silhouette_score] 這樣的列表）。
# if s[1] == best_silhouette 是過濾條件，這裡是說，只有當 s[1]（即每個元素中的第
# 二個項，對應於 silhouette_score）等於 best_silhouette 時，才會把這個元素包含在
# 迭代中。
# s[0] 就是對應於符合條件的元素中的第一項，也就是 eps 值。
# 
# 先執行 for，通過 if 的，才會進 expression，否則該元素跳過不執行 expression。
# =============================================================================
print("Best epsilon =", int(best_eps * 10000) / 10000)
print("MAX sil = ", round(max_sil, 4))

# Associated model and labels for best epsilon

# Check for unassigned datapoints in the labels
# TODO
labels = [s[-1] for s in silhouette_scores if s[1] == max_sil][0]
print(labels)

# Number of clusters in the data 
# TODO
print("Estimated number of clusters =", max(labels) + 1)

# Extracts the core samples from the trained model
# TODO
data = pd.read_csv('data_perf_add.txt', header = None)

cl = DBSCAN(best_eps, min_samples = 5)
e_silhouette_score = silhouette_score(data, cl.fit_predict(data))

print('new MAX sil = ', round(e_silhouette_score, 4))