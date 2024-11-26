import numpy as np
# TODO
import pandas as pd

input_file = ('data_perf.txt')
# Load data 載入資料
# TODO
# 載入資料，注意使用逗號作為分隔符
data = pd.read_csv(input_file, header = None)

# Find the best epsilon 
eps_grid = np.linspace(0.3, 1.2, num = 10) # 設定 eps 的範圍
silhouette_scores = []

# TODO
from sklearn.cluster import DBSCAN # 大小寫有差異
from sklearn.metrics import silhouette_score # 待編修檔會有一個同名的變數，務必修改變數名稱

for eps in eps_grid:
    # Train DBSCAN clustering model 訓練DBSCAN分群模型
    # ################
    # min_samples = 5
    # ################
    dbscan = DBSCAN(eps = eps, min_samples = 5)
    
    # Extract labels 提取標籤
    labels = dbscan.fit_predict(data) # 提取標籤
    
   
    # Extract performance metric 提取性能指標 
    if len(set(labels)) > 1:  # 確保至少有兩個群聚
        silhouette_score_value = silhouette_score(data, labels) # 計算 silhouette score
    else:
        silhouette_score_value = -1  # 若無法計算，設定為 -1
    
    print(f"Epsilon:{eps} --> silhouette score: {silhouette_score_value:.4f}")

    # TODO
    silhouette_scores.append(silhouette_score_value)  # 儲存 score
print()

# Best params
best_index = np.argmax(silhouette_scores) # 找出最大 silhouette score 的索引
best_eps = eps_grid[best_index]  # 對應的最佳 epsilon
best_silhouette_score = silhouette_scores[best_index] # 最大 silhouette score
print(f"Best epsilon = {best_eps:4f}, Max silhouette_score = {best_silhouette_score:.4f}")
# =============================================================================
# # Best params2
# max_sil = max(s[1] for s in silhouette_scores)
# best_eps = next(s[0] for s in silhouette_scores if s[1] == max_sil)
# print("Best epsilon =", int(best_eps * 10000) / 10000)
# print("MAX sil = ", round(max_sil, 4))
# # =============================================================================
# # [expression for item in iterable if condition]
# # expression 是對每個元素的操作，通常是對 item 進行某些處理。
# # iterable 是要遍歷的對象（例如列表、集合、字典等）。
# # if condition 是過濾條件，表示只有當 condition 為 True 時才會執行 expression。
# #     
# # for s in silhouette_scores 會遍歷 silhouette_scores 中的每個元素（每個元素是像 
# # [eps, silhouette_score] 這樣的列表）。
# # if s[1] == best_silhouette 是過濾條件，這裡是說，只有當 s[1]（即每個元素中的第
# # 二個項，對應於 silhouette_score）等於 best_silhouette 時，才會把這個元素包含在
# # 迭代中。
# # s[0] 就是對應於符合條件的元素中的第一項，也就是 eps 值。
# #
# # 先執行 for，通過 if 的，才會進 expression，否則該元素跳過不執行 expression。
# # =============================================================================
# =============================================================================
 
# =============================================================================
# # Best params3
# max_sil = max(s[1] for s in silhouette_scores)
# best_eps = [s[0] for s in silhouette_scores if s[1] == max_sil][0]
# 
# print("Best epsilon =", int(best_eps * 10000) / 10000)
# print("MAX sil = ", round(max_sil, 4))
# =============================================================================

# Associated model and labels for best epsilon

# Check for unassigned datapoints in the labels
# TODO
num_clusters = len(set(labels)) - (1 if -1 in labels else 0) # 減去未分配的資料點數

# Number of clusters in the data 
# TODO
print(f"Estimated number of clusters = {num_clusters}")

# Extracts the core samples from the trained model
# TODO
model = DBSCAN(eps = best_eps, min_samples = 5) # 使用最佳 epsilon 重新訓練模型
# 載入新資料
data = pd.read_csv('data_perf_add.txt', sep = ',', header = None, 
                   names = ['x', 'y'])  # 使用 ',' 作為分隔符
labels = model.fit_predict(data) # 提取標籤
print(f"New data of Max silhouette_score = {silhouette_score(data, labels):.4f}")


# =============================================================================
# DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一種基於
# 密度的聚類演算法，廣泛用於處理空間數據和其他高維數據。與其他聚類演算法（如 K-means
# ）不同，DBSCAN 能夠識別任意形狀的聚類，並且能有效地處理噪音數據。
# 
# 基本概念
#
# 1. 核心點（Core Point）：如果一個點的鄰域內包含至少 min_samples 數量的點（包括自
# 身），則該點被視為核心點。
# 2. 邊界點（Border Point）：一個點如果不是核心點，但在某個核心點的鄰域內，則該點為
# 邊界點。
# 3. 噪音點（Noise Point）：既不是核心點也不是邊界點的點被視為噪音點。
# 4. 鄰域（Epsilon Neighborhood）：在 DBSCAN 中，鄰域的範圍是由參數 eps 決定的，表
# 示核心點的影響範圍。
# 
# 
# DBSCAN 的主要步驟如下：
# 
# 1. 初始化：指定兩個參數：eps 和 min_samples。eps 是定義鄰域的半徑，min_samples 
# 是一個核心點所需的最小點數量。
# 
# 2. 選擇點：隨機選擇一個未被標記的點。
# 
# 3. 核心點檢查：檢查該點是否為核心點。如果是核心點，則開始形成一個新的聚類。
# 
# 4. 擴展聚類：將核心點的所有鄰域內的點加入到該聚類中，並檢查這些新加入的點是否也為
# 核心點。如果是，則重複擴展過程。
# 
# 5. 重複步驟：重複上述步驟，直到所有點都被標記為聚類點或噪音點。
# =============================================================================
