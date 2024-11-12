#from matplotlib import pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs # 新版的 Scikit-learn 中已經被移除。
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#  載入scikit-learn資料集範例資料
# TODO
X, y = make_blobs(n_samples = 200, centers = 4, cluster_std = 0.5, 
                  random_state = 0)

#inertia_集群內誤差平方和，做轉折判斷法的依據
# TODO

# 測試不同分群數
k = 1
inertia = 91

while inertia > 90: # 直到 inertia 未大於 90 時停止
    kmeans = KMeans(n_clusters = k, init = 'k-means++', n_init = 15, 
                    max_iter = 200, random_state = 0)
    kmeans.fit(X)
    
    print(f"inertia = {kmeans.inertia_}, 分群數量為: {k}")
    
    inertia = kmeans.inertia_
    k += 1

# =============================================================================
# # 測試不同分群數
# for k in range(1, 10):  # 設定合理的範圍
#     kmeans = KMeans(n_clusters=k, init='k-means++', n_init=15, 
#                     max_iter=200, random_state=0)
#     kmeans.fit(X)
#     
#     print(f"inertia = {kmeans.inertia_}, 分群數量為: {k}")
#     
#     if kmeans.inertia_ <= 90:
#         break
# 
# =============================================================================

# 實作 K-means 演算法
kmeans = KMeans(n_clusters = 4, init = 'k-means++', n_init = 15, 
                max_iter = 200, random_state = 0)
kmeans.fit(X)

# 取得所有中心點座標
centers = kmeans.cluster_centers_
#print(centers)

# 找出最小的中心點 X 軸位置
min_x_center = min(row[0] for row in centers)
print(f"min_x cluster_centers= {min_x_center:.4f}")

# 找出最大的中心點 Y 軸位置
max_y_center = max(row[1] for row in centers)
print(f"max_y cluster_centers= {max_y_center:.4f}")


