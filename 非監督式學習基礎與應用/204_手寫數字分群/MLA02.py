# #############################################################################
# 本題參數設定，請勿更改
seed = 0    # 亂數種子數
# #############################################################################
# TODO

#import numpy as np
from sklearn.datasets import load_digits

# 載入手寫數字資料集
data = load_digits()
X_digits = data.data # 數值特徵
y_digits = data.target # 數字類別

# 特徵標準化(scale/StandardScaler)
from sklearn.preprocessing import scale
data = scale(X_digits)

# =============================================================================
# from sklearn.preprocessing import StandardScaler
# s = StandardScaler()
# data = s.fit_transform(data)
# =============================================================================

# 取出資料集的數字類別數
n_digits = 10

# 建立兩個 K-Means 模型，除以下參數設定外，其餘為預設值
# #############################################################################
# kmean1: init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed
# kmean2: init='random', n_clusters=n_digits, n_init=10, random_state=seed
# #############################################################################
from sklearn.cluster import KMeans
kmean1 = KMeans(init = 'k-means++', n_clusters = n_digits, n_init = 10, 
                random_state = seed)
kmean2 = KMeans(init = 'random', n_clusters = n_digits, n_init = 10, 
                random_state = seed)

# 利用 PCA 結果建立 K-Means 模型，除以下參數設定外，其餘為預設值
# #############################################################################
# pca: n_components=n_digits, random_state=seed
# kmean3: init=pca.components_, n_clusters=n_digits, n_init=1, random_state=seed
# #############################################################################
from sklearn.decomposition import PCA
pca = PCA(n_components = n_digits, random_state = seed).fit(data)
kmean3 = KMeans(init = pca.components_, n_clusters = n_digits, n_init = 1, 
                random_state = seed)

# 訓練模型並進行分群
kmean1.fit(data)
kmean2.fit(data)
kmean3.fit(data)

# 計算輪廓係數(Silhouette coefficient)與分類準確率(accuracy)
from sklearn.metrics import silhouette_score, accuracy_score
lst_name = ['K-Mean (k-means++)', 'K-Means (random)', 'K-Means (PCA-based)', 
            'PCA+KMeans']

# 利用分群標籤的結果計算輪廓係數和準確率
silhouette_scores = [
    silhouette_score(data, kmean1.labels_, metric = 'euclidean'),
    silhouette_score(data, kmean2.labels_, metric = 'euclidean'),
    silhouette_score(data, kmean3.labels_, metric = 'euclidean')
]

# 將預測標籤轉換以對比真實類別標籤計算準確度
accuracies = [
    accuracy_score(y_digits, kmean1.labels_),
    accuracy_score(y_digits, kmean2.labels_),
    accuracy_score(y_digits, kmean3.labels_)
]

for name, silhouette, accuracy in zip(lst_name, silhouette_scores, accuracies):
    print(f"{name} Silhouette= {silhouette:.4f}")
    print(f"{name} Accuracy= {accuracy:.4f}")
    print()

# 進行 PCA 降維後再做 K-Means
# #############################################################################
# kmeans: init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed
# PCA: n_components=2, random_state=seed
# #############################################################################
#pca_2 = PCA(n_components = 2, random_state = seed).fit(data)
pca_2 = PCA(n_components = 2, random_state = seed).fit_transform(data)
kmeans_pca = KMeans(init = 'k-means++', n_clusters = n_digits, n_init = 10, 
                    random_state = seed)
kmeans_pca.fit(pca_2)

# 計算 PCA+KMeans 的輪廓係數與準確率
silhouette_pca = silhouette_score(data, kmeans_pca.labels_, 
                                  metric = 'euclidean')
silhouette_scores.append(silhouette_pca)

accuracy_pca = accuracy_score(y_digits, kmeans_pca.labels_)
accuracies.append(accuracy_pca)

print(f'PCA+KMeans Silhouette = {silhouette_pca:.4f}')
print(f'PCA+KMeans Accuracy = {accuracy_pca:.4f}')

print()
print(f'Max Accuracy = {max(accuracies):.4f}')