# #############################################################################
# 本題參數設定，請勿更改
seed = 0    # 亂數種子數
# #############################################################################
# TODO

import numpy as np
from sklearn.datasets import load_digits

# 載入手寫數字資料集
data = load_digits()
X_digits = data.data
y_digits = data.target
#print(data)

# 特徵標準化(scale/StandardScaler)
from sklearn.preprocessing import StandardScaler
# TODO
s = StandardScaler()
s_X = s.fit_transform(X_digits)
# =============================================================================
# from sklearn.preprocessing import scale
# s_X = scale(X_digits)
# =============================================================================


# 取出資料集的數字類別數
n_digits = 10


# 建立兩個 K-Means 模型，除以下參數設定外，其餘為預設值
# #############################################################################
# kmean1: init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed
# kmean2: init='random', n_clusters=n_digits, n_init=10, random_state=seed
# #############################################################################
from sklearn.cluster import KMeans
# TODO
kmean1 = KMeans(init = 'k-means++', n_clusters = n_digits, n_init = 10, 
                random_state = seed)
kmean1.fit(s_X)

kmean2 = KMeans(init = 'random', n_clusters = n_digits, n_init = 10, 
                random_state = seed)
kmean2.fit(s_X)

# 利用 PCA 結果建立 K-Means 模型，除以下參數設定外，其餘為預設值
# #############################################################################
# pca: n_components=n_digits, random_state=seed
# kmean3: init=pca.components_, n_clusters=n_digits, n_init=1, random_state=seed
# #############################################################################
from sklearn.decomposition import PCA
# TODO
pca = PCA(n_components = n_digits, random_state = seed)
pca.fit(s_X)
kmean3 = KMeans(init = pca.components_, n_clusters = n_digits, n_init = 1, 
                random_state = seed)
kmean3.fit(s_X)


# 分別計算上述三個 K-Means 模型的輪廓係數(Silhouette coefficient)與
# 分類準確率(accuracy)，除以下參數設定外，其餘為預設值
# #############################################################################
# silhouette_score: metric='euclidean'
# #############################################################################
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
lst_name = ['K-Mean (k-means++)', 'K-Means (random)', 'K-Means (PCA-based)']
# TODO
sils = [silhouette_score(s_X, kmean1.labels_), 
        silhouette_score(s_X, kmean2.labels_),
        silhouette_score(s_X, kmean3.labels_)]

accs = [accuracy_score(y_digits, kmean1.labels_),
        accuracy_score(y_digits, kmean2.labels_),
        accuracy_score(y_digits, kmean3.labels_)]

for n, s, a in zip(lst_name, sils, accs):
    print(n, 'sil =', round(s, 4), 'acc =', round(a, 4))

# 進行 PCA 降維後再做 K-Means，除以下參數設定外，其餘為預設值
# #############################################################################
# kmeans: init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed
# PCA: n_components=2, random_state=seed
# #############################################################################
# TODO

pca = PCA(n_components = 2, random_state = seed)
pca.fit(s_X)
kmean4 = KMeans(init = 'k-means++', n_clusters = n_digits, n_init = 10, 
                random_state = seed)
kmean4.fit(s_X)

print('kmean4 PCA+KMeans Silhouette =', round(silhouette_score(s_X, kmean4.labels_), 4))
print('kmean4 Accuracy =', round(accuracy_score(y_digits, kmean4.labels_), 4))

accs.append(accuracy_score(y_digits, kmean4.labels_))
print('highst Accuracy =', round(max(accs), 4))