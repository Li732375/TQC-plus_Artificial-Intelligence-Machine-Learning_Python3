# #############################################################################
# 本題參數設定，請勿更改
seed = 0    # 亂數種子數
# #############################################################################
# TODO

import numpy as np
from sklearn.datasets import load_digits

data = load_digits()
#print(data)
# 載入手寫數字資料集
X_digits = data.data
y_digits = data.target

# 特徵標準化(scale/StandardScaler)
#from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
# TODO
s = StandardScaler()
X = s.fit_transform(X_digits)

# 取出資料集的數字類別數
n_digits = 10

# 建立兩個 K-Means 模型，除以下參數設定外，其餘為預設值
# #############################################################################
# kmean1: init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed
# kmean2: init='random', n_clusters=n_digits, n_init=10, random_state=seed
# #############################################################################
from sklearn.cluster import KMeans
# TODO
k1 = KMeans(init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed)
k2 = KMeans(init='random', n_clusters=n_digits, n_init=10, random_state=seed)

# 利用 PCA 結果建立 K-Means 模型，除以下參數設定外，其餘為預設值
# #############################################################################
# pca: n_components=n_digits, random_state=seed
# kmean3: init=pca.components_, n_clusters=n_digits, n_init=1, random_state=seed
# #############################################################################
from sklearn.decomposition import PCA
# TODO
pca = PCA(n_components=n_digits, random_state=seed)
pca.fit(X)

k3 = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1, random_state=seed)

# 分別計算上述三個 K-Means 模型的輪廓係數(Silhouette coefficient)與
# 分類準確率(accuracy)，除以下參數設定外，其餘為預設值
# #############################################################################
# silhouette_score: metric='euclidean'
# #############################################################################
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
#lst_name = ['K-Mean (k-means++)', 'K-Means (random)', 'K-Means (PCA-based)']
# TODO
silhouette_s = []
accs = []

for cl in [k1, k2, k3]:
    print(X.shape)
    
    cl.fit(X)
    
    sil = silhouette_score(X, cl.labels_)
    silhouette_s.append(sil)
    
    acc = accuracy_score(y_digits, cl.predict(X))
    accs.append(acc)
    
    print('sil =', round(sil, 4), ' acc =', round(acc, 4))


# 進行 PCA 降維後再做 K-Means，除以下參數設定外，其餘為預設值
# #############################################################################
# kmeans: init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed
# PCA: n_components=2, random_state=seed
# #############################################################################
# TODO
pca2 = PCA(n_components=2, random_state=seed)
pca2_X = pca2.fit_transform(X)

k4 = KMeans(init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed)
k4.fit(pca2_X)

sil = silhouette_score(pca2_X, k4.labels_)
silhouette_s.append(sil)

acc = accuracy_score(y_digits, k4.predict(pca2_X))
accs.append(acc)

print('PCA+KMeans Silhouette= ', round(sil, 4))
print('Accuracy= ', round(acc, 4))

print('MAX Accuracy= ', round(max(accs), 4))