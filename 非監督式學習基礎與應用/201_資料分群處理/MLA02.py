#from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

#  載入scikit-learn資料集範例資料
# TODO
#print(help(make_blobs))
X, y = make_blobs(n_samples = 200, centers = 4, cluster_std = 0.5, 
                  random_state = 0)
#print(X)
#print(y)
print(X.shape)
print(y.shape)

#inertia_集群內誤差平方和，做轉折判斷法的依據
# TODO
count = 0
for i in range(1, 15):
    #實作
    # TODO
    cl = KMeans(n_clusters = i, n_init = 15, max_iter = 200, random_state = 0)
    cl.fit(X)
    
    print(i, cl.inertia_)
    
    if cl.inertia_ < 90:
        break
    
    count += 1

print('vaild clusters =', count)
    
kmeans = KMeans(n_clusters = 4, n_init = 15, max_iter = 200, random_state = 0)
kmeans.fit(X)
print("cluster_centers=", kmeans.cluster_centers_)
print("min X =", round(min(i[0] for i in kmeans.cluster_centers_), 4))
print("max X =", round(max(i[1] for i in kmeans.cluster_centers_), 4))

