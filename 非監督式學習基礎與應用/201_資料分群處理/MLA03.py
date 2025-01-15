from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

#  載入scikit-learn資料集範例資料
# TODO
X, y = make_blobs(n_samples = 200, centers = 4, cluster_std = 0.5, 
                  random_state = 0)

#inertia_集群內誤差平方和，做轉折判斷法的依據
# TODO
import pandas as pd
t = pd.DataFrame(columns = ['n', 'inertia', 'cluster_centers'])

for i in range(1,10):
    #實作
    # TODO
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init=15, max_iter=200,
                    random_state = 0)
    kmeans.fit(X, y)
    #print(i , ' ', kmeans.inertia_)
    t.loc[len(t)] = [i, 
                     kmeans.inertia_.round(4), 
                     kmeans.cluster_centers_]
    
    if kmeans.inertia_ < 90:
        break

print(t[['n', 'inertia']])
print(min([x[0] for x in t.loc[3, 'cluster_centers']]).round(4))
print(max([x[1] for x in t.loc[3, 'cluster_centers']]).round(4))

