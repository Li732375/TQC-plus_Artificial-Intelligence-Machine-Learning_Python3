import numpy as np
# TODO

input_file = ('data_perf.txt')
# Load data 載入資料
# TODO
import pandas as pd

data = pd.read_csv(input_file, header = None)
df = data.copy()

# Find the best epsilon 
eps_grid = np.linspace(0.3, 1.2, num = 10)
silhouette_scores = []
# TODO
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

for eps in eps_grid:
    # Train DBSCAN clustering model 訓練DBSCAN分群模型
    # ################
    # min_samples = 5
    # ################
    cl = DBSCAN(eps, min_samples = 5)

    # Extract labels 提取標籤
    cl.fit(df)
    label = cl.labels_

    # Extract performance metric 提取性能指標
    silhouette_s = silhouette_score(df, label)
    
    print("Epsilon:", eps, " --> silhouette score:", silhouette_s)

    # TODO
    silhouette_scores.append([eps, silhouette_s, label])
    

# Best params
higher_sil = max([e[1] for e in silhouette_scores])
best_eps = [e[0] for e in silhouette_scores if e[1] == higher_sil][0]
print("Best epsilon =", int(best_eps * 10000) / 10000)
print('higher sil =', round(higher_sil, 4))


# Associated model and labels for best epsilon


# Check for unassigned datapoints in the labels
# TODO


# Number of clusters in the data 
# TODO
best_cluster = [e[-1] for e in silhouette_scores if e[1] == higher_sil][0]
print("Estimated number of clusters =", max(best_cluster) + 1)

# Extracts the core samples from the trained model
# TODO
model = DBSCAN(best_eps, min_samples = 5)

new_data = pd.read_csv('data_perf_add.txt', header = None)
label = model.fit_predict(new_data)

new_silhouette_s = silhouette_score(new_data, label)

print('new higher sil =', round(new_silhouette_s, 4))
