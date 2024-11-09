import numpy as np
# TODO

input_file = ('data_perf.txt')
# Load data 載入資料
# TODO




# Find the best epsilon 
eps_grid = np.linspace(            )
silhouette_scores = []
# TODO

    # Train DBSCAN clustering model 訓練DBSCAN分群模型
    # ################
    # min_samples = 5
    # ################
    

    # Extract labels 提取標籤


    # Extract performance metric 提取性能指標
 
    

    print("Epsilon:", eps, " --> silhouette score:", silhouette_score)

    # TODO
    
    

# Best params
print("Best epsilon ="           )

# Associated model and labels for best epsilon
model =   # TODO
labels =  # TODO

# Check for unassigned datapoints in the labels
# TODO


# Number of clusters in the data 
# TODO
print("Estimated number of clusters ="           )

# Extracts the core samples from the trained model
# TODO



