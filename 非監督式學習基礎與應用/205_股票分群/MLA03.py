import json
import pandas as pd
import numpy as np
# TODO
symbol_file = 'symbol_map.json'
table = pd.read_json(symbol_file, orient = 'index')
print(table)
print(table.index)
# TODO
delta = pd.DataFrame()

for symbol in table.index:
    # The daily fluctuations of the quotes 報價的每日波動
    # TODO
    data = pd.read_csv(f"{symbol}.csv")
    delta[symbol] = data['close'] - data['open']
    
print(delta)

# Build a graph model from the correlations 根據相關性建立圖模型
# TODO
from sklearn.covariance import GraphicalLasso

cov = GraphicalLasso()
# Standardize the data 標準化資料
# TODO
from sklearn.preprocessing import StandardScaler

s = StandardScaler()
s_d = s.fit_transform(delta)

# Train the model 訓練模型
# TODO
cov.fit(s_d)
# =============================================================================
# ConvergenceWarning: graphical_lasso: did not converge after 100 iteration: 
# dual gap: 2.908e-04
#   warnings.warn('graphical_lasso: did not converge after '
#                 
# 由於 GraphicalLasso 模型在進行迭代優化時，未能在預設的 100 次迭代內達到收斂條件 
# (dual gap 小於設定的閾值)。
# =============================================================================

# Build clustering model using affinity propagation 用相似性傳播構建分群模型
# TODO
from sklearn.cluster import affinity_propagation

cl = affinity_propagation(cov.covariance_)
print(cl)
print(len(cl[0]))
print(len(cl[1]))

table['cluster'] = cl[1]
print(table)


print()
# Print the results of clustering 列印分群結果
# TODO
#    print("Cluster", i+1, "-->"                   )

#print(type(table.groupby('cluster')))
#print(help(table.groupby('cluster')))

for cluster, group in table.groupby('cluster'):
    print(f"Cluster {cluster}:\n{group}\n")
###############################################################################
print()
print(table.loc[table[0] == 'Cisco', 'cluster'])
print(table.loc[table[0] == 'Cisco', 'cluster'].iloc[0])

# Print the results of clustering 列印分群結果
# TODO
for i in ['Cisco', 'AIG', 'Boeing']:
    print(table[table['cluster'] == table.loc[table[0] == i, 'cluster'].iloc[0]])

