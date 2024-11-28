import json
import pandas as pd
import numpy as np
# TODO
symbol_file = 'symbol_map.json'

# TODO
symbols = pd.read_json(symbol_file, orient = 'index')
#print(symbols)
#print(type(symbols))

# TODO
stocks = []
for symbol in symbols.index:
    stocks.append(pd.read_csv(f'{symbol}.csv'))

print(stocks)
# The daily fluctuations of the quotes 報價的每日波動
# TODO
st_open = pd.DataFrame([s['open'] for s in stocks], index = symbols.index)
st_close = pd.DataFrame([s['close'] for s in stocks], index = symbols.index)
daily_delta = st_close - st_open
print(daily_delta)

# Build a graph model from the correlations 根據相關性建立圖模型
# TODO
from sklearn.covariance import GraphicalLasso
g = GraphicalLasso()

# Standardize the data 標準化資料
# TODO
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(daily_delta.T)

# Train the model 訓練模型
# TODO
g.fit(X)

# Build clustering model using affinity propagation 用相似性傳播構建分群模型
# TODO
from sklearn.cluster import affinity_propagation
cl = affinity_propagation(g.covariance_)

# Print the results of clustering 列印分群結果
# TODO
print(cl)
print(cl[-1])
print('Cluster labels =', len(set(cl[-1])))

# 列印每個群集中的公司名稱
for i in range(len(set(cl[-1]))):
    print(f"Cluster {i+1} --> {', '.join(symbols[cl[-1] == i][0])}")
