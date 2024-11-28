import json
import pandas as pd

# 讀取股票代號
with open('symbol_map.json') as fp:
    json_data = json.load(fp)
symbols = pd.DataFrame(sorted(json_data.items()))[0]
names = pd.DataFrame(sorted(json_data.items()))[1]

# 讀取股票資料
stocks = []
for symbol in symbols:
    stocks.append(pd.read_csv(f"{symbol}.csv"))

# 計算每日波動
open_price = pd.DataFrame([stock['open'] for stock in stocks], index = symbols)
close_price = pd.DataFrame([stock['close'] for stock in stocks], index = symbols)
daily_var_price = close_price - open_price  # 注意這裡是關閉價減去開盤價

# 標準化資料
X = pd.DataFrame(daily_var_price).T
X /= X.std()


from sklearn.covariance import GraphicalLasso

# 使用 Graphical Lasso 計算共變異數矩陣
graphical_lasso_model = GraphicalLasso()
graphical_lasso_model.fit(X)
cov_matrix_graphical_lasso = graphical_lasso_model.covariance_

# =============================================================================

from sklearn.cluster import AffinityPropagation

# 建立 Affinity Propagation 分群模型(預設值參數)，使用 Graphical Lasso 的共變異數矩陣
Affinity_model_graphical_lasso = AffinityPropagation()
Affinity_model_graphical_lasso.fit(cov_matrix_graphical_lasso)

# 獲取分群結果
labels_graphical_lasso = Affinity_model_graphical_lasso.labels_

# 計算分群數量
num_labels = len(set(labels_graphical_lasso))

# 列印分群結果
print(f'num_labels: {num_labels}')

# Print the results of clustering 列印分群結果
for i in range(num_labels):
    print("Cluster", i + 1, "-->", ', '.join(names[labels_graphical_lasso == i]))

print()
print()
# =============================================================================

from sklearn.cluster import affinity_propagation

# 使用 affinity_propagation 分群函數(預設值參數)
labels = affinity_propagation(cov_matrix_graphical_lasso)[1]

# 計算分群數量
num_labels = len(set(labels))
print(f'num_labels: {num_labels}')

# Print the results of clustering 列印分群結果
for i in range(len(set(labels))):
    print("Cluster", i+1, "-->" ,', '.join(names[labels == i]))