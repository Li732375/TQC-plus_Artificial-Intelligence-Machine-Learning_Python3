import json
import pandas as pd
from sklearn.cluster import affinity_propagation

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

# =============================================================================

from sklearn.covariance import GraphicalLasso

# 使用 Graphical Lasso 計算共變異數矩陣
graphical_lasso_model = GraphicalLasso()
graphical_lasso_model.fit(X)

# Build clustering model using affinity propagation 用相似性傳播構建分群模型
labels_graphical_lasso_cv = affinity_propagation(graphical_lasso_model.covariance_, 
                                                 random_state = 0)[1]

# 計算分群數量
num_labels_graphical_lasso = len(set(labels_graphical_lasso_cv))

# 列印分群結果
print(f'Number of clusters using Graphical Lasso: {num_labels_graphical_lasso}')
print("\nClustering Results using Graphical Lasso:")
for i in range(num_labels_graphical_lasso):
    print("Cluster", i + 1, "-->", ', '.join(names[labels_graphical_lasso_cv == i]))

# 印出與 Cisco 同群的公司名稱
cisco_index = symbols[symbols == 'CSCO'].index[0]
cisco_cluster = labels_graphical_lasso_cv[cisco_index]
print("\nthe same cluster as Cisco:", ', '.join(names[labels_graphical_lasso_cv == cisco_cluster]))

# 印出與 AIG 同群的公司名稱
aig_index = symbols[symbols == 'AIG'].index[0]
aig_cluster = labels_graphical_lasso_cv[aig_index]
print("the same cluster as AIG:", ', '.join(names[labels_graphical_lasso_cv == aig_cluster]))

# 印出與 Boeing 同群的公司名稱
boeing_index = symbols[symbols == 'BA'].index[0]
boeing_cluster = labels_graphical_lasso_cv[boeing_index]
print("the same cluster as Boeing:", ', '.join(names[labels_graphical_lasso_cv == boeing_cluster]))

print()
print()
# =============================================================================

from sklearn.covariance import GraphicalLassoCV

# 使用 Graphical Lasso CV 計算共變異數矩陣
graphical_lasso_cv_model = GraphicalLassoCV()
graphical_lasso_cv_model.fit(X)

# Build clustering model using affinity propagation 用相似性傳播構建分群模型
labels_graphical_lasso_cv = affinity_propagation(graphical_lasso_cv_model.covariance_, 
                                                 random_state = 0)[1]

# 計算分群數量
num_labels_graphical_lasso_cv = len(set(labels_graphical_lasso_cv))

# 列印分群結果
print(f'Number of clusters using Graphical Lasso CV: {num_labels_graphical_lasso_cv}')
print("\nClustering Results using Graphical Lasso CV:")
for i in range(num_labels_graphical_lasso_cv):
    print("Cluster", i + 1, "-->", ', '.join(names[labels_graphical_lasso_cv == i]))

# 印出與 Cisco 同群的公司名稱
cisco_index = symbols[symbols == 'CSCO'].index[0]
cisco_cluster = labels_graphical_lasso_cv[cisco_index]
print("\nthe same cluster as Cisco:", ', '.join(names[labels_graphical_lasso_cv == cisco_cluster]))

# 印出與 AIG 同群的公司名稱
aig_index = symbols[symbols == 'AIG'].index[0]
aig_cluster = labels_graphical_lasso_cv[aig_index]
print("the same cluster as AIG:", ', '.join(names[labels_graphical_lasso_cv == aig_cluster]))

# 印出與 Boeing 同群的公司名稱
boeing_index = symbols[symbols == 'BA'].index[0]
boeing_cluster = labels_graphical_lasso_cv[boeing_index]
print("the same cluster as Boeing:", ', '.join(names[labels_graphical_lasso_cv == boeing_cluster]))

print()
print()
# =============================================================================
# 以下皆錯
# =============================================================================

from sklearn.covariance import OAS

# 使用 OAS 計算共變異數矩陣
oas_model = OAS()
oas_model.fit(X)

# Build clustering model using affinity propagation 用相似性傳播構建分群模型
labels_oas = affinity_propagation(oas_model.covariance_, 
                                                 random_state = 0)[1]

# 計算分群數量
num_labels_oas = len(set(labels_oas))

# 列印分群結果
print(f'Number of clusters using OAS: {num_labels_oas}')
print("\nClustering Results using OAS:")
for i in range(num_labels_oas):
    print("Cluster", i + 1, "-->", ', '.join(names[labels_oas == i]))

# 印出與 Cisco 同群的公司名稱
cisco_index = symbols[symbols == 'CSCO'].index[0]
cisco_cluster = labels_oas[cisco_index]
print("\nthe same cluster as Cisco:", ', '.join(names[labels_oas == cisco_cluster]))

# 印出與 AIG 同群的公司名稱
aig_index = symbols[symbols == 'AIG'].index[0]
aig_cluster = labels_oas[aig_index]
print("the same cluster as AIG:", ', '.join(names[labels_oas == aig_cluster]))

# 印出與 Boeing 同群的公司名稱
boeing_index = symbols[symbols == 'BA'].index[0]
boeing_cluster = labels_oas[boeing_index]
print("the same cluster as Boeing:", ', '.join(names[labels_oas == boeing_cluster]))

print()
print()
# =============================================================================

from sklearn.covariance import EmpiricalCovariance

# 使用 EmpiricalCovariance 計算共變異數矩陣
empirical_model = EmpiricalCovariance()
empirical_model.fit(X)

# Build clustering model using affinity propagation 用相似性傳播構建分群模型
labels_empirical = affinity_propagation(empirical_model.covariance_, 
                                                 random_state = 0)[1]

# 計算分群數量
num_labels_empirical = len(set(labels_empirical))

# 列印分群結果
print(f'Number of clusters using EmpiricalCovariance: {num_labels_empirical}')
print("\nClustering Results using EmpiricalCovariance:")
for i in range(num_labels_empirical):
    print("Cluster", i + 1, "-->", ', '.join(names[labels_empirical == i]))

# 印出與 Cisco 同群的公司名稱
cisco_index = symbols[symbols == 'CSCO'].index[0]
cisco_cluster = labels_empirical[cisco_index]
print("\nthe same cluster as Cisco:", ', '.join(names[labels_empirical == cisco_cluster]))

# 印出與 AIG 同群的公司名稱
aig_index = symbols[symbols == 'AIG'].index[0]
aig_cluster = labels_empirical[aig_index]
print("the same cluster as AIG:", ', '.join(names[labels_empirical == aig_cluster]))

# 印出與 Boeing 同群的公司名稱
boeing_index = symbols[symbols == 'BA'].index[0]
boeing_cluster = labels_empirical[boeing_index]
print("the same cluster as Boeing:", ', '.join(names[labels_empirical == boeing_cluster]))

print()
print()
# =============================================================================

from sklearn.covariance import LedoitWolf

# 使用 Ledoit-Wolf 收縮計算共變異數矩陣
ledoit_wolf_model = LedoitWolf()
ledoit_wolf_model.fit(X)

# Build clustering model using affinity propagation 用相似性傳播構建分群模型
labels_ledoit_wolf = affinity_propagation(ledoit_wolf_model.covariance_, random_state = 0)[1]

# 計算分群數量
num_labels_ledoit_wolf = len(set(labels_ledoit_wolf))

# 列印分群結果
print(f'Number of clusters using Ledoit-Wolf: {num_labels_ledoit_wolf}')
print("\nClustering Results using Ledoit-Wolf:")
for i in range(num_labels_ledoit_wolf):
    print("Cluster", i + 1, "-->", ', '.join(names[labels_ledoit_wolf == i]))

# 印出與 Cisco 同群的公司名稱
cisco_index = symbols[symbols == 'CSCO'].index[0]
cisco_cluster = labels_ledoit_wolf[cisco_index]
print("\nthe same cluster as Cisco:", ', '.join(names[labels_ledoit_wolf == cisco_cluster]))

# 印出與 AIG 同群的公司名稱
aig_index = symbols[symbols == 'AIG'].index[0]
aig_cluster = labels_ledoit_wolf[aig_index]
print("the same cluster as AIG:", ', '.join(names[labels_ledoit_wolf == aig_cluster]))

# 印出與 Boeing 同群的公司名稱
boeing_index = symbols[symbols == 'BA'].index[0]
boeing_cluster = labels_ledoit_wolf[boeing_index]
print("the same cluster as Boeing:", ', '.join(names[labels_ledoit_wolf == boeing_cluster]))

print()
print()
# =============================================================================

from sklearn.covariance import EllipticEnvelope

# 使用 MCD 計算共變異數矩陣
mcd_model = EllipticEnvelope()  # contamination 是異常值比例
mcd_model.fit(X)

# Build clustering model using affinity propagation 用相似性傳播構建分群模型
labels_mcd = affinity_propagation(mcd_model.covariance_, random_state = 0)[1]

# 計算分群數量
num_labels_mcd = len(set(labels_ledoit_wolf))

# 列印分群結果
print(f'Number of clusters using Minimum Covariance Determinant: {num_labels_mcd}')
print("\nClustering Results using Minimum Covariance Determinant:")
for i in range(num_labels_mcd):
    print("Cluster", i + 1, "-->", ', '.join(names[labels_mcd == i]))

# 印出與 Cisco 同群的公司名稱
cisco_index = symbols[symbols == 'CSCO'].index[0]
cisco_cluster = labels_mcd[cisco_index]
print("\nthe same cluster as Cisco:", ', '.join(names[labels_mcd == cisco_cluster]))

# 印出與 AIG 同群的公司名稱
aig_index = symbols[symbols == 'AIG'].index[0]
aig_cluster = labels_mcd[aig_index]
print("the same cluster as AIG:", ', '.join(names[labels_mcd == aig_cluster]))

# 印出與 Boeing 同群的公司名稱
boeing_index = symbols[symbols == 'BA'].index[0]
boeing_cluster = labels_mcd[boeing_index]
print("the same cluster as Boeing:", ', '.join(names[labels_mcd == boeing_cluster]))

print()
print()
