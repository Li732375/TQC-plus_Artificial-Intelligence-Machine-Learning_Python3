import pandas as pd

# 載入寶可夢資料集
# TODO
df = pd.read_csv('pokemon.csv')

# 處理遺漏值
features = ['Attack', 'Defense']
# TODO
df = df.dropna(subset = features)

# 取出目標寶可夢的 Type1 與兩個特徵欄位
# TODO
df = df[df['Type1'].isin(['Normal', 'Fighting', 'Ghost'])] # 從欄 'Type1' 的多樣類型裡，過濾出部分指定類型
# =============================================================================
# 檢查某列或 Series 中的值是否存在於給定的列表或集合中。返回一個布林值序列，指出每個
# 元素是否在指定的列表或集合中。
# 
# Series.isin(values)
# 
# values: 一個串列、集合或其他類似的資料結構，包含要檢查的值。
# 
# 複合條件檢查
# filtered_df = df[(df['Type1'].isin(['Normal', 'Fighting', 'Ghost'])) & (df['Attack'] > 50)]
# 得出兼具 Type1 和 Attack 條件的資料。
# =============================================================================

X = df[features].values
y = df['Type1'].values

# 編碼 Type1
from sklearn.preprocessing import LabelEncoder
# TODO
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 建立線性支援向量分類器，除以下參數設定外，其餘為預設值
# #############################################################################
# C=0.1, dual=False, class_weight='balanced'
# #############################################################################
from sklearn.svm import LinearSVC
# TODO
classifier = LinearSVC(C = 0.1, dual = False, class_weight = 'balanced', 
                       random_state = 42)
classifier.fit(X_scaled, y_encoded)

# 計算分類錯誤的數量
# TODO
y_pred = classifier.predict(X_scaled)

#沒有內建的函數來直接計算錯誤數量。
num_errors = (y_pred != y_encoded).sum() 
# 計算布林數組中 True 的數量，也就是計算 y_pred 和 y_encoded 不相等的元素個數。
print(f'Number of classification errors: {num_errors}')

# 計算準確度(accuracy)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_encoded, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# 計算有加權的 F1-score (weighted)
from sklearn.metrics import f1_score
# TODO
f1 = f1_score(y_encoded, y_pred, average = 'weighted')
print(f'F1-score: {f1:.4f}')

# 預測未知寶可夢的 Type1
# TODO
unknown_pokemon = [[100, 75]]
#unknown_pokemon = pd.DataFrame([100, 75]).T

unknown_pokemon_scaled = scaler.transform(unknown_pokemon) # 標準化
predicted_type = classifier.predict(unknown_pokemon_scaled) # 模型預測
predicted_label = label_encoder.inverse_transform(predicted_type) # 轉換回其原始的類別名稱。
print(f'The predicted Type1 for the unknown Pokemon: {predicted_label[0]}')

