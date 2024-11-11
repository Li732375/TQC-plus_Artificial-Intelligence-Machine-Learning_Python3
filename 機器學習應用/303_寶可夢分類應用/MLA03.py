# #############################################################################
# 本題參數設定，請勿更改
seed = 0    # 亂數種子數
# #############################################################################

import pandas as pd

# 載入寶可夢資料
# TODO
pokemon_data = pd.read_csv('pokemon.csv')

# 取出目標欄位
X = pokemon_data[['Defense', 'SpecialAtk']] #特徵欄位
y = pokemon_data['Type1'] #Type1 欄位

# 編碼 Type1
from sklearn import preprocessing
# TODO
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# 切分訓練集、測試集，除以下參數設定外，其餘為預設值
# #########################################################################
# X, y, test_size=0.2, random_state=seed
# #########################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = seed)

# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
scalar = StandardScaler().fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

# 訓練集
# 分別建立 RandomForest, kNN, SVC, Voting，除以下參數設定外，其餘為預設值
# #############################################################################
# RandomForest: n_estimators=10, random_state=seed
# kNN: n_neighbors=4
# SVC: gamma=.1, kernel='rbf', probability=True
# Voting: estimators=[('RF', clf1), ('kNN', clf2), ('SVC', clf3)], 
#         voting='hard', n_jobs=-1
# #############################################################################    
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# TODO
# 分別建立 RandomForest, kNN, SVC, Voting
clf1 = RandomForestClassifier(n_estimators = 10, random_state = seed)
clf2 = KNeighborsClassifier(n_neighbors = 4)
clf3 = SVC(gamma = 0.1, kernel = 'rbf', probability = True)

# Voting classifier (hard voting)
voting_clf = VotingClassifier(estimators = [('RF', clf1), ('kNN', clf2), 
                                            ('SVC', clf3)], voting = 'hard', 
                              n_jobs = -1)

# 建立函式 kfold_cross_validation() 執行 k 折交叉驗證，並回傳準確度的平均值
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
def kfold_cross_validation(scalar, model):
    """ 函式描述：執行 k 折交叉驗證
    參數：
        scalar (StandardScaler):標準化適配的結果
        model: 機器學習模型

    回傳：
        k 折交叉驗證的準確度(accuracy)平均值
    """
    # 建立管線，用來進行(標準化 -> 機器學習模型)
    pipeline = make_pipeline(scalar, model)
    
    # 產生 k 折交叉驗證，除以下參數設定外，其餘為預設值
    # #########################################################################
    # n_splits=5, shuffle=True, random_state=seed
    # #########################################################################
    kf = KFold(n_splits = 5, shuffle = True, random_state = seed)
    
    # 執行 k 折交叉驗證
    # #########################################################################
    # pipeline, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1
    # #########################################################################
    cv_result = cross_val_score(pipeline, X_train, y_train, cv = kf, 
                                    scoring = 'accuracy', n_jobs = -1)
    
    return cv_result.mean()

# 利用 kfold_cross_validation()，分別讓分類器執行 k 折交叉驗證，計算準確度(accuracy)
#TODO
models = {'RandomForest': clf1, 'kNN': clf2, 'SVC': clf3, 'Voting': voting_clf}

accuracies_train = {}
errors_test = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = kfold_cross_validation(scalar, model)
    accuracies_train[name] = round(accuracy, 4)
    
# 印出四個分類器對訓練集的準確度平均值 (四捨五入取至小數點後第四位)
print("1. 最大分類準確度平均值 (訓練集):", max(accuracies_train.values()))

# #############################################################################
    
# 利用訓練集的標準化結果，針對測試集進行標準化
# TODO

# 上述分類器針對測試集進行預測，並計算分類錯誤的個數與準確度
from sklearn.metrics import accuracy_score

accuracies_test = {}
errors_test = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_test[name] = round(accuracy, 4)
    
    errors = (y_pred != y_test).sum()  # 計算錯誤的樣本數
    errors_test[name] = errors


# 印出四個分類器對測試集的最大分類準確度
print("2. 最大分類準確度 (測試集):", max(accuracies_test.values()))

# 計算最小分類錯誤樣本數
print("3. 最小分類錯誤樣本數:", min(errors_test.values()))

# #############################################################################
    
# 分別利用上述分類器預測分類
print("===== 預測分類 ======")
# TODO
# 預測未知寶可夢的 Type1，Defense=100, SpecialAtk=70
unknown_pokemon = [[100, 70]]
unknown_pokemon_scaled = scalar.transform(unknown_pokemon)
predicted_type = voting_clf.predict(unknown_pokemon_scaled)

# 預測結果並輸出
print("4. 投票分類器預測的Type1分類選項:", le.inverse_transform(predicted_type)[0])

