# #############################################################################
# 本題參數設定，請勿更改
seed = 0    # 亂數種子數
# #############################################################################

import pandas as pd

# 載入寶可夢資料
# TODO
data = pd.read_csv('pokemon.csv')
#print(data)
df = data.copy()

# 取出目標欄位
#TODO 特徵欄位
X = df[['Defense', 'SpecialAtk']]
#TODO Type1 欄位
y = df['Type1']
       
#print(X)
#print(y)

# 編碼 Type1
from sklearn import preprocessing
# TODO
en = preprocessing.LabelEncoder()
y = en.fit_transform(y)
#print(y)

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
s = StandardScaler().fit(X_train)
X_train = s.transform(X_train)
X_test = s.transform(X_test)
#print(X_train)
#print(X_test)

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
clf1 = RandomForestClassifier(n_estimators = 10, random_state = seed)
clf2 = KNeighborsClassifier(n_neighbors = 4)
clf3 = SVC(gamma = .1, kernel = 'rbf', probability = True)
v = VotingClassifier(estimators = [('RF', clf1), ('kNN', clf2), ('SVC', clf3)], 
         voting = 'hard', n_jobs = -1)

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
    kf = KFold(n_splits = 5, shuffle = True, random_state = seed) # 產生 5 折交叉驗證的分割器
    
    # 執行 k 折交叉驗證
    # #########################################################################
    # pipeline, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1
    # #########################################################################
    cv_result = cross_val_score(pipeline, X_train, y_train, cv = kf, 
                                scoring = 'accuracy', n_jobs = -1) # 執行交叉驗證，計算模型在每折上的準確度。
    
    return  cv_result.mean() # 計算 5 折的準確度平均值，作為模型的整體表現。

# 利用 kfold_cross_validation()，分別讓分類器執行 k 折交叉驗證，計算準確度(accuracy)
#TODO

# #############################################################################
    
# 利用訓練集的標準化結果，針對測試集進行標準化
# TODO

# 上述分類器針對測試集進行預測，並計算分類錯誤的個數與準確度
from sklearn.metrics import accuracy_score
# TODO
k_accs = []
t_accs = []
errs = []

for i in [('RF', clf1), ('kNN', clf2), ('SVC', clf3), ('Voting', v)]:
    i[1].fit(X_train, y_train)
    acc = kfold_cross_validation(s, i[1])
    k_accs.append(round(acc, 4))
    
    p_y = i[1].predict(X_test)
    t_accs.append(round(accuracy_score(y_test, p_y), 4))
    errs.append((y_test != p_y).sum())

print('k highest acc =', max(k_accs))
print('test highest acc =', max(t_accs))
print('min err =', min(errs))

# #############################################################################
    
# 分別利用上述分類器預測分類
print("===== 預測分類 ======")
# TODO
p = v.predict(s.transform([[100, 70]]))
print('p =', en.inverse_transform(p))