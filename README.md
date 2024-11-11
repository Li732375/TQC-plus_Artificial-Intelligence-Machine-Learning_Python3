# **TQC+ 人工智慧：機器學習Python 3** 認證

## 認證方式

本認證為操作題，總分為100分。

操作題為第一至三類各考一題共三大題十二小題，第一大題至第二大題每題30分，第三大題40分，總計100分。
於認證時間60分鐘內作答完畢，成績加總達70分（含）以上者該科合格。

> 私註：簡述就是在時限內，依據題意撰寫程式，**回答題目的簡答題或選擇題內容（合計共四題）**，並繳交程式碼，無限繳交次數，可以多次修改程式碼。
> 若該題檔名有 '-1' 等等其他檔案，"通常" 是[參考答案](https://github.com/babymlin/TQC_AI_Licence/tree/main)或補充內容。部分會註記更新寫法。

## 各題得分彙總

|題號|可得分|推估主要原因|
|:-:|:-:|:-:|
|非監督式學習基礎與應用|
|101|Ｏ||
|102|Ｘ|過去與現今的執行結果有異|
|103|Ｏ||
|104|Ｏ||
|105|Ｏ||
|監督式學習基礎與應用|
|201|Ｏ||
|202|Ｏ||
|203|Ｏ||
|204|Ｘ|過去與現今執行結果有異|
|205|Ｏ||
|機器學習應用|
|301|？|此題停考|
|302|Ｘ|套件的資料來源斷供|
|303|Ｏ||
|304|Ｘ|過去與現今執行結果有異|
|305|Ｏ||

測試環境
Spyder: 5.5.5

Python: 3.12.4

Scikit-learn:
```
CMD > python -m pip show scikit-learn
Name: scikit-learn
Version: 1.5.1
Summary: A set of python modules for machine learning and data mining
Home-page: https://scikit-learn.org
Author:
Author-email:
License: new BSD
Location: C:\Users\你的 OS 帳號名稱\AppData\Local\Programs\Python\Python312\Lib\site-packages
Requires: joblib, numpy, scipy, threadpoolctl
Required-by:

```

## 環境調整流程

### 檢視已安裝 Python 版本
```
CMD > where python
C:\Users\你的 OS 帳號名稱\AppData\Local\Programs\Python\Python312\python.exe
C:\Users\你的 OS 帳號名稱\AppData\Local\Microsoft\WindowsApps\python.exe

```

若沒有指定版本（如 3.9），前往官網 > Downloads > Windows，Ctrl + F 找目標號碼 3.9，並且提供 **Download Windows installer (64-bit)** 下載安裝。
> 安裝時，記得勾選路徑添加！

再次檢視，確認已安裝
```
CMD > where python
C:\Users\你的 OS 帳號名稱\AppData\Local\Programs\Python\Python39\python.exe
C:\Users\你的 OS 帳號名稱\AppData\Local\Programs\Python\Python312\python.exe
C:\Users\你的 OS 帳號名稱\AppData\Local\Microsoft\WindowsApps\python.exe

```

### 安裝目標套件1（若是採用 spyder 作 IDE，需安裝 spyder-kernels 2.5.*，否則跳過，至 **[安裝目標套件2](https://github.com/Li732375/TQC-plus_Artificial-Intelligence-Machine-Learning_Python3/blob/master/README.md#%E5%AE%89%E8%A3%9D%E7%9B%AE%E6%A8%99%E5%A5%97%E4%BB%B62%E4%BB%A5-scikit-learn-11-%E7%82%BA%E4%BE%8B)**） 

以 python 3.9 為例
```
py -3.9 -m pip install spyder-kernels==2.5.*
```

檢查套件成功安裝與否
```
CMD > py -3.9 -m pip show spyder-kernels
Name: spyder-kernels
Version: 2.5.2
Summary: Jupyter kernels for Spyder's console
Home-page: https://github.com/spyder-ide/spyder-kernels
Author: Spyder Development Team
Author-email: spyderlib@googlegroups.com
License: MIT
Location: c:\users\你的 OS 帳號名稱\appdata\local\programs\python\python39\lib\site-packages
Requires: cloudpickle, ipykernel, ipython, jupyter-client, pyzmq
Required-by:

```

### 安裝目標套件2（以 scikit-learn 1.1、python 3.9 為例） 
```
CMD > py -3.9 -m pip install scikit-learn==1.1
Collecting scikit-learn==1.1
  Downloading scikit_learn-1.1.0-cp39-cp39-win_amd64.whl (8.3 MB)
     ---------------------------------------- 8.3/8.3 MB 3.8 MB/s eta 0:00:00
Collecting scipy>=1.3.2
  Downloading scipy-1.13.1-cp39-cp39-win_amd64.whl (46.2 MB)
     ---------------------------------------- 46.2/46.2 MB 3.4 MB/s eta 0:00:00
Collecting joblib>=1.0.0
  Downloading joblib-1.4.2-py3-none-any.whl (301 kB)
     ---------------------------------------- 301.8/301.8 KB 4.7 MB/s eta 0:00:00
Collecting threadpoolctl>=2.0.0
  Downloading threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
Collecting numpy>=1.17.3
  Downloading numpy-2.0.2-cp39-cp39-win_amd64.whl (15.9 MB)
     ---------------------------------------- 15.9/15.9 MB 3.2 MB/s eta 0:00:00
Installing collected packages: threadpoolctl, numpy, joblib, scipy, scikit-learn
Successfully installed joblib-1.4.2 numpy-2.0.2 scikit-learn-1.2.0 scipy-1.13.1 threadpoolctl-3.5.0
WARNING: You are using pip version 22.0.4; however, version 24.3.1 is available.
You should consider upgrading via the 'C:\Users\你的 OS 帳號名稱\AppData\Local\Programs\Python\Python39\python.exe -m pip install --upgrade pip' command.

```

> 安裝過程中，還安裝了相關的依賴套件，如 scipy, joblib, numpy 和 threadpoolctl。此外，提示中有關 pip 版本較舊的警告，會被建議將 pip 升級到最新版本。（就是因為執行結果有差呀...）


檢查套件成功安裝與否
```
CMD > py -3.9 -m pip show scikit-learn
Name: scikit-learn
Version: 1.1.0
Summary: A set of python modules for machine learning and data mining
Home-page: http://scikit-learn.org
Author:
Author-email:
License: new BSD
Location: c:\users\你的 OS 帳號名稱\appdata\local\programs\python\python39\lib\site-packages
Requires: joblib, numpy, scipy, threadpoolctl
Required-by:

```

### 安裝目標套件3（其他項目，以 python 3.9 為例）
```
py -3.9 -m pip install pandas==1.1.5 numpy==1.19.5 scipy==1.7.3
```

> 這是現今測試時，有效版本組合的其中一種，可以再自行嘗試喔～(累

其中一種因版本不相容造成的錯誤
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

### 安裝總結（隨後記內容持續更新）
最終更新為 python 3.8，彙整前述內容如下
```
py -3.8 -m pip install spyder-kernels==2.5.* scikit-learn==0.22.1 pandas==1.0.1 numpy==1.18.1 scipy==1.4.1
```

> 若採用 spyder 作 IDE，需開啟 spyder 設定， **[設定方式](https://youtu.be/miJOoagmWAw)**） 


### 後記
113/11/11 304 要執行出正確答案，scikit-learn 必須為 1.0，以上版本開始會導致答案有誤，已對上面建議內容更新。
113/11/11 為了 204 要執行出正確答案，依據官方公告 Anaconda 2020.02，參考[官網內容](https://docs.anaconda.com/anaconda/release-notes/#anaconda-2020-02-mar-11-2020)，安裝 python 3.8，更新安裝總結彙總如下
```
py -3.8 -m pip install spyder-kernels==2.5.* scikit-learn==0.22.1 pandas==1.0.1 numpy==1.18.1 scipy==1.4.1
```
