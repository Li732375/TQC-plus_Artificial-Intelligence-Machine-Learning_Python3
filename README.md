# **TQC+ 人工智慧：機器學習Python 3** 認證

## 認證方式

本認證為操作題，總分為100分。

操作題為第一至三類各考一題共三大題十二小題，第一大題至第二大題每題30分，第三大題40分，總計100分。
於認證時間60分鐘內作答完畢，成績加總達70分（含）以上者該科合格。

> 私註：簡述就是在時限內，依據題意撰寫程式，**回答題目的簡答題或選擇題內容（合計共四題）**，並繳交程式碼。

> 若該題檔名 'MLA02' 是適用於官方公告版本執行的答案，由最新版的答案 'MLA01' 調整而來。

> 若該題檔名有 '-1' 等等其他檔案，"通常" 是[參考答案](https://github.com/babymlin/TQC_AI_Licence/tree/main)或我補充內容。部分會註記更新寫法。

> 各題內部注意事項等，為依據如下測試環境配置執行時撰寫註記（尚未調整環境）。

## 各題得分彙總（依據如下測試環境配置執行）

> Wiki 提供環境配置調整流程（若跟我一樣不想安裝 Anaco... XP）

|題號|可得分|推估主要原因|
|:-:|:-:|:-:|
|非監督式學習基礎與應用|
|101|Ｏ||
|102|Ｘ|過去與現今執行結果有異|
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

### 後記
+ 113/11/11 304 要執行出正確答案，scikit-learn 必須為 1.0，以上版本開始會導致答案有誤，建議內容已更新。
+ 113/11/11 為了 204 要執行出正確答案，依據官方公告 Anaconda 2020.02，參考[官網內容](https://docs.anaconda.com/anaconda/release-notes/#anaconda-2020-02-mar-11-2020)，安裝 python 3.8，更新安裝總結彙總如下
```
py -3.8 -m pip install spyder-kernels==2.5.* scikit-learn==0.22.1 pandas==1.0.1 numpy==1.18.1 scipy==1.4.1
```

+  113/11/25 更正該科考試私註，新增適用舊版參考答案。
+  114/01/14 新增題目抽選小幫手
