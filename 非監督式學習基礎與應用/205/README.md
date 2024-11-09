## 特別注意
 - 待編修檔裡提到

```
# Build clustering model using affinity propagation 用相似性傳播構建分群模型
```

 > 這裡的 "affinity propagation" 要採用函數，而非類別的那個（AffinityPropagation），盡管預設配置相同，執行結果卻不同（不是同一個演算法？怪...），比較於 MLA02-2.py。

 - 參考答案 MLA02-1.py 採用 GraphicalLassoCV，經測試摸索，用 GraphicalLasso 亦可，皆集中在 MLA02.py 供對照。
 > 也彙整其他計算共變異數矩陣方法做對照，可以看出執行結果有異。
