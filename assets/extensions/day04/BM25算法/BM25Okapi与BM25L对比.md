# BM25算法BM25Okapi（标准版）与BM25L（升级版）

## ✔ 对比Okapi

| 项目    | BM25Okapi | BM25L  |
| ------- | --------- | ------ |
| 长文档  | 惩罚明显  | 更平滑 |
| 短文档  | 偏高      | 更稳定 |
| RAG适配 | 一般      | 更强   |

## ✔ 直觉理解

BM25L解决的问题：

- ❌ Okapi：长文档容易被压分
- ✔ BM25L：长文档更公平

## ✔ 代码详解

数据

```python
corpus = [
    "machine learning is great",
    "deep learning for NLP tasks",
    "BM25 is used in information retrieval"
]
```

代码实现

```python
from rank_bm25 import BM25Okapi, BM25L

corpus = [
    "machine learning is great",
    "deep learning for NLP tasks",
    "BM25 is used in information retrieval"
]

tokenized_corpus = [doc.split() for doc in corpus]

query = "learning NLP"
tokenized_query = query.split()

bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(tokenized_query)

bm25_l = BM25L(tokenized_corpus)
scores_l = bm25_l.get_scores(tokenized_query)

print(scores, scores_l)
```





