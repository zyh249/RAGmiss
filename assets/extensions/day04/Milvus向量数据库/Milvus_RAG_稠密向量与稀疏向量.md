# Milvus 中稠密向量与稀疏向量

## 1. 核心结论

在 Milvus 的 RAG 系统中，稠密向量和稀疏向量本质上都是对文本的向量化表示，但它们表达的信息不同。

可以先记住一句话：

> 稠密向量存语义相似度，稀疏向量存关键词匹配强度。

也可以理解为：

```text
稠密向量 = 负责"意思像不像"
稀疏向量 = 负责"关键词准不准"
Hybrid Search = 同时考虑语义相似和关键词命中
```

---

## 2. 稠密向量存什么？

稠密向量一般来自 Embedding 模型，例如：

```text
text-embedding-v3
bge-large-zh
mxbai-embed-large
bge-m3 dense
```

Embedding 模型会把一段文本转换成一个固定长度的浮点数列表。

例如原文：

```text
Milvus 是一个向量数据库，适合做 RAG 检索。
```

经过 Embedding 模型后，可能变成：

```python
[0.023, -0.118, 0.452, 0.009, ..., -0.031]
```

真实项目中，稠密向量通常是几百维或几千维，例如：

```text
768 维
1024 维
1536 维
```

在 Milvus 中，稠密向量字段一般这样定义：

```python
{
    "name": "dense_vector",
    "type": DataType.FLOAT_VECTOR,
    "dim": 1024
}
```

插入 Milvus 的数据格式类似：

```python
{
    "id": 1,
    "text": "Milvus 是一个向量数据库，适合做 RAG 检索。",
    "dense_vector": [0.023, -0.118, 0.452, 0.009, ..., -0.031]
}
```

### 稠密向量的作用

稠密向量主要用于语义检索。

例如用户问题是：

```text
怎么把文本变成向量？
```

文档中写的是：

```text
Embedding 是将文本映射到高维空间的过程。
```

虽然两句话没有完全相同的关键词，但是语义接近，稠密向量可以把它们匹配起来。

---

## 3. 稀疏向量存什么？

稀疏向量一般来自 BM25、SPLADE、BGE-M3 sparse 等方法。

稀疏向量不是存完整的浮点数列表，而是只存"哪些词重要，以及这些词的权重"。

例如原文：

```text
Milvus 是一个向量数据库，适合做 RAG 检索。
```

可能被表示成：

```python
{
    105: 0.83,
    927: 1.42,
    3051: 0.76,
    8812: 1.18
}
```

含义可以理解为：

```text
词编号 105   权重 0.83
词编号 927   权重 1.42
词编号 3051  权重 0.76
词编号 8812  权重 1.18
```

这里的 `105`、`927`、`3051`、`8812` 可以理解为词表中的词 ID。

稀疏向量的特点是：

```text
大部分词的权重都是 0
只有少量出现过、重要的词有权重
```

所以它叫稀疏向量。

在 Milvus 中，稀疏向量字段一般这样定义：

```python
{
    "name": "sparse_vector",
    "type": DataType.SPARSE_FLOAT_VECTOR
}
```

插入 Milvus 的数据格式类似：

```python
{
    "id": 1,
    "text": "Milvus 是一个向量数据库，适合做 RAG 检索。",
    "sparse_vector": {105: 0.83, 927: 1.42, 3051: 0.76, 8812: 1.18}
}
```

### 稀疏向量的作用

稀疏向量主要用于关键词检索。

例如用户问题是：

```text
BM25 的 k1 参数是什么意思？
```

稀疏向量会重点关注：

```text
BM25
k1
参数
```

如果某个文档明确包含这些关键词，那么它的稀疏向量匹配分数通常会比较高。

---

## 4. 稠密向量与稀疏向量格式对比

| 类型 | 存储内容 | 数据格式 | 主要作用 |
|---|---|---|---|
| 稠密向量 | 文本语义信息 | `[0.12, -0.03, 0.88, ...]` | 找意思相近的文本 |
| 稀疏向量 | 关键词权重信息 | `{词ID: 权重, 词ID: 权重}` | 找关键词命中的文本 |

---

## 5. 用一个例子理解

用户问题：

```text
Milvus 如何创建索引？
```

假设有三段文档：

```text
文档 A：Milvus 支持 IVF_FLAT、HNSW 等索引类型。
文档 B：向量数据库可以提升语义检索效率。
文档 C：如何在 Milvus 中创建 Collection？
```

### 5.1 稠密向量的判断方式

稠密向量会从语义角度判断：

```text
Milvus
索引
向量数据库
Collection
```

这些概念是否在语义空间中接近。

所以它可能认为：

```text
文档 A 和问题最相关
文档 C 也有一定相关性
文档 B 和问题有弱相关性
```

稠密向量的优势是可以理解语义，不完全依赖字面关键词。

### 5.2 稀疏向量的判断方式

稀疏向量更关注关键词是否命中。

用户问题中重点词是：

```text
Milvus
创建
索引
```

如果文档中明确出现这些词，稀疏向量分数就会更高。

例如：

```text
文档 A：包含 Milvus、索引
文档 C：包含 Milvus、创建
文档 B：没有明显关键词命中
```

所以稀疏向量会更偏向文档 A 和文档 C。

---

## 6. RAG 中为什么两个都要用？

单独使用稠密向量或稀疏向量都有问题。

### 6.1 只用稠密向量的问题

稠密向量擅长语义理解，但是有时会"过度联想"。

例如用户问：

```text
BM25 的 k1 参数是什么意思？
```

稠密向量可能找出一些语义相关但不够精确的内容，例如：

```text
向量检索中的相似度计算方法
检索排序算法介绍
```

这些内容和检索有关系，但不一定真正解释了 BM25 的 k1 参数。

### 6.2 只用稀疏向量的问题

稀疏向量擅长关键词命中，但是不太擅长理解同义表达。

例如用户问：

```text
怎么把文本变成向量？
```

文档中写的是：

```text
Embedding 是将文本映射到高维空间的过程。
```

如果文档没有出现"文本变成向量"这些字面词，稀疏向量可能匹配不强。

### 6.3 混合检索的价值

因此，在 RAG 系统中经常使用 Hybrid Search：

```text
Dense Search：负责语义召回
Sparse Search：负责关键词召回
Rerank：负责最终精排
```

这样可以同时解决两个问题：

```text
1. 用户换一种说法，也能通过语义找到相关文档
2. 用户提到具体术语、参数、函数名时，也能精准命中文档
```

---

## 7. Milvus 中常见的混合存储结构

在 RAG 系统中，文档通常会先被切分成 chunk。

每个 chunk 入库时，一般会同时保存以下内容：

```text
id             文档片段 ID
text           原始文本片段
source         来源文件
page           页码或章节
dense_vector   稠密向量
sparse_vector  稀疏向量
```

完整数据结构示例：

```python
{
    "id": 1,
    "text": "Milvus 是一个向量数据库，适合做 RAG 检索。",
    "source": "milvus教程.md",
    "page": 3,
    "dense_vector": [0.023, -0.118, 0.452, 0.009, ..., -0.031],
    "sparse_vector": {105: 0.83, 927: 1.42, 3051: 0.76}
}
```

对应关系如下：

```text
text          给大模型生成答案时使用
source/page   给用户展示引用来源时使用
dense_vector  用来做语义检索
sparse_vector 用来做关键词检索
```

---

## 8. Milvus Schema 简化示例

下面是一个简化版 Milvus Schema 设计：

```python
from pymilvus import DataType

schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=True
)

schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True
)

schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    max_length=4096
)

schema.add_field(
    field_name="source",
    datatype=DataType.VARCHAR,
    max_length=512
)

schema.add_field(
    field_name="page",
    datatype=DataType.INT64
)

schema.add_field(
    field_name="dense_vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=1024
)

schema.add_field(
    field_name="sparse_vector",
    datatype=DataType.SPARSE_FLOAT_VECTOR
)
```

注意：

```text
稠密向量字段需要指定 dim
稀疏向量字段不需要指定 dim
```

因为稠密向量是固定长度数组，而稀疏向量只保存非零位置和对应权重。

---

## 9. 查询时的数据流

用户输入问题：

```text
Milvus 如何创建索引？
```

系统通常会做两件事：

```text
1. 把问题转换成稠密向量
2. 把问题转换成稀疏向量
```

示例：

```python
query_dense_vector = [0.031, -0.086, 0.517, ..., 0.022]

query_sparse_vector = {
    105: 1.21,
    998: 0.95,
    3051: 1.43
}
```

然后在 Milvus 中分别检索：

```text
dense_vector 字段：做语义相似度搜索
sparse_vector 字段：做关键词权重搜索
```

最后通过 WeightedRanker 或 Reranker 进行融合排序。

---

## 10. 最终总结

在 Milvus 的 RAG 系统中：

```text
稠密向量 = 存文本的语义信息
稀疏向量 = 存文本的关键词权重信息
```

对应数据格式：

```python
dense_vector = [0.1, -0.2, 0.3, ...]

sparse_vector = {
    12: 0.8,
    305: 1.5,
    9012: 0.6
}
```

最终在一条 RAG 文档数据中，通常会同时保存：

```text
1. 原始文本 text
2. 元数据 source/page
3. 稠密向量 dense_vector
4. 稀疏向量 sparse_vector
```

可以这样记忆：

```text
稠密向量：像"语义坐标"，看意思是否接近。
稀疏向量：像"关键词得分表"，看关键词是否命中。
混合检索：语义召回 + 关键词召回 + 重排序。
```


---

# Milvus 稀疏向量索引类型

## 1. 核心结论

在 Milvus 中，稀疏向量字段的索引类型重点记住一个：

```text
SPARSE_INVERTED_INDEX
```

它是 Milvus 当前推荐使用的稀疏向量倒排索引。

旧版本中还可以看到：

```text
SPARSE_WAND
```

但是从 Milvus 2.5.4 开始，`SPARSE_WAND` 已经逐步废弃。新版更推荐使用：

```python
index_type="SPARSE_INVERTED_INDEX"
params={"inverted_index_algo": "DAAT_WAND"}
```

也就是说：

```text
旧写法：SPARSE_WAND
新写法：SPARSE_INVERTED_INDEX + DAAT_WAND
```

---

## 2. 稀疏向量索引类型对比

| 索引类型 | 是否推荐 | 说明 |
|---|---|---|
| `SPARSE_INVERTED_INDEX` | 推荐 | 稀疏向量倒排索引，当前主流写法 |
| `SPARSE_WAND` | 不推荐新项目使用 | 旧版本支持，从 Milvus 2.5.4 开始逐步废弃 |

在新项目里，建议直接使用：

```text
SPARSE_INVERTED_INDEX
```

不要再优先使用：

```text
SPARSE_WAND
```

---

## 3. SPARSE_INVERTED_INDEX 是什么？

`SPARSE_INVERTED_INDEX` 可以理解为：

```text
给稀疏向量建立"词编号 -> 文档列表"的倒排索引
```

例如一条文档的稀疏向量是：

```python
{
    105: 0.83,
    927: 1.42,
    3051: 0.76
}
```

可以理解为：

```text
词编号 105   权重 0.83
词编号 927   权重 1.42
词编号 3051  权重 0.76
```

建立倒排索引以后，大概会形成类似结构：

```text
105  -> 文档1、文档8、文档20
927  -> 文档1、文档3、文档17
3051 -> 文档1、文档5、文档9
```

当用户查询时，查询问题也会被转换成稀疏向量：

```python
{
    927: 1.25,
    3051: 0.91
}
```

Milvus 就可以优先查找包含这些维度的文档，而不是对所有文档做全量扫描。

---

## 4. 创建稀疏向量索引的推荐代码

下面是 RAG 系统中给稀疏向量字段创建索引的常见写法：

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://192.168.88.100:19530")

index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="sparse_vector",
    index_name="sparse_inverted_index",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",
    params={
        "inverted_index_algo": "DAAT_MAXSCORE"
    }
)

client.create_index(
    collection_name="doc_crud_collection",
    index_params=index_params
)
```

这段代码的核心是：

```python
index_type="SPARSE_INVERTED_INDEX"
metric_type="IP"
params={"inverted_index_algo": "DAAT_MAXSCORE"}
```

---

## 5. metric_type 为什么用 IP？

稀疏向量常用：

```python
metric_type="IP"
```

`IP` 是 Inner Product，中文叫：

```text
内积
```

稀疏向量检索时，Milvus 会比较查询稀疏向量和文档稀疏向量在相同维度上的权重乘积。

例如查询向量：

```python
{
    927: 1.25,
    3051: 0.91
}
```

文档向量：

```python
{
    927: 1.42,
    3051: 0.76,
    5000: 0.33
}
```

它们共同命中的维度是：

```text
927
3051
```

分数可以简单理解为：

```text
1.25 × 1.42 + 0.91 × 0.76
```

也就是：

```text
共同关键词越多，关键词权重越高，IP 分数越大，越相似。
```

---

## 6. inverted_index_algo 参数

使用 `SPARSE_INVERTED_INDEX` 时，可以通过 `inverted_index_algo` 指定倒排索引的检索算法。

常见参数有 3 个：

| 参数值 | 含义 | 适合场景 |
|---|---|---|
| `DAAT_MAXSCORE` | 默认推荐，按文档累积分数，并用 MaxScore 剪枝 | 通用场景 |
| `DAAT_WAND` | 使用 WAND 思路跳过不可能进入 TopK 的候选 | 查询较短、TopK 较小、追求速度 |
| `TAAT_NAIVE` | 按 term 逐个遍历，逻辑简单 | 小数据量、教学、测试 |

### 6.1 DAAT_MAXSCORE

可以理解为：

```text
按文档逐个计算分数，同时跳过明显不可能进入 TopK 的文档。
```

这是比较稳妥的默认选择。

推荐写法：

```python
params={
    "inverted_index_algo": "DAAT_MAXSCORE"
}
```

### 6.2 DAAT_WAND

可以理解为：

```text
优先跳过低价值候选文档，减少完整打分次数。
```

它更偏性能优化。

如果你之前看到旧写法：

```python
index_type="SPARSE_WAND"
```

新版可以改成：

```python
index_type="SPARSE_INVERTED_INDEX"
params={
    "inverted_index_algo": "DAAT_WAND"
}
```

### 6.3 TAAT_NAIVE

可以理解为：

```text
按查询词一个一个遍历候选文档，然后累积分数。
```

它更适合教学理解，不是大规模生产环境的首选。

---

## 7. drop_ratio_search 参数

搜索时还可以设置：

```python
search_params={
    "metric_type": "IP",
    "params": {
        "drop_ratio_search": 0.2
    }
}
```

它的作用是：

```text
忽略查询稀疏向量中权重较小的一部分维度。
```

例如：

```python
{
    105: 0.05,
    927: 1.42,
    3051: 0.76,
    8812: 0.03
}
```

如果设置：

```python
"drop_ratio_search": 0.2
```

就可以丢弃一部分权重很小的维度，减少噪声，提高检索效率。

但需要注意：

```text
drop_ratio_search 设置越大，速度可能越快，但召回率可能下降。
```

教学阶段可以先设置为：

```python
"drop_ratio_search": 0.0
```

或者不设置，让 Milvus 使用默认行为。

---

## 8. RAG 混合检索中的完整索引设计

在 RAG 系统中，一条文档 chunk 通常同时保存稠密向量和稀疏向量：

```python
{
    "id": 1,
    "text": "Milvus 支持稠密向量、稀疏向量和混合检索。",
    "source": "milvus教程.md",
    "dense_vector": [0.023, -0.118, 0.452, ...],
    "sparse_vector": {105: 0.83, 927: 1.42, 3051: 0.76}
}
```

因此通常需要创建两个索引：

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://192.168.88.100:19530")

index_params = MilvusClient.prepare_index_params()

# 1. 稠密向量索引
index_params.add_index(
    field_name="dense_vector",
    index_name="dense_index",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={"nlist": 128}
)

# 2. 稀疏向量索引
index_params.add_index(
    field_name="sparse_vector",
    index_name="sparse_index",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",
    params={
        "inverted_index_algo": "DAAT_MAXSCORE"
    }
)

client.create_index(
    collection_name="doc_crud_collection",
    index_params=index_params
)
```

对应关系是：

```text
dense_vector  -> IVF_FLAT / HNSW 等稠密向量索引
sparse_vector -> SPARSE_INVERTED_INDEX 稀疏向量索引
```

---

## 9. 最小记忆版

Milvus 稀疏向量索引可以这样记：

```text
稀疏向量字段类型：SPARSE_FLOAT_VECTOR
稀疏向量索引类型：SPARSE_INVERTED_INDEX
稀疏向量相似度：IP
常用算法参数：DAAT_MAXSCORE
性能优化算法：DAAT_WAND
旧索引类型：SPARSE_WAND，已逐步废弃
```

最推荐写法：

```python
index_params.add_index(
    field_name="sparse_vector",
    index_name="sparse_index",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",
    params={
        "inverted_index_algo": "DAAT_MAXSCORE"
    }
)
```

如果想使用 WAND 思路，不要优先写：

```python
index_type="SPARSE_WAND"
```

而是写：

```python
index_type="SPARSE_INVERTED_INDEX"
params={
    "inverted_index_algo": "DAAT_WAND"
}
```

---

## 10. 一句话总结

```text
Milvus 中稀疏向量推荐使用 SPARSE_INVERTED_INDEX。
SPARSE_WAND 属于旧写法，新版建议用 SPARSE_INVERTED_INDEX + DAAT_WAND 替代。
```
