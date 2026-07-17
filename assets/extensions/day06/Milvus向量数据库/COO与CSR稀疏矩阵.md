# COO 与 CSR 稀疏矩阵格式

## 一、学习目标

学完本知识点后，你能够：

1. 理解什么是稀疏向量和稀疏矩阵。
2. 说清楚 COO 和 CSR 的基本区别。
3. 理解 `row.col`、`row.indices`、`row.data` 分别表示什么。
4. 看懂 BGE-M3 稀疏向量转换代码。
5. 将 COO 或 CSR 格式转换为 Milvus 可以接收的字典格式。

---

## 二、为什么需要学习 COO 和 CSR？

在 RAG 混合检索中，BGE-M3 可以同时生成：

- 稠密向量：用于语义检索。
- 稀疏向量：用于关键词和词项匹配。

例如，一条文本生成的稀疏向量可能有几万维：

```python
[0, 0, 0, 0.72, 0, 0, 0, 0.51, 0, ...]
```

虽然维度很高，但绝大部分位置都是 `0`。

如果把所有的 `0` 都保存下来，会造成：

- 浪费内存。
- 浪费磁盘空间。
- 降低计算效率。

因此，稀疏向量通常只保存非零元素：

```python
{
    3: 0.72,
    7: 0.51
}
```

含义是：

```text
第 3 个位置的值是 0.72
第 7 个位置的值是 0.51
```

COO 和 CSR 就是两种常见的稀疏矩阵存储格式。

需要特别注意：

> COO 和 CSR 不是两种不同的向量，而是同一份稀疏数据的两种存储方式。

---

## 三、什么是稀疏矩阵？

先观察一个普通矩阵：

```text
[
    [0,   0, 0.8, 0,   0.5],
    [0.3, 0, 0,   0.6, 0  ]
]
```

这个矩阵一共有 10 个元素，但只有 4 个非零值。

非零元素分别是：

```text
第 0 行，第 2 列，值为 0.8
第 0 行，第 4 列，值为 0.5
第 1 行，第 0 列，值为 0.3
第 1 行，第 3 列，值为 0.6
```

稀疏矩阵不会保存所有的 `0`，而是重点保存：

1. 非零元素在哪里。
2. 非零元素的值是多少。

---

## 四、COO 格式

### 4.1 COO 是什么？

COO 全称：

```text
Coordinate Format
坐标格式
```

COO 的核心思想是：

> 每一个非零元素，都保存它的行坐标、列坐标和值。

还是下面这个矩阵：

```text
[
    [0,   0, 0.8, 0,   0.5],
    [0.3, 0, 0,   0.6, 0  ]
]
```

COO 格式会保存成三组数据：

```python
row = [0,   0,   1,   1]
col = [2,   4,   0,   3]
data = [0.8, 0.5, 0.3, 0.6]
```

三组数据要按照相同位置一起阅读。

第一个位置：

```text
row[0]  = 0
col[0]  = 2
data[0] = 0.8
```

表示：

```text
第 0 行，第 2 列，值为 0.8
```

第二个位置：

```text
row[1]  = 0
col[1]  = 4
data[1] = 0.5
```

表示：

```text
第 0 行，第 4 列，值为 0.5
```

因此，COO 可以理解为一张坐标表：

| 行号 row | 列号 col | 数值 data |
|---:|---:|---:|
| 0 | 2 | 0.8 |
| 0 | 4 | 0.5 |
| 1 | 0 | 0.3 |
| 1 | 3 | 0.6 |

### 4.2 COO 常见属性

在 SciPy 中，COO 稀疏矩阵常见属性有：

```python
matrix.row
matrix.col
matrix.data
```

| 属性 | 含义 |
|---|---|
| `row` | 非零元素所在的行索引 |
| `col` | 非零元素所在的列索引 |
| `data` | 非零元素的实际值 |

### 4.3 COO 适合什么场景？

COO 比较适合：

- 创建稀疏矩阵。
- 不断添加新的非零元素。
- 直接描述“某行某列是什么值”。
- 作为其他稀疏格式的中间转换格式。

可以把 COO 理解为：

> 一张记录非零元素坐标的明细表。

---

## 五、CSR 格式

### 5.1 CSR 是什么？

CSR 全称：

```text
Compressed Sparse Row
压缩稀疏行格式
```

CSR 的核心思想是：

> 按照行来压缩存储稀疏矩阵。

仍然使用这个矩阵：

```text
[
    [0,   0, 0.8, 0,   0.5],
    [0.3, 0, 0,   0.6, 0  ]
]
```

CSR 通常使用三组数据：

```python
data = [0.8, 0.5, 0.3, 0.6]
indices = [2, 4, 0, 3]
indptr = [0, 2, 4]
```

### 5.2 `data` 表示什么？

```python
data = [0.8, 0.5, 0.3, 0.6]
```

`data` 保存所有非零元素的值。

### 5.3 `indices` 表示什么？

```python
indices = [2, 4, 0, 3]
```

`indices` 保存每个非零元素所在的列索引。

```text
0.8 位于第 2 列
0.5 位于第 4 列
0.3 位于第 0 列
0.6 位于第 3 列
```

### 5.4 `indptr` 表示什么？

```python
indptr = [0, 2, 4]
```

`indptr` 用于说明每一行的数据，在 `data` 和 `indices` 中从哪里开始、到哪里结束。

读取第 0 行：

```python
data[indptr[0]:indptr[1]]
```

也就是：

```python
data[0:2]
```

得到：

```python
[0.8, 0.5]
```

对应列索引：

```python
indices[0:2]
```

得到：

```python
[2, 4]
```

所以第 0 行是：

```text
第 2 列 → 0.8
第 4 列 → 0.5
```

读取第 1 行：

```python
data[indptr[1]:indptr[2]]
```

也就是：

```python
data[2:4]
```

得到：

```python
[0.3, 0.6]
```

对应列索引：

```python
indices[2:4]
```

得到：

```python
[0, 3]
```

所以第 1 行是：

```text
第 0 列 → 0.3
第 3 列 → 0.6
```

### 5.5 CSR 常见属性

在 SciPy 中，CSR 稀疏矩阵常见属性有：

```python
matrix.data
matrix.indices
matrix.indptr
```

| 属性 | 含义 |
|---|---|
| `data` | 保存非零元素的值 |
| `indices` | 保存非零元素的列索引 |
| `indptr` | 保存每一行的起止位置 |

### 5.6 CSR 适合什么场景？

CSR 比较适合：

- 按行读取数据。
- 获取某一条文本的稀疏向量。
- 矩阵计算。
- 稀疏向量检索。
- 批量处理多条文本。

在文本向量场景中：

```text
一条文本 = 稀疏矩阵中的一行
```

因此 CSR 非常适合 BGE-M3 这类批量文本向量化任务。

可以把 CSR 理解为：

> 把所有非零数据按行排列，并记录每一行从哪里开始。

---

## 六、COO 和 CSR 的区别

| 对比项 | COO | CSR |
|---|---|---|
| 中文名称 | 坐标格式 | 压缩稀疏行格式 |
| 核心思想 | 保存每个非零值的行、列坐标 | 按行压缩存储 |
| 列索引属性 | `col` | `indices` |
| 非零值属性 | `data` | `data` |
| 行信息 | `row` | `indptr` |
| 适合场景 | 创建、添加、转换 | 按行读取、计算、检索 |
| 代码读取 | `row.col` | `row.indices` |

最小记忆：

```text
COO：使用 row、col、data，像坐标表。

CSR：使用 indices、indptr、data，按行压缩。

两者的 data 都表示非零值。
```

---

## 七、COO 和 CSR 的格式由谁决定？

在 BGE-M3 项目中，调用代码通常是：

```python
query_embedding = self.embedding_function([query])
```

这里的 `self.embedding_function` 是：

```python
BGEM3EmbeddingFunction
```

BGE-M3 模型负责计算：

```text
哪些维度非零
每个维度的权重是多少
```

但是，最终使用 COO 还是 CSR，通常由下面这些因素决定：

- `BGEM3EmbeddingFunction` 内部实现。
- `milvus-model` 版本。
- `FlagEmbedding` 版本。
- SciPy 版本。
- 稀疏矩阵的切片方式。
- 是否调用了 `tocoo()` 或 `tocsr()`。

因此：

> 模型决定稀疏向量的数据内容，SciPy 和相关封装库决定数据使用 COO 还是 CSR 格式保存。

---

## 八、SciPy 在哪里？

业务代码中可能没有直接写：

```python
import scipy
```

但 `BGEM3EmbeddingFunction` 内部可能使用了 SciPy 的稀疏矩阵。

调用流程大致是：

```text
你的业务代码
    ↓
BGEM3EmbeddingFunction
    ↓
milvus-model / FlagEmbedding
    ↓
SciPy 稀疏矩阵
    ↓
query_embedding["sparse"]
```

可以通过下面代码查看实际返回类型：

```python
query_embedding = self.embedding_function(["什么是人工智能"])

sparse_matrix = query_embedding["sparse"]

print(type(sparse_matrix))
print(type(sparse_matrix[0]))
```

可能输出：

```text
<class 'scipy.sparse._csr.csr_matrix'>
<class 'scipy.sparse._csr.csr_matrix'>
```

如果类型中出现：

```text
scipy.sparse
```

说明这个对象来自 SciPy。

还可以查看矩阵格式：

```python
print(sparse_matrix.getformat())
```

可能输出：

```text
csr
```

---

## 九、为什么代码需要判断 COO 和 CSR？

项目中的代码：

```python
row = query_embedding["sparse"][0]

if hasattr(row, "col"):
    indices, values = row.col, row.data
else:
    indices, values = row.indices, row.data
```

其目的就是兼容两种稀疏矩阵格式。

逻辑可以理解为：

```text
如果对象有 col 属性：
    说明按照 COO 格式读取

否则：
    按照 CSR 格式读取
```

COO 读取：

```python
indices = row.col
values = row.data
```

CSR 读取：

```python
indices = row.indices
values = row.data
```

---

## 十、为什么要取 `[0]`？

代码：

```python
query_embedding = self.embedding_function([query])
```

传入的是一个列表：

```python
["什么是人工智能"]
```

即使只有一条文本，模型也会按照批量数据处理。

结果可以理解为：

```python
query_embedding = {
    "dense": [
        第一条文本的稠密向量
    ],
    "sparse": [
        第一条文本的稀疏向量
    ]
}
```

因此：

```python
query_embedding["sparse"][0]
```

表示：

> 获取第 0 条文本对应的完整稀疏向量。

这里的 `[0]` 不是获取稀疏向量的第 0 个维度，而是获取第 0 条文本。

---

## 十一、如何转换成 Milvus 字典格式？

假设获取到：

```python
indices = [12, 485, 1024]
values = [0.67, 0.57, 0.60]
```

可以使用 `zip()` 组合：

```python
for index, value in zip(indices, values):
    print(index, value)
```

输出：

```text
12 0.67
485 0.57
1024 0.60
```

转换为字典：

```python
sparse_vector = {}

for index, value in zip(indices, values):
    sparse_vector[int(index)] = float(value)
```

最终结果：

```python
{
    12: 0.67,
    485: 0.57,
    1024: 0.60
}
```

这就是 Milvus 可以接收的稀疏向量格式：

```text
维度索引 → 权重
```

---

## 十二、推荐代码写法

### 12.1 同时兼容 COO 和 CSR

```python
row = query_embedding["sparse"][0]

if hasattr(row, "col"):
    # COO 格式：col 保存非零元素的列索引
    indices = row.col
    values = row.data

elif hasattr(row, "indices"):
    # CSR 格式：indices 保存非零元素的列索引
    indices = row.indices
    values = row.data

else:
    raise TypeError(
        f"不支持的稀疏向量格式：{type(row).__name__}"
    )

sparse_query_vector = {
    int(index): float(value)
    for index, value in zip(indices, values)
}
```

相比直接使用 `else`，这里使用：

```python
elif hasattr(row, "indices")
```

更加严谨。

因为没有 `col` 属性，不一定就代表对象一定存在 `indices` 属性。

### 12.2 统一转换为 CSR

也可以主动将稀疏矩阵统一转换为 CSR：

```python
sparse_matrix = query_embedding["sparse"].tocsr()

row = sparse_matrix.getrow(0)

indices = row.indices
values = row.data
```

然后转换为字典：

```python
sparse_query_vector = {
    int(index): float(value)
    for index, value in zip(indices, values)
}
```

完整代码：

```python
query_embedding = self.embedding_function([query])

sparse_matrix = query_embedding["sparse"].tocsr()
row = sparse_matrix.getrow(0)

sparse_query_vector = {
    int(index): float(value)
    for index, value in zip(row.indices, row.data)
}
```

这种写法的优点是：

- 后续统一按照 CSR 处理。
- 不再需要判断 `col` 和 `indices`。
- 代码更简洁。
- 更适合按行获取文本向量。

---

## 十三、最小记忆版

```text
COO 和 CSR 不是两种向量，
而是同一份稀疏数据的两种存储格式。

COO：
row + col + data
适合创建和描述坐标。

CSR：
indices + indptr + data
适合按行读取和计算。

COO 的列索引：col
CSR 的列索引：indices
两者的非零值：data

BGE-M3 负责生成稀疏权重，
SciPy 等库负责使用 COO 或 CSR 保存这些数据。
```
