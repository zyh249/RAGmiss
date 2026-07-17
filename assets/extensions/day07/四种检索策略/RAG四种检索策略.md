# RAG 四种检索策略

## 一、整体说明

在 RAG 系统中，用户问题不一定都适合直接检索。根据问题复杂度和表达方式，可以选择不同的检索策略。

本文整理四种常见检索方式：

| 检索方式 | 核心思想 | 适用场景 |
|---|---|---|
| 直接检索 | 原始问题直接进入检索系统 | 问题明确、关键词清楚 |
| HyDE 检索 | 先生成假设答案，再用假设答案检索 | 问题抽象、表达不完整 |
| 子查询检索 | 把复杂问题拆成多个子问题分别检索 | 多条件、多知识点、对比类问题 |
| 回溯检索 | 把复杂问题简化成基础问题后检索 | 问题太复杂、需要背景知识 |

对应到原始 `RAGSystem` 代码中：

| 检索方式 | 原始代码方法 | 核心作用 |
|---|---|---|
| 直接检索 | `retrieve_and_merge()` 的 `else` 分支 | 原问题直接检索 |
| HyDE 检索 | `_retrieve_with_hyde()` | 先生成假设答案，再检索 |
| 子查询检索 | `_retrieve_with_subqueries()` | 拆成多个小问题分别检索 |
| 回溯检索 | `_retrieve_with_backtracking()` | 把复杂问题简化成基础问题再检索 |

---

# 二、直接检索案例

## 1. 适用场景

直接检索适合问题非常明确的情况。

比如用户已经把关键词说得很清楚，不需要模型帮忙改写、拆分或推理。

## 2. 检索示例

用户问题：

```text
什么是 BM25 算法？
```

这个问题非常明确，关键词就是：

```text
BM25
算法
```

所以可以直接拿原始问题去向量库或混合检索系统中搜索。

## 3. 直接检索代码案例

```python
class DirectRetrievalCase:
    def __init__(self, vector_store):
        # 保存向量数据库对象
        self.vector_store = vector_store

    def retrieve(self, query, source_filter=None):
        """
        直接检索：
        不改写问题，不拆分问题，不生成假设答案，
        直接用用户原始问题进行检索。
        """

        print(f"使用直接检索策略，原始问题：{query}")

        docs = self.vector_store.hybrid_search_with_rerank(
            query,
            k=5,
            source_filter=source_filter
        )

        return docs


# =========================
# 使用示例
# =========================

query = "什么是 BM25 算法？"

retriever = DirectRetrievalCase(vector_store)
docs = retriever.retrieve(query)

for i, doc in enumerate(docs, start=1):
    print(f"第 {i} 个文档：")
    print(doc.page_content)
    print("-" * 30)
```

## 4. 执行流程

```text
用户问题：什么是 BM25 算法？
        ↓
直接把问题送入 hybrid_search_with_rerank
        ↓
向量检索 + BM25 检索 + Rerank
        ↓
返回相关文档
```

## 5. 对应原始代码位置

```python
else:  # 默认或“直接检索”
    logger.info(f"使用直接检索策略 (查询: '{query}')")
    ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
        query, k=conf.RETRIEVAL_K, source_filter=source_filter
    )
```

这里就是直接检索。

---

# 三、HyDE 检索案例

## 1. 适用场景

HyDE 适合问题比较抽象、不好直接命中文档的情况。

HyDE 的完整名字是：

```text
Hypothetical Document Embeddings
```

可以理解成：

```text
先让大模型根据问题“猜一个可能的答案”，
然后用这个“假设答案”去检索文档。
```

因为假设答案通常比原始问题包含更多语义信息，所以更容易召回相关文档。

## 2. 检索示例

用户问题：

```text
为什么向量数据库适合做 RAG？
```

这个问题有点抽象，直接搜可能只命中“向量数据库”或者“RAG”关键词。

HyDE 会先生成一段假设答案，例如：

```text
向量数据库适合做 RAG，是因为它可以存储文本的向量表示，
并通过语义相似度快速召回与用户问题相关的文档片段。
在 RAG 系统中，向量数据库负责检索知识，大语言模型负责基于知识生成答案。
```

然后系统不是用原问题检索，而是用这段假设答案检索。

## 3. HyDE 检索代码案例

```python
class HyDERetrievalCase:
    def __init__(self, vector_store, llm):
        # 保存向量数据库对象
        self.vector_store = vector_store

        # 保存大语言模型调用函数
        self.llm = llm

    def build_hyde_prompt(self, query):
        """
        构造 HyDE Prompt：
        让大模型根据用户问题，生成一个可能的假设答案。
        """
        prompt = f"""
请根据下面的问题，生成一段可能的专业答案。
注意：这段答案不是最终答案，只用于帮助检索相关文档。

用户问题：
{query}

请生成假设答案：
"""
        return prompt

    def retrieve(self, query):
        """
        HyDE 检索：
        第一步：用 LLM 生成假设答案
        第二步：用假设答案进行检索
        """

        print(f"使用 HyDE 检索策略，原始问题：{query}")

        # 1. 构造 HyDE Prompt
        prompt = self.build_hyde_prompt(query)

        # 2. 调用大模型生成假设答案
        hypo_answer = self.llm(prompt).strip()

        print("生成的假设答案：")
        print(hypo_answer)

        # 3. 用假设答案进行检索
        docs = self.vector_store.hybrid_search_with_rerank(
            hypo_answer,
            k=5
        )

        return docs


# =========================
# 使用示例
# =========================

query = "为什么向量数据库适合做 RAG？"

retriever = HyDERetrievalCase(vector_store, llm)
docs = retriever.retrieve(query)

for i, doc in enumerate(docs, start=1):
    print(f"第 {i} 个文档：")
    print(doc.page_content)
    print("-" * 30)
```

## 4. 执行流程

```text
用户问题：为什么向量数据库适合做 RAG？
        ↓
LLM 生成假设答案
        ↓
用假设答案进行检索
        ↓
召回相关文档
        ↓
后续再交给大模型生成最终答案
```

## 5. 对应原始代码位置

```python
def _retrieve_with_hyde(self, query):
    logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")

    hyde_prompt_template = RAGPrompts.hyde_prompt()

    try:
        hypo_answer = self.llm(hyde_prompt_template.format(query=query)).strip()

        logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")

        return self.vector_store.hybrid_search_with_rerank(
            hypo_answer, k=conf.RETRIEVAL_K
        )

    except Exception as e:
        logger.error(f"HyDE 策略执行失败: {e}")
        return []
```

重点是先生成假设答案：

```python
hypo_answer = self.llm(...).strip()
```

然后用假设答案检索：

```python
self.vector_store.hybrid_search_with_rerank(hypo_answer, k=conf.RETRIEVAL_K)
```

---

# 四、子查询检索案例

## 1. 适用场景

子查询检索适合一个问题里包含多个子任务的情况。

例如：

```text
BM25 和向量检索有什么区别？它们在 RAG 中分别适合什么场景？
```

这个问题里面至少包含 3 个子问题：

```text
1. BM25 是什么？
2. 向量检索是什么？
3. BM25 和向量检索在 RAG 中分别适合什么场景？
```

如果直接检索，可能召回不全。

所以更好的方式是：先拆分，再分别检索，最后合并。

## 2. 检索示例

用户问题：

```text
BM25 和向量检索有什么区别？它们在 RAG 中分别适合什么场景？
```

子查询可以拆成：

```text
BM25 检索的原理是什么？
向量检索的原理是什么？
BM25 和向量检索在 RAG 中的应用场景有什么区别？
```

然后每个子查询分别进入检索系统。

## 3. 子查询检索代码案例

```python
class SubQueryRetrievalCase:
    def __init__(self, vector_store, llm):
        # 保存向量数据库对象
        self.vector_store = vector_store

        # 保存大语言模型调用函数
        self.llm = llm

    def build_subquery_prompt(self, query):
        """
        构造子查询 Prompt：
        让大模型把复杂问题拆成多个简单问题。
        """
        prompt = f"""
请把下面这个复杂问题拆分成 3 个更容易检索的小问题。
要求：
1. 每行只输出一个子问题
2. 不要编号
3. 不要解释

复杂问题：
{query}

拆分结果：
"""
        return prompt

    def retrieve(self, query):
        """
        子查询检索：
        第一步：把复杂问题拆成多个子问题
        第二步：每个子问题分别检索
        第三步：合并所有检索结果
        第四步：根据 page_content 去重
        """

        print(f"使用子查询检索策略，原始问题：{query}")

        # 1. 生成子查询
        prompt = self.build_subquery_prompt(query)
        subqueries_text = self.llm(prompt).strip()

        # 2. 按行切分子查询
        subqueries = [
            q.strip()
            for q in subqueries_text.split("\n")
            if q.strip()
        ]

        print("生成的子查询：")
        for sub_q in subqueries:
            print(sub_q)

        # 3. 分别检索每一个子查询
        all_docs = []

        for sub_q in subqueries:
            docs = self.vector_store.hybrid_search_with_rerank(
                sub_q,
                k=5
            )

            all_docs.extend(docs)

            print(f"子查询：{sub_q}")
            print(f"检索到文档数量：{len(docs)}")

        # 4. 基于文档内容去重
        unique_docs_dict = {
            doc.page_content: doc
            for doc in all_docs
        }

        unique_docs = list(unique_docs_dict.values())

        print(f"合并前文档数量：{len(all_docs)}")
        print(f"去重后文档数量：{len(unique_docs)}")

        return unique_docs


# =========================
# 使用示例
# =========================

query = "BM25 和向量检索有什么区别？它们在 RAG 中分别适合什么场景？"

retriever = SubQueryRetrievalCase(vector_store, llm)
docs = retriever.retrieve(query)

for i, doc in enumerate(docs, start=1):
    print(f"第 {i} 个文档：")
    print(doc.page_content)
    print("-" * 30)
```

## 4. 执行流程

```text
复杂问题：
BM25 和向量检索有什么区别？它们在 RAG 中分别适合什么场景？
        ↓
LLM 拆分子查询：
1. BM25 检索的原理是什么？
2. 向量检索的原理是什么？
3. BM25 和向量检索在 RAG 中的应用场景有什么区别？
        ↓
每个子查询分别检索
        ↓
合并所有文档
        ↓
根据 page_content 去重
        ↓
返回最终候选文档
```

## 5. 对应原始代码位置

```python
def _retrieve_with_subqueries(self, query):
    logger.info(f"使用子查询策略进行检索 (查询: '{query}')")

    subquery_prompt_template = RAGPrompts.subquery_prompt()

    try:
        subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()

        subqueries = [
            q.strip()
            for q in subqueries_text.split("\n")
            if q.strip()
        ]

        all_docs = []

        for sub_q in subqueries:
            docs = self.vector_store.hybrid_search_with_rerank(
                sub_q, k=conf.RETRIEVAL_K
            )

            all_docs.extend(docs)

        unique_docs_dict = {
            doc.page_content: doc
            for doc in all_docs
        }

        unique_docs = list(unique_docs_dict.values())

        return unique_docs

    except Exception as e:
        logger.error(f"子查询策略执行失败: {e}")
        return []
```

这里最重要的是两步。

第一步，生成多个子问题：

```python
subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
```

第二步，每个子问题单独检索：

```python
for sub_q in subqueries:
    docs = self.vector_store.hybrid_search_with_rerank(sub_q, k=conf.RETRIEVAL_K)
    all_docs.extend(docs)
```

---

# 五、回溯检索案例

## 1. 适用场景

回溯检索适合原问题太复杂，不适合直接检索的情况。

它的思路是：

```text
先不要直接解决最终问题，
而是先把问题退回到一个更基础、更容易检索的问题。
```

例如用户问：

```text
为什么 RAG 系统在长文档问答中容易出现答案不准确？
```

这个问题比较复杂，里面涉及：

```text
长文档切分
召回不准确
上下文截断
重排序
生成幻觉
```

直接检索可能不稳定。

可以先回溯成基础问题：

```text
RAG 系统的答案准确性受哪些因素影响？
```

然后用这个更基础的问题进行检索。

## 2. 检索示例

原始问题：

```text
为什么 RAG 系统在长文档问答中容易出现答案不准确？
```

回溯问题：

```text
RAG 系统的答案准确性受哪些因素影响？
```

系统先检索基础问题，再用检索到的背景知识帮助回答原始问题。

## 3. 回溯检索代码案例

```python
class BacktrackingRetrievalCase:
    def __init__(self, vector_store, llm):
        # 保存向量数据库对象
        self.vector_store = vector_store

        # 保存大语言模型调用函数
        self.llm = llm

    def build_backtracking_prompt(self, query):
        """
        构造回溯 Prompt：
        让大模型把复杂问题简化成一个更基础、更容易检索的问题。
        """
        prompt = f"""
请把下面这个复杂问题，改写成一个更基础、更容易检索的问题。
要求：
1. 只输出一个问题
2. 不要解释
3. 保留原问题的核心方向

复杂问题：
{query}

回溯后的基础问题：
"""
        return prompt

    def retrieve(self, query):
        """
        回溯检索：
        第一步：把复杂问题改写成基础问题
        第二步：用基础问题进行检索
        """

        print(f"使用回溯检索策略，原始问题：{query}")

        # 1. 生成回溯问题
        prompt = self.build_backtracking_prompt(query)
        simplified_query = self.llm(prompt).strip()

        print("生成的回溯问题：")
        print(simplified_query)

        # 2. 使用回溯问题进行检索
        docs = self.vector_store.hybrid_search_with_rerank(
            simplified_query,
            k=5
        )

        return docs


# =========================
# 使用示例
# =========================

query = "为什么 RAG 系统在长文档问答中容易出现答案不准确？"

retriever = BacktrackingRetrievalCase(vector_store, llm)
docs = retriever.retrieve(query)

for i, doc in enumerate(docs, start=1):
    print(f"第 {i} 个文档：")
    print(doc.page_content)
    print("-" * 30)
```

## 4. 执行流程

```text
原始复杂问题：
为什么 RAG 系统在长文档问答中容易出现答案不准确？
        ↓
LLM 生成回溯问题：
RAG 系统的答案准确性受哪些因素影响？
        ↓
用回溯问题进行检索
        ↓
召回基础背景知识
        ↓
再结合原始问题生成最终答案
```

## 5. 对应原始代码位置

```python
def _retrieve_with_backtracking(self, query):
    logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")

    backtrack_prompt_template = RAGPrompts.backtracking_prompt()

    try:
        simplified_query = self.llm(
            backtrack_prompt_template.format(query=query)
        ).strip()

        logger.info(f"生成的回溯问题: '{simplified_query}'")

        return self.vector_store.hybrid_search_with_rerank(
            simplified_query, k=conf.RETRIEVAL_K
        )

    except Exception as e:
        logger.error(f"回溯问题策略执行失败: {e}")
        return []
```

重点是先生成一个更基础的问题：

```python
simplified_query = self.llm(...).strip()
```

然后用基础问题去检索：

```python
self.vector_store.hybrid_search_with_rerank(simplified_query, k=conf.RETRIEVAL_K)
```

---

# 六、四种策略整合成教学版主类

上面是 4 个独立小案例。

实际项目中可以像原始代码一样，把它们统一放到一个 `RAGSystem` 里面。

下面是一个更清晰的教学版结构：

```python
class SimpleRAGSystem:
    def __init__(self, vector_store, llm):
        # 向量数据库对象
        self.vector_store = vector_store

        # 大语言模型调用函数
        self.llm = llm

    def direct_retrieve(self, query, source_filter=None):
        """
        方式一：直接检索
        适合：问题明确、关键词清楚
        """

        print("当前策略：直接检索")

        docs = self.vector_store.hybrid_search_with_rerank(
            query,
            k=5,
            source_filter=source_filter
        )

        return docs

    def hyde_retrieve(self, query):
        """
        方式二：HyDE 检索
        适合：问题抽象、表达不完整
        """

        print("当前策略：HyDE 检索")

        prompt = f"""
请根据用户问题，生成一段可能的专业答案。
这段答案只用于检索，不是最终答案。

用户问题：
{query}

假设答案：
"""

        hypo_answer = self.llm(prompt).strip()

        print("HyDE 生成的假设答案：")
        print(hypo_answer)

        docs = self.vector_store.hybrid_search_with_rerank(
            hypo_answer,
            k=5
        )

        return docs

    def subquery_retrieve(self, query):
        """
        方式三：子查询检索
        适合：复杂问题、多条件问题、对比问题
        """

        print("当前策略：子查询检索")

        prompt = f"""
请把下面的问题拆成 3 个适合检索的子问题。
每行只输出一个子问题，不要编号。

原始问题：
{query}

子问题：
"""

        subqueries_text = self.llm(prompt).strip()

        subqueries = [
            q.strip()
            for q in subqueries_text.split("\n")
            if q.strip()
        ]

        print("拆分后的子查询：")
        for sub_q in subqueries:
            print(sub_q)

        all_docs = []

        for sub_q in subqueries:
            docs = self.vector_store.hybrid_search_with_rerank(
                sub_q,
                k=5
            )

            all_docs.extend(docs)

        unique_docs_dict = {
            doc.page_content: doc
            for doc in all_docs
        }

        unique_docs = list(unique_docs_dict.values())

        return unique_docs

    def backtracking_retrieve(self, query):
        """
        方式四：回溯检索
        适合：原问题太复杂、需要先补背景知识
        """

        print("当前策略：回溯检索")

        prompt = f"""
请把下面这个复杂问题，改写成一个更基础、更容易检索的问题。
只输出一个问题，不要解释。

复杂问题：
{query}

基础问题：
"""

        simplified_query = self.llm(prompt).strip()

        print("生成的回溯问题：")
        print(simplified_query)

        docs = self.vector_store.hybrid_search_with_rerank(
            simplified_query,
            k=5
        )

        return docs
```

---

# 七、四个案例调用方式

## 1. 直接检索调用

```python
rag = SimpleRAGSystem(vector_store, llm)

query = "什么是 BM25 算法？"

docs = rag.direct_retrieve(query)
```

## 2. HyDE 检索调用

```python
rag = SimpleRAGSystem(vector_store, llm)

query = "为什么向量数据库适合做 RAG？"

docs = rag.hyde_retrieve(query)
```

## 3. 子查询检索调用

```python
rag = SimpleRAGSystem(vector_store, llm)

query = "BM25 和向量检索有什么区别？它们在 RAG 中分别适合什么场景？"

docs = rag.subquery_retrieve(query)
```

## 4. 回溯检索调用

```python
rag = SimpleRAGSystem(vector_store, llm)

query = "为什么 RAG 系统在长文档问答中容易出现答案不准确？"

docs = rag.backtracking_retrieve(query)
```

---

# 八、四种检索方式的核心区别

| 检索方式 | 是否改写问题 | 是否生成新问题 | 是否多次检索 | 适合场景 |
|---|---:|---:|---:|---|
| 直接检索 | 否 | 否 | 否 | 问题明确 |
| HyDE 检索 | 是 | 生成假设答案 | 否 | 问题抽象 |
| 子查询检索 | 是 | 生成多个子问题 | 是 | 问题复杂、多个条件 |
| 回溯检索 | 是 | 生成基础问题 | 否 | 原问题太难直接检索 |

---

# 九、结合原始代码时需要注意的问题

原始代码整体思路是对的，但是有几个地方需要修正。

## 1. `config` 和 `conf` 混用了

开头导入的是：

```python
from base.config import config
```

但是后面很多地方写的是：

```python
conf.RETRIEVAL_K
conf.CANDIDATE_M
conf.CUSTOMER_SERVICE_PHONE
```

如果没有额外定义 `conf`，这里会报错。

应该统一成：

```python
config.RETRIEVAL_K
config.CANDIDATE_M
config.CUSTOMER_SERVICE_PHONE
```

例如：

```python
return self.vector_store.hybrid_search_with_rerank(
    hypo_answer,
    k=config.RETRIEVAL_K
)
```

## 2. `generate_answer()` 里面有一处缩进问题

原始代码中这段：

```python
if query_category == "通用知识":
    logger.info("查询为通用知识，直接调用 LLM")
     prompt_input = self.rag_prompt.format(
        context="", question=query, history="", phone=config.CUSTOMER_SERVICE_PHONE
    )
```

这里：

```python
 prompt_input
```

前面多了一个空格，会导致 Python 报缩进错误。

应该改成：

```python
if query_category == "通用知识":
    logger.info("查询为通用知识，直接调用 LLM")

    prompt_input = self.rag_prompt.format(
        context="",
        question=query,
        history="",
        phone=config.CUSTOMER_SERVICE_PHONE
    )
```

## 3. `rag_prompt.format()` 的参数要统一

通用知识这里写了：

```python
self.rag_prompt.format(
    context="", question=query, history="", phone=config.CUSTOMER_SERVICE_PHONE
)
```

但是专业咨询这里写的是：

```python
self.rag_prompt.format(
    context=context, question=query, phone=conf.CUSTOMER_SERVICE_PHONE
)
```

如果 Prompt 模板里需要 `history`，那么两处都要传：

```python
prompt_input = self.rag_prompt.format(
    context=context,
    question=query,
    history="",
    phone=config.CUSTOMER_SERVICE_PHONE
)
```

否则可能报：

```text
KeyError: 'history'
```

---

# 十、最终记忆版

```text
直接检索：
原问题 → 检索
适合：问题明确

HyDE 检索：
原问题 → 假设答案 → 检索
适合：问题抽象

子查询检索：
复杂问题 → 多个子问题 → 分别检索 → 合并
适合：多条件、多知识点问题

回溯检索：
复杂问题 → 基础问题 → 检索 → 回到原问题
适合：需要背景知识、推理链较长的问题
```

最核心的一句话：

```text
直接检索是不改问题；
HyDE 是把问题变成假设答案；
子查询是把一个问题拆成多个问题；
回溯检索是把复杂问题退回成基础问题。
```
