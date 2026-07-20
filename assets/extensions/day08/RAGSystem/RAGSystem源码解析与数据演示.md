# RAGSystem 源码解析与数据演示

# 学习目标

学习完成后，应能够理解：

1. `RAGSystem` 类在整个 RAG 系统中的作用。
2. `generate_answer()`、`retrieve_and_merge()` 和不同检索策略函数之间的调用关系。
3. HyDE、子查询检索、回溯问题检索、直接检索分别适合什么场景。
4. 用户问题如何一步步变成检索文档、上下文和最终答案。
5. 源码中可能存在的变量、缩进和参数问题。

---

# 一、先看整体流程

这份代码不要一开始就逐行死记，先记住一句话：

> `RAGSystem` 是一个 RAG 问答总控制器。
>
> 它负责：判断问题类型、选择检索策略、检索知识库、拼接上下文、调用大模型生成答案。

用户输入一个问题：

```python
query = "学校退费流程是什么？"
```

系统整体流程如下：

```text
用户问题
   ↓
generate_answer()
   ↓
判断问题类型：通用知识 / 专业咨询
   ↓
如果是通用知识：直接问大模型
如果是专业咨询：进入 RAG 流程
   ↓
选择检索策略
   ↓
retrieve_and_merge()
   ↓
根据策略调用不同检索方法
   ↓
从向量库 / BM25 / rerank 中拿到文档
   ↓
拼接 context
   ↓
把 context + question 填入 prompt
   ↓
调用 LLM 生成最终答案
```

---

# 二、RAGSystem 类整体作用

源码中定义了一个类：

```python
class RAGSystem:
```

这个类表示一个完整的 RAG 问答系统。

它里面主要包含 6 个方法：

| 方法 | 作用 |
|---|---|
| `__init__()` | 初始化 RAG 系统需要的组件 |
| `_retrieve_with_hyde()` | 使用 HyDE 策略检索 |
| `_retrieve_with_subqueries()` | 使用子查询策略检索 |
| `_retrieve_with_backtracking()` | 使用回溯问题策略检索 |
| `retrieve_and_merge()` | 根据策略检索文档并合并结果 |
| `generate_answer()` | 对外提供问答入口，生成最终答案 |

其中最重要的是：

```python
generate_answer()
```

因为用户真正调用的一般就是它。

---

# 三、`__init__()`：初始化系统

## 1. 源码

```python
def __init__(self, vector_store, llm):
    self.vector_store = vector_store
    self.llm = llm
    self.rag_prompt = RAGPrompts.rag_prompt()
    self.query_classifier = QueryClassifier(model_path=f'{config.MODELS_DIR}/bert_query_classifier')
    self.strategy_selector = StrategySelector()
```

## 2. 函数作用

`__init__()` 是初始化方法。

当创建 `RAGSystem` 对象时，它会自动执行。

例如：

```python
rag_system = RAGSystem(vector_store, llm)
```

这行代码执行时，会自动调用：

```python
__init__(self, vector_store, llm)
```

它的任务是把系统运行需要的组件准备好。

---

## 3. 参数解释

### 3.1 `vector_store`

`vector_store` 表示向量数据库对象。

它负责：

```text
根据问题去 Milvus / 向量库 / BM25 中检索相关文档。
```

例如用户问：

```text
学校退费流程是什么？
```

`vector_store` 可能会检索出：

```text
文档1：学生申请退费时，需要提交退费申请表。
文档2：退费申请由教务老师审核，财务部门处理。
文档3：退费周期一般为 7 到 15 个工作日。
```

---

### 3.2 `llm`

`llm` 表示大语言模型调用函数。

它负责：

```text
根据 prompt 生成答案。
```

例如：

```python
answer = self.llm(prompt_input)
```

可以理解为：

```text
把整理好的问题和资料发给大模型，让大模型回答。
```

---

### 3.3 `self.rag_prompt`

```python
self.rag_prompt = RAGPrompts.rag_prompt()
```

这行代码用于获取 RAG 问答模板。

模板可能长这样：

```text
请根据以下上下文回答用户问题。

上下文：
{context}

用户问题：
{question}

如果无法回答，请提示联系人工客服：{phone}
```

后续代码会通过：

```python
self.rag_prompt.format(...)
```

把真实问题、上下文和客服电话填进去。

---

### 3.4 `self.query_classifier`

```python
self.query_classifier = QueryClassifier(model_path=f'{config.MODELS_DIR}/bert_query_classifier')
```

这行代码初始化问题分类器。

它负责判断用户问题属于哪一类。

| 用户问题 | 分类结果 |
|---|---|
| 今天北京天气怎么样？ | 通用知识 |
| 学校退费流程是什么？ | 专业咨询 |
| 课程有效期多久？ | 专业咨询 |

如果是通用知识，可以直接让大模型回答。

如果是专业咨询，就需要走 RAG 检索流程。

---

### 3.5 `self.strategy_selector`

```python
self.strategy_selector = StrategySelector()
```

这行代码初始化策略选择器。

它负责判断这个问题应该使用哪种检索策略。

| 用户问题 | 适合策略 |
|---|---|
| 退费流程是什么？ | 直接检索 |
| 我报名后不想学了，钱还能退吗？ | 回溯问题检索 |
| 课程价格、有效期、售后服务分别是什么？ | 子查询检索 |
| 如何提升 RAG 系统效果？ | HyDE 检索 |

---

# 四、`_retrieve_with_hyde()`：HyDE 检索

## 1. 源码核心

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

---

## 2. HyDE 是什么

HyDE 可以理解为：

```text
先让大模型假装回答一次，再用这个假设答案去检索。
```

普通检索是：

```text
用户问题 → 检索
```

HyDE 检索是：

```text
用户问题 → 生成假设答案 → 用假设答案检索
```

---

## 3. 为什么需要 HyDE

有些用户问题很口语化，关键词不明显，直接检索效果不好。

例如用户问：

```text
学到一半不想学了，钱咋办？
```

这个问题里面没有明显关键词：

```text
退费、退款、退学、费用
```

直接检索时，可能找不到最相关的知识库文档。

HyDE 会先让大模型生成一个假设答案：

```text
如果学员中途不想继续学习，可以根据机构退费政策申请退费。
通常需要提交退费申请，由教务或财务审核。
```

然后用这个假设答案去检索。

假设答案中包含了更多检索关键词：

```text
退费、申请、审核、财务、政策
```

所以检索效果可能更好。

---

## 4. 代入数据演示

假设用户问题是：

```python
query = "学到一半不想学了，钱咋办？"
```

### 第一步：获取 HyDE prompt

```python
hyde_prompt_template = RAGPrompts.hyde_prompt()
```

假设模板是：

```text
请根据用户问题生成一个可能的标准答案：
用户问题：{query}
```

### 第二步：填入用户问题

```python
hyde_prompt_template.format(query=query)
```

得到：

```text
请根据用户问题生成一个可能的标准答案：
用户问题：学到一半不想学了，钱咋办？
```

### 第三步：调用大模型

```python
hypo_answer = self.llm(...)
```

得到假设答案：

```python
hypo_answer = "学员中途不想继续学习时，可以根据退费规则提交退费申请，由教务和财务审核后处理。"
```

### 第四步：用假设答案检索

```python
self.vector_store.hybrid_search_with_rerank(
    hypo_answer,
    k=conf.RETRIEVAL_K
)
```

假设：

```python
conf.RETRIEVAL_K = 5
```

表示先检索前 5 个候选文档。

返回结果可能是：

```python
[
    Document(page_content="学员申请退费时，需要填写退费申请表。"),
    Document(page_content="退费申请由教务部门审核，财务部门在 7 到 15 个工作日内处理。"),
    Document(page_content="课程开始后退费金额需根据学习进度计算。")
]
```

---

## 5. 这个函数最终返回什么

返回的是：

```text
和“假设答案”最相关的文档列表。
```

也就是：

```python
return [Document1, Document2, Document3]
```

---

# 五、`_retrieve_with_subqueries()`：子查询检索

## 1. 源码核心

```python
def _retrieve_with_subqueries(self, query):
    logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
    subquery_prompt_template = RAGPrompts.subquery_prompt()
    try:
        subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
        subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
        logger.info(f"生成的子查询: {subqueries}")
        if not subqueries:
             logger.warning("未能生成有效的子查询")
             return []

        all_docs = []
        for sub_q in subqueries:
            docs = self.vector_store.hybrid_search_with_rerank(
                sub_q, k=conf.RETRIEVAL_K
            )
            all_docs.extend(docs)
            logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

        unique_docs_dict = {doc.page_content: doc for doc in all_docs}
        unique_docs = list(unique_docs_dict.values())

        logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
        return unique_docs

    except Exception as e:
        logger.error(f"子查询策略执行失败: {e}")
        return []
```

---

## 2. 子查询检索是什么

子查询检索就是：

```text
把一个复杂问题拆成多个简单问题，然后分别检索。
```

---

## 3. 适合什么问题

适合这种“一句话问了好几件事”的问题：

```text
课程价格是多少，有效期多久，售后服务怎么联系？
```

这个问题实际上包含 3 个小问题：

```text
1. 课程价格是多少？
2. 课程有效期多久？
3. 售后服务怎么联系？
```

如果直接检索，可能只找到其中一部分。

子查询检索会把它拆开，再分别检索。

---

## 4. 代入数据演示

假设用户问题是：

```python
query = "课程价格是多少，有效期多久，售后服务怎么联系？"
```

### 第一步：获取子查询 prompt

```python
subquery_prompt_template = RAGPrompts.subquery_prompt()
```

假设模板是：

```text
请把下面的问题拆成多个独立的小问题：
{query}
```

### 第二步：调用大模型生成子查询

```python
subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
```

大模型返回：

```text
课程价格是多少？
课程有效期多久？
售后服务怎么联系？
```

### 第三步：按换行拆分

```python
subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
```

得到：

```python
subqueries = [
    "课程价格是多少？",
    "课程有效期多久？",
    "售后服务怎么联系？"
]
```

---

## 5. 每个子查询分别检索

源码：

```python
all_docs = []

for sub_q in subqueries:
    docs = self.vector_store.hybrid_search_with_rerank(
        sub_q, k=conf.RETRIEVAL_K
    )
    all_docs.extend(docs)
```

这段代码的意思是：

```text
每一个子问题都去检索一次，然后把结果放到 all_docs 里面。
```

假设：

```python
conf.RETRIEVAL_K = 2
```

表示每个子问题检索 2 条。

---

### 子问题 1

```python
sub_q = "课程价格是多少？"
```

检索结果：

```python
docs = [
    Document(page_content="Python就业班价格为 12980 元。"),
    Document(page_content="AI大模型课程价格为 16800 元。")
]
```

加入 `all_docs` 后：

```python
all_docs = [
    "Python就业班价格为 12980 元。",
    "AI大模型课程价格为 16800 元。"
]
```

---

### 子问题 2

```python
sub_q = "课程有效期多久？"
```

检索结果：

```python
docs = [
    Document(page_content="课程有效期为报名后 2 年。"),
    Document(page_content="学员可在有效期内反复观看录播课程。")
]
```

此时：

```python
all_docs = [
    "Python就业班价格为 12980 元。",
    "AI大模型课程价格为 16800 元。",
    "课程有效期为报名后 2 年。",
    "学员可在有效期内反复观看录播课程。"
]
```

---

### 子问题 3

```python
sub_q = "售后服务怎么联系？"
```

检索结果：

```python
docs = [
    Document(page_content="售后服务电话为 400-xxx-xxxx。"),
    Document(page_content="学员也可以联系班主任处理售后问题。")
]
```

最终：

```python
all_docs = [
    "Python就业班价格为 12980 元。",
    "AI大模型课程价格为 16800 元。",
    "课程有效期为报名后 2 年。",
    "学员可在有效期内反复观看录播课程。",
    "售后服务电话为 400-xxx-xxxx。",
    "学员也可以联系班主任处理售后问题。"
]
```

---

## 6. 去重逻辑

源码：

```python
unique_docs_dict = {doc.page_content: doc for doc in all_docs}
unique_docs = list(unique_docs_dict.values())
```

作用：

```text
根据文档内容去重。
```

假设 `all_docs` 里面有重复内容：

```python
all_docs = [
    Document(page_content="课程有效期为报名后 2 年。"),
    Document(page_content="课程有效期为报名后 2 年。"),
    Document(page_content="售后服务电话为 400-xxx-xxxx。")
]
```

执行：

```python
{doc.page_content: doc for doc in all_docs}
```

会变成：

```python
{
    "课程有效期为报名后 2 年。": Document(...),
    "售后服务电话为 400-xxx-xxxx。": Document(...)
}
```

字典的 key 不能重复，所以相同内容只保留一份。

最终：

```python
unique_docs = [
    Document(page_content="课程有效期为报名后 2 年。"),
    Document(page_content="售后服务电话为 400-xxx-xxxx。")
]
```

---

## 7. 这个函数最终返回什么

返回的是：

```text
多个子问题检索出来的文档，去重后的结果。
```

也就是：

```python
return unique_docs
```

---

# 六、`_retrieve_with_backtracking()`：回溯问题检索

## 1. 源码核心

```python
def _retrieve_with_backtracking(self, query):
    logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
    backtrack_prompt_template = RAGPrompts.backtracking_prompt()
    try:
        simplified_query = self.llm(backtrack_prompt_template.format(query=query)).strip()
        logger.info(f"生成的回溯问题: '{simplified_query}'")
        return self.vector_store.hybrid_search_with_rerank(
            simplified_query, k=conf.RETRIEVAL_K
        )
    except Exception as e:
        logger.error(f"回溯问题策略执行失败: {e}")
        return []
```

---

## 2. 回溯问题是什么

回溯问题可以理解为：

```text
把复杂、口语化、带场景的问题，改写成更标准、更容易检索的问题。
```

---

## 3. 举例

用户原问题：

```text
我报完名之后发现时间不合适，这种情况钱还能不能退？
```

这个问题很口语化。

回溯后可能变成：

```text
报名后因个人原因无法学习，是否可以申请退费？
```

这个问题更标准，更容易检索到知识库中的“退费政策”。

---

## 4. 和 HyDE 的区别

| 策略 | 做了什么 |
|---|---|
| 回溯问题检索 | 把问题改写成更标准的问题 |
| HyDE 检索 | 先生成一个假设答案，再用假设答案检索 |

简单记：

```text
回溯问题：问题 → 标准问题 → 检索
HyDE：问题 → 假设答案 → 检索
```

---

## 5. 代入数据演示

用户问题：

```python
query = "我报完名之后发现时间不合适，这种情况钱还能不能退？"
```

### 第一步：生成回溯问题

```python
simplified_query = self.llm(backtrack_prompt_template.format(query=query)).strip()
```

得到：

```python
simplified_query = "报名后因个人原因无法学习，是否可以申请退费？"
```

### 第二步：用标准问题检索

```python
docs = self.vector_store.hybrid_search_with_rerank(
    simplified_query,
    k=conf.RETRIEVAL_K
)
```

假设返回：

```python
docs = [
    Document(page_content="学员报名后如因个人原因无法继续学习，可提交退费申请。"),
    Document(page_content="课程开始后退费金额需要根据已学习进度进行核算。"),
    Document(page_content="退费申请需要经过教务和财务审核。")
]
```

最终返回：

```python
return docs
```

---

# 七、`retrieve_and_merge()`：统一检索入口

## 1. 源码核心

```python
def retrieve_and_merge(self, query, source_filter=None, strategy=None):
    if not strategy:
        strategy = self.strategy_selector.select_strategy(query)

    ranked_sub_chunks = []
    if strategy == "回溯问题检索":
        ranked_sub_chunks = self._retrieve_with_backtracking(query)
    elif strategy == "子查询检索":
        ranked_sub_chunks = self._retrieve_with_subqueries(query)
    elif strategy == "假设问题检索":
        ranked_sub_chunks = self._retrieve_with_hyde(query)
    else:
        logger.info(f"使用直接检索策略 (查询: '{query}')")
        ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
            query, k=conf.RETRIEVAL_K, source_filter=source_filter
        )

    logger.info(f"策略 '{strategy}' 检索到 {len(ranked_sub_chunks)} 个候选文档 (可能已是父文档)")
    final_context_docs = ranked_sub_chunks[:conf.CANDIDATE_M]
    logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
    return final_context_docs
```

---

## 2. 函数作用

`retrieve_and_merge()` 的作用是：

```text
根据策略，选择对应的检索方法，然后返回最终要放入 prompt 的文档。
```

它是一个统一入口。

外部代码不需要关心到底是 HyDE、子查询、回溯问题，还是直接检索。

只需要调用：

```python
context_docs = self.retrieve_and_merge(query)
```

---

## 3. 参数解释

| 参数 | 作用 |
|---|---|
| `query` | 用户问题 |
| `source_filter` | 数据来源过滤，比如只查某个学科、某个文件 |
| `strategy` | 指定检索策略，如果不指定，就自动选择 |

---

## 4. 自动选择策略

```python
if not strategy:
    strategy = self.strategy_selector.select_strategy(query)
```

意思是：

```text
如果没有传入 strategy，就让策略选择器自动判断。
```

比如：

```python
query = "课程价格是多少，有效期多久，售后怎么联系？"
```

策略选择器可能返回：

```python
strategy = "子查询检索"
```

---

## 5. 根据策略调用不同方法

```python
if strategy == "回溯问题检索":
    ranked_sub_chunks = self._retrieve_with_backtracking(query)

elif strategy == "子查询检索":
    ranked_sub_chunks = self._retrieve_with_subqueries(query)

elif strategy == "假设问题检索":
    ranked_sub_chunks = self._retrieve_with_hyde(query)

else:
    ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(...)
```

这段代码可以翻译成：

```text
如果策略是“回溯问题检索”，就调用 _retrieve_with_backtracking()
如果策略是“子查询检索”，就调用 _retrieve_with_subqueries()
如果策略是“假设问题检索”，就调用 _retrieve_with_hyde()
否则，默认直接检索。
```

---

## 6. 代入数据演示

假设：

```python
query = "课程价格是多少，有效期多久，售后服务怎么联系？"
strategy = "子查询检索"
```

进入：

```python
elif strategy == "子查询检索":
    ranked_sub_chunks = self._retrieve_with_subqueries(query)
```

返回：

```python
ranked_sub_chunks = [
    Document(page_content="AI大模型课程价格为 16800 元。"),
    Document(page_content="课程报名后可享受阶段性优惠。"),
    Document(page_content="课程有效期为报名后 2 年。"),
    Document(page_content="课程有效期内可以反复观看录播视频。"),
    Document(page_content="售后服务电话为 400-xxx-xxxx。"),
    Document(page_content="学员可以联系班主任处理售后问题。")
]
```

---

## 7. 截取最终上下文文档

```python
final_context_docs = ranked_sub_chunks[:conf.CANDIDATE_M]
```

假设：

```python
conf.CANDIDATE_M = 3
```

那么只取前 3 个：

```python
final_context_docs = [
    Document(page_content="AI大模型课程价格为 16800 元。"),
    Document(page_content="课程报名后可享受阶段性优惠。"),
    Document(page_content="课程有效期为报名后 2 年。")
]
```

---

## 8. `RETRIEVAL_K` 和 `CANDIDATE_M` 的区别

这两个参数学生很容易混。

### `RETRIEVAL_K`

```python
k=conf.RETRIEVAL_K
```

表示：

```text
检索阶段先找多少个候选文档。
```

比如：

```python
conf.RETRIEVAL_K = 10
```

意思是先从知识库里找 10 个相关文档。

---

### `CANDIDATE_M`

```python
final_context_docs = ranked_sub_chunks[:conf.CANDIDATE_M]
```

表示：

```text
最终放进 prompt 的文档数量。
```

比如：

```python
conf.CANDIDATE_M = 3
```

意思是虽然前面找了 10 个，但最终只把前 3 个交给大模型。

---

### 记忆方式

```text
RETRIEVAL_K：先捞多少个
CANDIDATE_M：最后用多少个
```

---

# 八、`generate_answer()`：完整问答入口

RAG：检索 + 增强 + 生成

① 检索M个文档

② query + 检索M个文档（上下文）喂给LLM

③ LLM根据提交数据，生成最终结果

## 1. 源码核心

```python
def generate_answer(self, query, source_filter=None):
    start_time = time.time()
    logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")

    query_category = self.query_classifier.predict_category(query)
    logger.info(f"查询分类结果：{query_category} (查询: '{query}')")

    if query_category == "通用知识":
        logger.info("查询为通用知识，直接调用 LLM")
        prompt_input = self.rag_prompt.format(
            context="", question=query, history="", phone=config.CUSTOMER_SERVICE_PHONE
        )
        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f"直接调用 LLM 失败: {e}")
            answer = f"抱歉，处理您的通用知识问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
        processing_time = time.time() - start_time
        logger.info(
            f"通用知识查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
        )
        return answer

    logger.info("查询为专业咨询，执行 RAG 流程")
    strategy = self.strategy_selector.select_strategy(query)

    context_docs = self.retrieve_and_merge(
        query, source_filter=source_filter, strategy=strategy
    )

    if context_docs:
        context = "\n\n".join([doc.page_content for doc in context_docs])
        logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
    else:
        context = ""
        logger.info("未检索到相关文档，上下文为空")

    prompt_input = self.rag_prompt.format(
        context=context, question=query, phone=conf.CUSTOMER_SERVICE_PHONE
    )

    try:
        answer = self.llm(prompt_input)
    except Exception as e:
        logger.error(f"调用 LLM 生成最终答案失败: {e}")
        answer = f"抱歉，处理您的专业咨询问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

    processing_time = time.time() - start_time
    logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
    return answer
```

---

## 2. 函数作用

`generate_answer()` 是最核心的入口函数。

用户真正问问题时，调用的就是它。

例如：

```python
answer = rag_system.generate_answer("课程价格是多少？")
```

它负责完整执行：

```text
问题分类 → 判断是否需要 RAG → 选择策略 → 检索文档 → 拼接上下文 → 生成答案
```

## 3. rag_system代码测试

```powershell
if __name__ == '__main__':
	from rag_qa.core.vector_store import VectorStore
    from openai import OpenAI
    query = "AI学科主要学哪些内容？"

    vector_store = VectorStore()

    client = OpenAI(
        api_key=Config().DASHSCOPE_API_KEY,
        base_url=Config().DASHSCOPE_BASE_URL
    )

    def call_llm(prompt):
        response = client.chat.completions.create(
            model="qwen3.6-plus",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    rag_system = RAGSystem(vector_store, call_llm)

    answer = rag_system.generate_answer(query, source_filter="ai")

    print(answer)
```

注：由于向量数据库中还没有数据，所以可以提前把文档写入到Milvus，在vector_store.py代码中添加如下内容：

```python
if __name__ == "__main__":
    from rag_qa.core.document_processor import process_documents

    vector_store = VectorStore()

    # 用绝对路径，避免找错目录
    directory_path = f"{rag_qa_path}/data/ai_data"

    # 1. 处理文档
    documents = process_documents(
        directory_path,
        Config().PARENT_CHUNK_SIZE,
        Config().CHILD_CHUNK_SIZE,
        Config().CHUNK_OVERLAP,
    )

    print("文档块数量：", len(documents))

    # 2. 给所有 AI 文档打上 source=ai 标签
    for doc in documents:
        doc.metadata["source"] = "ai"
        doc.metadata["timestamp"] = "2026-06-23"

    # 3. 写入 Milvus
    vector_store.add_documents(documents)

    print("AI 文档入库完成")
    
    # --------------------------------------------------------------

    # 4. 测试检索
    results = vector_store.hybrid_search_with_rerank(
        query="人工智能主要学什么？",
        k=5,
        source_filter="ai"
    )

    print("检索结果数量：", len(results))

    for doc in results:
        print("metadata:", doc.metadata)
        print("content:", doc.page_content[:300])
        print("-" * 50)

    # 测试代码
    query = "课程新增了什么技术"
    results = vector_store.hybrid_search_with_rerank(query, source_filter='ai')
    print(f'results-->{results}')
    print(f'results-->{len(results)}')
```



# 九、完整数据代入演示

下面用一个完整问题演示整个流程。

## 1. 输入数据

```python
query = "课程价格是多少，有效期多久，售后服务怎么联系？"
source_filter = "AI大模型课程"
```

假设配置如下：

```python
conf.RETRIEVAL_K = 5
conf.CANDIDATE_M = 3
conf.CUSTOMER_SERVICE_PHONE = "400-123-4567"
```

---

## 2. 第一步：记录开始时间

源码：

```python
start_time = time.time()
```

作用：

```text
记录系统开始处理问题的时间，用来计算总耗时。
```

假设：

```python
start_time = 1710000000.00
```

---

## 3. 第二步：记录日志

源码：

```python
logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")
```

日志类似：

```text
开始处理查询: '课程价格是多少，有效期多久，售后服务怎么联系？', 学科过滤: AI大模型课程
```

---

## 4. 第三步：判断问题类型

源码：

```python
query_category = self.query_classifier.predict_category(query)
```

假设分类器返回：

```python
query_category = "专业咨询"
```

意思是：

```text
这个问题需要查知识库，不能直接让大模型凭空回答。
```

---

## 5. 第四步：如果是通用知识，直接回答

代码中有这个判断：

```python
if query_category == "通用知识":
```

如果用户问：

```text
太阳为什么会发光？
```

可能分类为：

```python
query_category = "通用知识"
```

那么系统直接调用：

```python
answer = self.llm(prompt_input)
return answer
```

不会进入检索。

---

## 6. 第五步：当前问题是专业咨询，进入 RAG

因为当前问题分类结果是：

```python
query_category = "专业咨询"
```

所以执行：

```python
logger.info("查询为专业咨询，执行 RAG 流程")
```

---

## 7. 第六步：选择检索策略

源码：

```python
strategy = self.strategy_selector.select_strategy(query)
```

当前问题是：

```text
课程价格是多少，有效期多久，售后服务怎么联系？
```

它包含多个问题，所以策略选择器可能返回：

```python
strategy = "子查询检索"
```

---

## 8. 第七步：调用 `retrieve_and_merge()`

源码：

```python
context_docs = self.retrieve_and_merge(
    query, source_filter=source_filter, strategy=strategy
)
```

也就是：

```python
context_docs = self.retrieve_and_merge(
    "课程价格是多少，有效期多久，售后服务怎么联系？",
    source_filter="AI大模型课程",
    strategy="子查询检索"
)
```

---

## 9. 第八步：进入 `retrieve_and_merge()`

此时：

```python
query = "课程价格是多少，有效期多久，售后服务怎么联系？"
source_filter = "AI大模型课程"
strategy = "子查询检索"
```

代码判断：

```python
elif strategy == "子查询检索":
    ranked_sub_chunks = self._retrieve_with_subqueries(query)
```

所以进入：

```python
_retrieve_with_subqueries()
```

---

## 10. 第九步：生成子查询

大模型把原问题拆成：

```python
subqueries = [
    "AI大模型课程价格是多少？",
    "AI大模型课程有效期多久？",
    "AI大模型课程售后服务怎么联系？"
]
```

---

## 11. 第十步：每个子查询分别检索

### 子查询 1

```python
sub_q = "AI大模型课程价格是多少？"
```

检索结果：

```python
[
    Document(page_content="AI大模型课程价格为 16800 元。"),
    Document(page_content="课程报名后可享受阶段性优惠。")
]
```

### 子查询 2

```python
sub_q = "AI大模型课程有效期多久？"
```

检索结果：

```python
[
    Document(page_content="AI大模型课程有效期为报名后 2 年。"),
    Document(page_content="课程有效期内可以反复观看录播视频。")
]
```

### 子查询 3

```python
sub_q = "AI大模型课程售后服务怎么联系？"
```

检索结果：

```python
[
    Document(page_content="售后服务可以联系班主任，也可以拨打客服电话 400-123-4567。"),
    Document(page_content="学习过程中遇到问题，可以在班级群中联系助教。")
]
```

---

## 12. 第十一步：合并所有检索结果

```python
all_docs = [
    Document(page_content="AI大模型课程价格为 16800 元。"),
    Document(page_content="课程报名后可享受阶段性优惠。"),
    Document(page_content="AI大模型课程有效期为报名后 2 年。"),
    Document(page_content="课程有效期内可以反复观看录播视频。"),
    Document(page_content="售后服务可以联系班主任，也可以拨打客服电话 400-123-4567。"),
    Document(page_content="学习过程中遇到问题，可以在班级群中联系助教。")
]
```

---

## 13. 第十二步：去重

源码：

```python
unique_docs_dict = {doc.page_content: doc for doc in all_docs}
unique_docs = list(unique_docs_dict.values())
```

如果没有重复，结果还是 6 条。

```python
unique_docs = [
    Document(page_content="AI大模型课程价格为 16800 元。"),
    Document(page_content="课程报名后可享受阶段性优惠。"),
    Document(page_content="AI大模型课程有效期为报名后 2 年。"),
    Document(page_content="课程有效期内可以反复观看录播视频。"),
    Document(page_content="售后服务可以联系班主任，也可以拨打客服电话 400-123-4567。"),
    Document(page_content="学习过程中遇到问题，可以在班级群中联系助教。")
]
```

返回给 `retrieve_and_merge()`。

---

## 14. 第十三步：截取最终文档

回到：

```python
final_context_docs = ranked_sub_chunks[:conf.CANDIDATE_M]
```

假设：

```python
conf.CANDIDATE_M = 3
```

那么：

```python
final_context_docs = [
    Document(page_content="AI大模型课程价格为 16800 元。"),
    Document(page_content="课程报名后可享受阶段性优惠。"),
    Document(page_content="AI大模型课程有效期为报名后 2 年。")
]
```

注意：这里有一个潜在问题。

用户问了三个方面：

```text
价格、有效期、售后
```

但是 `CANDIDATE_M = 3` 后，只保留了前 3 条，可能把“售后服务”相关文档截掉了。

所以子查询检索时，`CANDIDATE_M` 不宜太小。

---

## 15. 第十四步：返回最终上下文文档

```python
return final_context_docs
```

回到：

```python
context_docs = self.retrieve_and_merge(...)
```

此时：

```python
context_docs = [
    Document(page_content="AI大模型课程价格为 16800 元。"),
    Document(page_content="课程报名后可享受阶段性优惠。"),
    Document(page_content="AI大模型课程有效期为报名后 2 年。")
]
```

---

## 16. 第十五步：拼接上下文

源码：

```python
context = "\n\n".join([doc.page_content for doc in context_docs])
```

把多个文档块拼成一个字符串。

原来是：

```python
[
    Document(page_content="AI大模型课程价格为 16800 元。"),
    Document(page_content="课程报名后可享受阶段性优惠。"),
    Document(page_content="AI大模型课程有效期为报名后 2 年。")
]
```

拼接后变成：

```text
AI大模型课程价格为 16800 元。

课程报名后可享受阶段性优惠。

AI大模型课程有效期为报名后 2 年。
```

也就是：

```python
context = """
AI大模型课程价格为 16800 元。

课程报名后可享受阶段性优惠。

AI大模型课程有效期为报名后 2 年。
"""
```

---

## 17. 第十六步：构造最终 Prompt

源码：

```python
prompt_input = self.rag_prompt.format(
    context=context,
    question=query,
    phone=conf.CUSTOMER_SERVICE_PHONE
)
```

假设 RAG Prompt 模板是：

```text
请根据上下文回答用户问题。

上下文：
{context}

用户问题：
{question}

如果上下文无法回答，请提示联系人工客服：{phone}
```

填入数据后变成：

```text
请根据上下文回答用户问题。

上下文：
AI大模型课程价格为 16800 元。

课程报名后可享受阶段性优惠。

AI大模型课程有效期为报名后 2 年。

用户问题：
课程价格是多少，有效期多久，售后服务怎么联系？

如果上下文无法回答，请提示联系人工客服：400-123-4567
```

---

## 18. 第十七步：调用大模型生成答案

源码：

```python
answer = self.llm(prompt_input)
```

大模型可能返回：

```text
AI大模型课程价格为 16800 元，课程报名后可能有阶段性优惠。课程有效期为报名后 2 年。关于售后服务联系方式，当前检索到的上下文中没有明确说明，建议联系人工客服：400-123-4567。
```

---

## 19. 第十八步：记录耗时并返回答案

源码：

```python
processing_time = time.time() - start_time
return answer
```

最终返回给用户：

```text
AI大模型课程价格为 16800 元，课程报名后可能有阶段性优惠。课程有效期为报名后 2 年。关于售后服务联系方式，当前检索到的上下文中没有明确说明，建议联系人工客服：400-123-4567。
```

---

# 十、完整调用链总结

对于这个问题：

```python
query = "课程价格是多少，有效期多久，售后服务怎么联系？"
```

完整执行链是：

```text
generate_answer()
   ↓
query_classifier.predict_category()
   ↓
结果：专业咨询
   ↓
strategy_selector.select_strategy()
   ↓
结果：子查询检索
   ↓
retrieve_and_merge()
   ↓
_retrieve_with_subqueries()
   ↓
生成 3 个子查询
   ↓
每个子查询分别 hybrid_search_with_rerank()
   ↓
合并检索结果
   ↓
去重
   ↓
截取前 CANDIDATE_M 个文档
   ↓
拼接 context
   ↓
构造 rag_prompt
   ↓
llm(prompt_input)
   ↓
返回最终答案
```

---

# 十一、四种检索方式对比

| 检索方式 | 输入是什么 | 中间做了什么 | 适合场景 |
|---|---|---|---|
| 直接检索 | 用户原问题 | 直接去知识库查 | 问题清楚、关键词明确 |
| 回溯问题检索 | 用户原问题 | 改写成标准问题 | 问题口语化、表达不标准 |
| 子查询检索 | 用户原问题 | 拆成多个小问题 | 一个问题包含多个需求 |
| HyDE 检索 | 用户原问题 | 先生成假设答案 | 用户问题短、模糊、缺关键词 |

---

# 十二、源码中的几个明显问题

学生看源码时不能只看流程，也要知道哪里可能有问题。

---

## 问题 1：`conf` 没有定义

代码里多次出现：

```python
conf.RETRIEVAL_K
conf.CANDIDATE_M
conf.CUSTOMER_SERVICE_PHONE
```

但是开头导入的是：

```python
from base.config import config
```

也就是说，当前代码里有 `config`，但没有 `conf`。

所以运行时可能会报错：

```text
NameError: name 'conf' is not defined
```

### 修正方式

把所有：

```python
conf
```

改成：

```python
config
```

例如：

```python
k=config.RETRIEVAL_K
```

```python
final_context_docs = ranked_sub_chunks[:config.CANDIDATE_M]
```

```python
phone=config.CUSTOMER_SERVICE_PHONE
```

---

## 问题 2：通用知识分支缩进错误

原代码这里有问题：

```python
if query_category == "通用知识":
    logger.info("查询为通用知识，直接调用 LLM")
     prompt_input = self.rag_prompt.format(
        context="", question=query, history="", phone=config.CUSTOMER_SERVICE_PHONE
    )
```

这一行：

```python
 prompt_input = self.rag_prompt.format(
```

前面多了一个空格，缩进不对。

可能会报错：

```text
IndentationError: unexpected indent
```

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

---

## 问题 3：Prompt 参数不一致

通用知识分支：

```python
self.rag_prompt.format(
    context="", question=query, history="", phone=config.CUSTOMER_SERVICE_PHONE
)
```

专业咨询分支：

```python
self.rag_prompt.format(
    context=context, question=query, phone=conf.CUSTOMER_SERVICE_PHONE
)
```

一个传了：

```python
history
```

另一个没传。

如果你的 prompt 模板里面有 `{history}`，专业咨询分支可能会报错：

```text
KeyError: 'history'
```

建议统一成：

```python
prompt_input = self.rag_prompt.format(
    context=context,
    question=query,
    history="",
    phone=config.CUSTOMER_SERVICE_PHONE
)
```

---

## 问题 4：`source_filter` 在特殊策略中没有传递

直接检索传了：

```python
self.vector_store.hybrid_search_with_rerank(
    query,
    k=config.RETRIEVAL_K,
    source_filter=source_filter
)
```

但是 HyDE、子查询、回溯问题没有传：

```python
self.vector_store.hybrid_search_with_rerank(
    hypo_answer,
    k=config.RETRIEVAL_K
)
```

这会导致：

```text
如果用户只想查某个学科或某个文件，特殊检索策略可能不受 source_filter 限制。
```

建议改成：

```python
def _retrieve_with_hyde(self, query, source_filter=None):
    ...
    return self.vector_store.hybrid_search_with_rerank(
        hypo_answer,
        k=config.RETRIEVAL_K,
        source_filter=source_filter
    )
```

子查询和回溯问题也应该同步传入 `source_filter`。

---

# 十三、给学生的最小记忆版

这份代码可以按 3 层理解。

## 第一层：入口

```python
generate_answer()
```

作用：

```text
用户问问题，系统生成答案。
```

---

## 第二层：检索调度

```python
retrieve_and_merge()
```

作用：

```text
根据策略决定怎么检索文档。
```

---

## 第三层：具体检索策略

```python
_retrieve_with_hyde()
_retrieve_with_subqueries()
_retrieve_with_backtracking()
```

作用：

```text
分别使用不同方式改造用户问题，再去知识库检索。
```

---

# 十四、一句话总结

这份代码的核心逻辑是：

```text
generate_answer() 是总入口；
它先判断问题是不是专业问题；
如果是专业问题，就选择检索策略；
retrieve_and_merge() 根据策略调用不同检索方法；
检索到文档后拼成 context；
最后把 context + question 放进 prompt，让大模型生成答案。
```

最重要的执行链是：

```text
generate_answer()
→ retrieve_and_merge()
→ 具体检索策略
→ hybrid_search_with_rerank()
→ 拼接 context
→ llm()
→ answer
```
