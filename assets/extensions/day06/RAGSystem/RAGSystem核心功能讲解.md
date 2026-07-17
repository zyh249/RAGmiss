# RAGSystem 核心功能讲解

# 一、这个类整体做什么？

源码核心类：

```python
class RAGSystem:
```

它的作用是：

```text
封装 RAG 系统的完整问答流程。
```

也就是：

```text
分类问题
选择检索策略
检索相关文档
构造上下文
调用大模型生成答案
返回最终结果
```

可以把它理解成：

```text
RAGSystem = 问答系统总控制器
```

---

# 二、整体执行流程

用户输入一个问题，比如：

```text
JAVA课程费用多少？
```

系统流程是：

```text
1. generate_answer 接收用户问题
2. QueryClassifier 判断问题类型
3. 如果是通用知识，直接调用大模型
4. 如果是专业咨询，进入 RAG 检索流程
5. StrategySelector 选择检索策略
6. retrieve_and_merge 执行检索
7. 组装 context 上下文
8. 使用 PromptTemplate 构造提示词
9. 调用大模型生成答案
10. 返回最终回答
```

---

# 三、初始化函数 `__init__`

源码：

```python
def __init__(self, vector_store, llm):
    self.vector_store = vector_store
    self.llm = llm
    self.rag_prompt = RAGPrompts.rag_prompt()
    self.query_classifier = QueryClassifier(
        model_path=f'{config.MODELS_DIR}/bert_query_classifier'
    )
    self.strategy_selector = StrategySelector()
```

这个函数在创建 `RAGSystem` 对象时自动执行。

例如：

```python
rag_system = RAGSystem(vector_store, llm)
```

---

## 1. `vector_store`

```python
self.vector_store = vector_store
```

作用：

```text
保存向量数据库对象。
```

它负责检索文档。

后面会调用：

```python
self.vector_store.hybrid_search_with_rerank(...)
```

也就是说，`vector_store` 里面通常封装了：

```text
向量检索
BM25 / 稀疏检索
混合检索
rerank 重排序
```

---

## 2. `llm`

```python
self.llm = llm
```

作用：

```text
保存大语言模型调用函数。
```

后面代码里会这样调用：

```python
answer = self.llm(prompt_input)
```

也就是说，`llm` 可能是：

```text
Qwen
DeepSeek
OpenAI 兼容接口
本地大模型
```

在这个类里，不关心具体是哪种模型，只要它能接收 prompt 并返回回答即可。

---

## 3. `rag_prompt`

```python
self.rag_prompt = RAGPrompts.rag_prompt()
```

作用：

```text
获取 RAG 问答提示词模板。
```

它后面会用来构造最终 prompt：

```python
prompt_input = self.rag_prompt.format(
    context=context,
    question=query,
    phone=conf.CUSTOMER_SERVICE_PHONE
)
```

也就是把：

```text
检索到的上下文
用户问题
客服电话
```

填入提示词模板。

---

## 4. `query_classifier`

```python
self.query_classifier = QueryClassifier(
    model_path=f'{config.MODELS_DIR}/bert_query_classifier'
)
```

作用：

```text
初始化 BERT 查询分类器。
```

它负责判断问题属于：

```text
通用知识
专业咨询
```

例如：

```python
query_category = self.query_classifier.predict_category(query)
```

可能返回：

```text
通用知识
```

或者：

```text
专业咨询
```

---

## 5. `strategy_selector`

```python
self.strategy_selector = StrategySelector()
```

作用：

```text
初始化检索策略选择器。
```

它负责判断当前问题应该使用哪种检索策略。

可能的策略有：

```text
直接检索
回溯问题检索
子查询检索
假设问题检索
```

---

# 四、HyDE 检索：`_retrieve_with_hyde`

源码：

```python
def _retrieve_with_hyde(self, query):
```

这个方法是私有方法，前面有 `_`，表示主要在类内部使用。

---

## 1. HyDE 是什么？

HyDE 可以理解为：

```text
先让大模型根据问题生成一个“假设答案”，再用这个假设答案去检索知识库。
```

为什么要这样？

因为有些用户问题太短、太模糊，直接检索效果不好。

例如用户问：

```text
学费怎么算？
```

这个问题比较短。

HyDE 会先让大模型生成一个假设答案：

```text
课程学费通常根据课程类型、学习周期、授课方式等因素决定。
```

然后用这个假设答案去做向量检索。

这样检索文本更丰富，可能更容易召回相关文档。

---

## 2. 函数流程

源码核心：

```python
hyde_prompt_template = RAGPrompts.hyde_prompt()
hypo_answer = self.llm(hyde_prompt_template.format(query=query)).strip()
return self.vector_store.hybrid_search_with_rerank(hypo_answer, k=conf.RETRIEVAL_K)
```

流程：

```text
用户问题
    ↓
HyDE Prompt
    ↓
大模型生成假设答案
    ↓
用假设答案去向量库检索
    ↓
返回相关文档
```

---

## 3. 示例

输入问题：

```text
JAVA课程费用多少？
```

大模型可能生成假设答案：

```text
JAVA课程费用通常包括基础班、就业班、进阶班等不同收费标准。
```

然后系统不是用原问题检索，而是用这个假设答案检索：

```python
self.vector_store.hybrid_search_with_rerank(
    hypo_answer,
    k=conf.RETRIEVAL_K
)
```

---

## 4. 返回值

返回：

```text
检索到的文档列表
```

大概长这样：

```python
[
    Document(page_content="JAVA课程费用为...", metadata={...}),
    Document(page_content="课程报名优惠政策...", metadata={...})
]
```

如果出错：

```python
return []
```

---

# 五、子查询检索：`_retrieve_with_subqueries`

源码：

```python
def _retrieve_with_subqueries(self, query):
```

---

## 1. 子查询检索是什么？

子查询检索就是：

```text
把一个复杂问题拆成多个小问题，再分别检索。
```

适合用户问题包含多个信息点的情况。

例如用户问：

```text
AI课程大纲是什么？费用多少？有哪些老师？
```

这个问题其实包含 3 个问题：

```text
1. AI课程大纲是什么？
2. AI课程费用多少？
3. AI课程有哪些老师？
```

子查询策略会先让大模型拆问题，再分别检索。

---

## 2. 函数流程

核心代码：

```python
subquery_prompt_template = RAGPrompts.subquery_prompt()
subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
```

这几行作用：

```text
调用大模型，把原问题拆成多个子问题。
```

例如：

```python
subqueries = [
    "AI课程大纲是什么？",
    "AI课程费用多少？",
    "AI课程有哪些老师？"
]
```

然后遍历每个子查询：

```python
for sub_q in subqueries:
    docs = self.vector_store.hybrid_search_with_rerank(
        sub_q, k=conf.RETRIEVAL_K
    )
    all_docs.extend(docs)
```

作用：

```text
每个子问题单独检索一次。
```

最后去重：

```python
unique_docs_dict = {doc.page_content: doc for doc in all_docs}
unique_docs = list(unique_docs_dict.values())
```

作用：

```text
把重复文档去掉。
```

这里是基于：

```python
doc.page_content
```

去重。

如果两个文档内容一样，只保留一个。

---

## 3. 示例流程

输入：

```text
AI课程大纲是什么？费用多少？有哪些老师？
```

大模型生成子查询：

```text
AI课程大纲是什么？
AI课程费用多少？
AI课程有哪些老师？
```

分别检索：

```text
子查询1 → 检索课程大纲文档
子查询2 → 检索费用文档
子查询3 → 检索老师介绍文档
```

合并：

```text
所有文档合并
    ↓
去重
    ↓
返回唯一文档列表
```

---

## 4. 返回值

返回：

```python
unique_docs
```

也就是去重后的文档列表。

如果失败：

```python
return []
```

---

# 六、回溯问题检索：`_retrieve_with_backtracking`

源码：

```python
def _retrieve_with_backtracking(self, query):
```

---

## 1. 回溯问题是什么？

回溯问题可以理解为：

```text
把具体问题改写成更本质、更宽泛的问题。
```

适合问题太具体、表达不标准的情况。

例如用户问：

```text
我想知道现在报名JAVA有没有优惠？
```

可以回溯成：

```text
JAVA课程报名政策是什么？
```

这样更容易匹配知识库中的标准文档。

---

## 2. 函数流程

核心代码：

```python
backtrack_prompt_template = RAGPrompts.backtracking_prompt()
simplified_query = self.llm(backtrack_prompt_template.format(query=query)).strip()
return self.vector_store.hybrid_search_with_rerank(
    simplified_query, k=conf.RETRIEVAL_K
)
```

流程：

```text
用户问题
    ↓
回溯 Prompt
    ↓
大模型生成更标准的问题
    ↓
使用标准问题检索
    ↓
返回文档
```

---

## 3. 示例

原问题：

```text
我现在想学AI，但是不知道适不适合，课程怎么安排？
```

回溯问题可能是：

```text
AI课程适合人群和课程安排是什么？
```

然后用回溯后的问题检索知识库。

---

# 七、检索总入口：`retrieve_and_merge`

源码：

```python
def retrieve_and_merge(self, query, source_filter=None, strategy=None):
```

这个函数是检索阶段的总入口。

---

## 1. 函数作用

```text
根据检索策略，调用不同的检索方法，然后返回最终上下文文档。
```

它负责选择：

```text
直接检索
回溯问题检索
子查询检索
HyDE检索
```

---

## 2. 参数说明

```python
query
```

用户问题。

```python
source_filter
```

数据来源过滤条件。

例如只检索某个学科、某个文件、某个来源。

```python
strategy
```

指定检索策略。

如果没有传入策略，则自动选择：

```python
strategy = self.strategy_selector.select_strategy(query)
```

---

## 3. 策略选择

源码：

```python
if not strategy:
    strategy = self.strategy_selector.select_strategy(query)
```

意思是：

```text
如果外部没有指定策略，就让策略选择器自动判断。
```

---

## 4. 根据策略执行不同检索

### 情况1：回溯问题检索

```python
if strategy == "回溯问题检索":
    ranked_sub_chunks = self._retrieve_with_backtracking(query)
```

### 情况2：子查询检索

```python
elif strategy == "子查询检索":
    ranked_sub_chunks = self._retrieve_with_subqueries(query)
```

### 情况3：假设问题检索

```python
elif strategy == "假设问题检索":
    ranked_sub_chunks = self._retrieve_with_hyde(query)
```

### 情况4：直接检索

```python
else:
    ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
        query,
        k=conf.RETRIEVAL_K,
        source_filter=source_filter
    )
```

直接检索就是：

```text
不改写问题，不拆分问题，直接拿用户问题去知识库检索。
```

---

## 5. 限制最终文档数量

源码：

```python
final_context_docs = ranked_sub_chunks[:conf.CANDIDATE_M]
```

作用：

```text
只取前 CANDIDATE_M 个文档作为最终上下文。
```

比如：

```python
conf.CANDIDATE_M = 3
```

检索到 10 个文档：

```text
Doc1, Doc2, Doc3, Doc4, ...
```

最终只取：

```text
Doc1, Doc2, Doc3
```

---

## 6. 返回值

```python
return final_context_docs
```

返回最终用于构造上下文的文档列表。

---

# 八、问答总入口：`generate_answer`

源码：

```python
def generate_answer(self, query, source_filter=None):
```

这是整个 RAG 系统最重要的方法。

---

## 1. 函数作用

```text
接收用户问题，生成最终答案。
```

它是完整问答流程入口。

---

## 2. 记录开始时间

```python
start_time = time.time()
```

作用：

```text
用于计算本次问答耗时。
```

后面：

```python
processing_time = time.time() - start_time
```

---

## 3. 查询分类

```python
query_category = self.query_classifier.predict_category(query)
```

作用：

```text
判断用户问题是“通用知识”还是“专业咨询”。
```

例如：

```python
query = "5*9等于多少？"
```

可能返回：

```text
通用知识
```

例如：

```python
query = "JAVA课程费用多少？"
```

可能返回：

```text
专业咨询
```

---

# 九、分支一：通用知识直接回答

源码：

```python
if query_category == "通用知识":
    logger.info("查询为通用知识，直接调用 LLM")
    prompt_input = self.rag_prompt.format(
        context="", question=query, history="", phone=config.CUSTOMER_SERVICE_PHONE
    )
    answer = self.llm(prompt_input)
    return answer
```

---

## 1. 什么问题算通用知识？

例如：

```text
5*9等于多少？
太阳为什么会发光？
Python是什么？
```

这种问题不需要查企业知识库。

---

## 2. 为什么直接调用 LLM？

因为大模型本身就可以回答。

流程：

```text
用户问题
    ↓
分类结果：通用知识
    ↓
不检索知识库
    ↓
context 为空
    ↓
直接调用大模型回答
```

---

## 3. 示例

输入：

```text
5*9等于多少？
```

分类结果：

```text
通用知识
```

构造 prompt：

```text
上下文：空
问题：5*9等于多少？
```

大模型回答：

```text
5*9=45
```

---

# 十、分支二：专业咨询走 RAG

如果不是通用知识，就执行：

```python
logger.info("查询为专业咨询，执行 RAG 流程")
```

---

## 1. 什么问题算专业咨询？

例如：

```text
JAVA课程费用多少？
AI学科课程大纲是什么？
AI培训有哪些老师？
报名后可以退款吗？
```

这类问题需要查企业知识库。

---

## 2. 选择检索策略

源码：

```python
strategy = self.strategy_selector.select_strategy(query)
```

作用：

```text
根据用户问题选择检索方式。
```

可能选择：

```text
直接检索
子查询检索
回溯问题检索
假设问题检索
```

---

## 3. 检索相关文档

源码：

```python
context_docs = self.retrieve_and_merge(
    query, source_filter=source_filter, strategy=strategy
)
```

作用：

```text
根据策略检索知识库，得到相关文档。
```

返回结果类似：

```python
[
    Document(page_content="JAVA课程费用为9800元...", metadata={"source": "java.txt"}),
    Document(page_content="JAVA课程支持分期付款...", metadata={"source": "policy.txt"})
]
```

---

## 4. 构造上下文

源码：

```python
if context_docs:
    context = "\n\n".join([doc.page_content for doc in context_docs])
else:
    context = ""
```

作用：

```text
把检索到的多个文档内容拼接成上下文。
```

例如：

```python
context_docs = [
    Document(page_content="JAVA课程费用为9800元。"),
    Document(page_content="JAVA课程支持分期付款。")
]
```

拼接后：

```text
JAVA课程费用为9800元。

JAVA课程支持分期付款。
```

这个结果会作为 `{context}` 填入 RAG Prompt。

---

## 5. 构造 Prompt

源码：

```python
prompt_input = self.rag_prompt.format(
    context=context,
    question=query,
    phone=conf.CUSTOMER_SERVICE_PHONE
)
```

作用：

```text
把上下文、问题、客服电话填入提示词模板。
```

最终 prompt 类似：

```text
你是一个智能客服助手，请基于以下上下文回答问题。

上下文：
JAVA课程费用为9800元。
JAVA课程支持分期付款。

问题：
JAVA课程费用多少？

如果无法回答，请联系人工客服：400-xxx-xxxx
```

---

## 6. 调用大模型生成答案

源码：

```python
answer = self.llm(prompt_input)
```

作用：

```text
让大模型根据上下文生成最终回答。
```

输出可能是：

```text
JAVA课程费用为9800元，同时支持分期付款。具体优惠可以联系人工客服进一步确认。
```

---

## 7. 返回最终答案

```python
return answer
```

---

# 十一、完整示例一：通用知识问题

用户输入：

```text
5*9等于多少？
```

执行流程：

```text
generate_answer("5*9等于多少？")
    ↓
query_classifier.predict_category(...)
    ↓
分类结果：通用知识
    ↓
不执行知识库检索
    ↓
context = ""
    ↓
调用 LLM
    ↓
返回答案
```

最终答案：

```text
5*9=45
```

---

# 十二、完整示例二：专业咨询问题

用户输入：

```text
JAVA课程费用多少？
```

执行流程：

```text
generate_answer("JAVA课程费用多少？")
    ↓
query_classifier.predict_category(...)
    ↓
分类结果：专业咨询
    ↓
strategy_selector.select_strategy(...)
    ↓
选择检索策略：直接检索
    ↓
retrieve_and_merge(...)
    ↓
vector_store.hybrid_search_with_rerank(...)
    ↓
检索到相关文档
    ↓
拼接 context
    ↓
构造 prompt
    ↓
调用 LLM
    ↓
返回答案
```

检索到文档：

```text
文档1：JAVA课程费用为9800元。
文档2：JAVA课程支持分期付款。
```

构造上下文：

```text
JAVA课程费用为9800元。

JAVA课程支持分期付款。
```

最终答案：

```text
JAVA课程费用为9800元，支持分期付款。具体优惠请以当前报名政策为准。
```

---

# 十三、四种检索策略总结

## 1. 直接检索

```text
直接用用户原问题检索。
```

适合：

```text
问题清楚、单一、表达标准
```

例如：

```text
JAVA课程费用多少？
```

---

## 2. 回溯问题检索

```text
先把问题改写成更本质的问题，再检索。
```

适合：

```text
问题表达口语化、不标准
```

例如：

```text
我想报AI，但是不知道这个课程到底适不适合我
```

可以改写为：

```text
AI课程适合人群是什么？
```

---

## 3. 子查询检索

```text
把复杂问题拆成多个小问题，分别检索。
```

适合：

```text
一个问题包含多个信息点
```

例如：

```text
AI课程大纲是什么？费用多少？有哪些老师？
```

拆成：

```text
AI课程大纲是什么？
AI课程费用多少？
AI课程有哪些老师？
```

---

## 4. HyDE 检索

```text
先生成假设答案，再用假设答案检索。
```

适合：

```text
问题短、模糊、关键词不足
```

例如：

```text
费用呢？
```

可以先生成假设答案：

```text
课程费用根据课程类型、学习周期、优惠政策等因素确定。
```

再用这个假设答案检索。

---

# 十四、源码中需要注意的问题

这段源码里有几个地方需要注意，否则运行可能报错。

---

## 1. `conf` 没有导入

代码顶部是：

```python
from base.config import config
```

但后面使用了：

```python
conf.RETRIEVAL_K
conf.CANDIDATE_M
conf.CUSTOMER_SERVICE_PHONE
```

这里的 `conf` 没有定义。

应该统一成：

```python
config.RETRIEVAL_K
config.CANDIDATE_M
config.CUSTOMER_SERVICE_PHONE
```

或者在顶部改成：

```python
from base.config import config as conf
```

---

## 2. 通用知识分支缩进错误

源码中：

```python
if query_category == "通用知识":
    logger.info("查询为通用知识，直接调用 LLM")
     prompt_input = self.rag_prompt.format(
```

`prompt_input` 前面多了一个空格，缩进不对。

应该改成：

```python
if query_category == "通用知识":
    logger.info("查询为通用知识，直接调用 LLM")
    prompt_input = self.rag_prompt.format(
        context="", question=query, history="", phone=config.CUSTOMER_SERVICE_PHONE
    )
```

---

## 3. Prompt 参数可能不统一

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

一个传了 `history`，一个没有。

要看你的 `RAGPrompts.rag_prompt()` 模板里是否包含：

```text
{history}
```

如果模板需要 `history`，专业咨询分支也要传。

如果模板不需要 `history`，通用知识分支可以去掉。

---

# 十五、最终总结

这段 `RAGSystem` 代码可以总结成一句话：

```text
它是 RAG 问答系统的总调度器，负责根据用户问题类型决定是直接调用大模型，还是先检索知识库再生成答案。
```

核心流程：

```text
用户问题
    ↓
BERT 查询分类器
    ↓
通用知识？
    ↓ 是
直接调用 LLM
    ↓
返回答案

用户问题
    ↓
BERT 查询分类器
    ↓
专业咨询？
    ↓ 是
选择检索策略
    ↓
知识库检索
    ↓
构造上下文
    ↓
调用 LLM
    ↓
返回答案
```

最核心的两个判断：

```text
1. QueryClassifier 判断问题类型
2. StrategySelector 判断检索策略
```

最核心的两个动作：

```text
1. vector_store.hybrid_search_with_rerank 检索文档
2. llm(prompt_input) 生成答案
```
