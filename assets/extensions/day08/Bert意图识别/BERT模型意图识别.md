# BERT 模型意图识别代码讲解

## 一、这个类到底干什么？

你的类叫：

```python
class QueryClassifier:
```

它的作用是：

```text
输入：用户问题
输出：问题类别
```

比如：

```python
classifier.predict_category("JAVA课程费用多少？")
```

输出：

```text
专业咨询
```

再比如：

```python
classifier.predict_category("5*9等于多少？")
```

输出：

```text
通用知识
```

所以它在 RAG 系统里面的作用是：

```text
先判断问题类型，再决定走哪条流程。
```

流程是：

```text
用户问题
   ↓
BERT 意图识别
   ↓
通用知识  → 直接让大模型回答
专业咨询  → 进入知识库检索 / RAG
```

---

## 二、先看代码整体结构

你的代码可以分成 3 大块：

```text
第一块：初始化
加载 tokenizer、加载模型、设置设备、设置标签映射

第二块：训练
读取数据 → 文本编码 → 标签转数字 → Dataset 封装 → Trainer 训练 → 保存模型

第三块：预测
用户输入一句话 → tokenizer 编码 → model 推理 → logits → argmax → 返回类别
```

对应代码方法是：

```python
__init__()              # 初始化
load_model()           # 加载模型
save_model()           # 保存模型

preprocess_data()      # 数据预处理
create_dataset()       # 创建训练数据集
train_model()          # 训练模型
compute_metrics()      # 计算准确率
evaluate_model()       # 评估模型

predict_category()     # 预测问题类别
```

---

## 三、先看最核心的 3 种数据

这段代码里数据一直在变形，你只要看懂这 3 种数据，就能理解大部分代码。

### 1. 原始问题

```python
"JAVA课程费用多少？"
```

这是普通字符串。

### 2. tokenizer 编码后的数据

BERT 不能直接吃中文字符串，所以要转成数字。

```python
{
    "input_ids": tensor([...]),
    "attention_mask": tensor([...]),
    "token_type_ids": tensor([...])
}
```

比如：

```text
JAVA课程费用多少？
```

会变成：

```text
[CLS] java 课 程 费 用 多 少 ？ [SEP] [PAD] [PAD]
```

再变成数字：

```python
{
    "input_ids": tensor([101, 8179, 4923, 3621, 6589, 5508, 1914, 2208, 8043, 102, 0, 0]),
    "attention_mask": tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    "token_type_ids": tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
}
```

简单记：

```text
input_ids：每个 token 的数字编号
attention_mask：1 表示真实内容，0 表示 PAD 补齐
token_type_ids：句子编号，单句分类任务基本全是 0
```

### 3. 标签数字

你定义了：

```python
self.label_map = {"通用知识": 0, "专业咨询": 1}
```

所以：

```text
通用知识 → 0
专业咨询 → 1
```

比如原始标签：

```python
["专业咨询", "通用知识", "专业咨询"]
```

会变成：

```python
[1, 0, 1]
```

---

## 四、初始化部分讲解

代码：

```python
def __init__(self, model_path='models/bert_query_classifier'):
```

这个方法在创建对象时自动执行。

比如：

```python
classifier = QueryClassifier(model_path="bert_query_classifier")
```

它会自动做这些事：

### 1. 设置本地 BERT 路径

```python
self.pre_trained_model_path = f'{rag_qa_path}/models/bert-base-chinese'
```

意思是：你的基础 BERT 模型放在本地这个目录。

也就是：

```text
models/bert-base-chinese
```

这个模型是原始中文 BERT，不是你训练好的意图分类模型。

### 2. 设置训练后模型保存位置

```python
self.model_path = model_path
```

这个路径表示你训练好的分类模型保存在哪里。

比如：

```python
model_path="bert_query_classifier"
```

表示后面会从这个目录加载已经训练好的分类模型。

### 3. 加载 tokenizer

```python
self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_model_path)
```

作用：

```text
加载 BERT 分词器，把中文文本转成 input_ids、attention_mask、token_type_ids。
```

例如：

```text
JAVA课程费用多少？
```

会被 tokenizer 变成：

```text
[CLS] java 课 程 费 用 多 少 ？ [SEP]
```

再变成数字。

### 4. 设置设备

```python
self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
```

作用：

```text
判断模型放在哪里运行。
```

可能结果是：

```text
cuda：NVIDIA GPU
mps：苹果芯片 GPU
cpu：普通 CPU
```

你是 Windows 环境，通常就是：

```text
cuda 或 cpu
```

### 5. 设置标签映射

```python
self.label_map = {"通用知识": 0, "专业咨询": 1}
```

作用：

```text
把中文类别转成数字类别。
```

因为模型只能训练数字标签，不能直接训练中文标签。

### 6. 加载模型

```python
self.load_model()
```

这句会调用下面的 `load_model()` 方法。

---

## 五、load_model：加载模型

代码：

```python
def load_model(self):
    if os.path.exists(self.model_path):
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        logger.info(f"加载模型: {self.model_path}")
    else:
        self.model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
        self.model.to(self.device)
        logger.info("初始化新 BERT 模型")
```

这个方法的逻辑是：

```text
如果已经训练过模型，就加载训练好的模型。
如果没有训练过，就初始化一个新的 BERT 二分类模型。
```

### 情况 1：模型路径存在

比如：

```text
bert_query_classifier/
```

目录存在，那么执行：

```python
self.model = BertForSequenceClassification.from_pretrained(self.model_path)
```

意思是：

```text
加载你已经训练好的意图识别模型。
```

### 情况 2：模型路径不存在

如果模型目录不存在，就执行：

```python
self.model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
```

意思是：

```text
加载原始 bert-base-chinese，然后加一个二分类头。
```

其中：

```python
num_labels=2
```

表示只有两个类别：

```text
0：通用知识
1：专业咨询
```

---

## 六、preprocess_data：数据预处理

代码：

```python
def preprocess_data(self, texts, labels):
    encodings = self.tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    return encodings, [self.label_map[label] for label in labels]
```

这个函数非常重要。

它的作用是：

```text
把文本转成 BERT 输入，把中文标签转成数字标签。
```

### 输入数据

假设：

```python
texts = [
    "JAVA课程费用多少？",
    "5*9等于多少？"
]

labels = [
    "专业咨询",
    "通用知识"
]
```

### 第一步：文本编码

执行：

```python
encodings = self.tokenizer(...)
```

得到：

```python
encodings = {
    "input_ids": tensor([
        [101, 8179, 4923, 3621, 6589, 5508, 1914, 2208, 8043, 102],
        [101, 126, 115, 130, 4638, 4917, 2208, 1914, 8043, 102]
    ]),

    "attention_mask": tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]),

    "token_type_ids": tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```

注意：

```text
第 0 行：第一条问题
第 1 行：第二条问题
```

### 第二步：标签转数字

执行：

```python
[self.label_map[label] for label in labels]
```

原来是：

```python
["专业咨询", "通用知识"]
```

变成：

```python
[1, 0]
```

所以 `preprocess_data()` 最终返回：

```python
encodings, [1, 0]
```

---

## 七、create_dataset：创建 Dataset

代码：

```python
def create_dataset(self, encodings, labels):
```

这个函数的作用是：

```text
把 tokenizer 编码结果和标签组合起来，变成 Trainer 能训练的数据集。
```

里面定义了一个内部类：

```python
class Dataset(torch.utils.data.Dataset):
```

这个 Dataset 必须提供两个方法：

```python
__getitem__()
__len__()
```

### 1. `__len__`

代码：

```python
def __len__(self):
    return len(self.labels)
```

作用：

```text
告诉 Trainer：这个数据集一共有多少条数据。
```

假设有 5000 条数据：

```python
len(dataset)
```

返回：

```python
5000
```

### 2. `__getitem__`

代码：

```python
def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.encodings.items()}
    item["labels"] = torch.tensor(self.labels[idx])
    return item
```

作用：

```text
根据下标 idx，取出第 idx 条训练样本。
```

比如：

```python
train_dataset[0]
```

返回：

```python
{
    "input_ids": tensor([101, 8179, 4923, 3621, 6589, 5508, 1914, 2208, 8043, 102]),
    "attention_mask": tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    "token_type_ids": tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "labels": tensor(1)
}
```

这就是模型训练时真正吃进去的一条数据。

---

## 八、train_model：训练模型

代码：

```python
def train_model(self, data_file="training_dataset_hybrid_5000.json"):
```

这个方法负责完整训练流程。

### 1. 检查数据文件是否存在

```python
if not os.path.exists(data_file):
    logger.error(f"数据集文件 {data_file} 不存在")
    raise FileNotFoundError(...)
```

作用：

```text
如果训练数据文件不存在，就报错。
```

### 2. 读取 JSON Lines 数据

```python
with open(data_file, "r", encoding="utf-8") as f:
    data = [json.loads(value) for value in f.readlines()]
```

你的数据文件大概长这样：

```json
{"query": "JAVA课程费用多少？", "label": "专业咨询"}
{"query": "5*9等于多少？", "label": "通用知识"}
{"query": "AI培训有哪些老师？", "label": "专业咨询"}
```

读取后变成 Python 列表：

```python
data = [
    {"query": "JAVA课程费用多少？", "label": "专业咨询"},
    {"query": "5*9等于多少？", "label": "通用知识"},
    {"query": "AI培训有哪些老师？", "label": "专业咨询"}
]
```

### 3. 拆出问题和标签

```python
texts = [item["query"] for item in data]
labels = [item["label"] for item in data]
```

得到：

```python
texts = [
    "JAVA课程费用多少？",
    "5*9等于多少？",
    "AI培训有哪些老师？"
]

labels = [
    "专业咨询",
    "通用知识",
    "专业咨询"
]
```

### 4. 划分训练集和验证集

```python
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
```

如果有 5000 条数据：

```text
train_texts：4000 条
val_texts：1000 条

train_labels：4000 个
val_labels：1000 个
```

意思是：

```text
80% 用来训练
20% 用来验证
```

### 5. 预处理训练集和验证集

```python
train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
```

这一步做了两件事：

```text
文本 → input_ids / attention_mask / token_type_ids
标签 → 0 / 1
```

比如：

```python
train_texts = ["JAVA课程费用多少？", "5*9等于多少？"]
train_labels = ["专业咨询", "通用知识"]
```

变成：

```python
train_encodings = {
    "input_ids": tensor(...),
    "attention_mask": tensor(...),
    "token_type_ids": tensor(...)
}

train_labels = [1, 0]
```

### 6. 创建 Dataset

```python
train_dataset = self.create_dataset(train_encodings, train_labels)
val_dataset = self.create_dataset(val_encodings, val_labels)
```

这一步把数据变成 Trainer 能读取的形式。

一条样本长这样：

```python
{
    "input_ids": tensor([...]),
    "attention_mask": tensor([...]),
    "token_type_ids": tensor([...]),
    "labels": tensor(1)
}
```

### 7. 设置训练参数

```python
training_args = TrainingArguments(
    output_dir="./bert_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./bert_logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    fp16=False,
)
```

这些参数不用死记，可以分组理解。

#### 训练轮数

```python
num_train_epochs=3
```

意思是：

```text
整个训练集重复训练 3 遍。
```

#### batch size

```python
per_device_train_batch_size=8
```

意思是：

```text
每次给模型喂 8 条数据。
```

#### 日志

```python
logging_dir="./bert_logs"
logging_steps=10
```

意思是：

```text
每 10 步记录一次训练日志。
```

#### 评估和保存

```python
evaluation_strategy="epoch"
save_strategy="epoch"
```

意思是：

```text
每训练完一轮，就评估一次，并保存一次模型。
```

#### 最佳模型

```python
load_best_model_at_end=True
metric_for_best_model="eval_loss"
```

意思是：

```text
训练结束后，不一定用最后一轮模型，而是加载验证损失最小的那个模型。
```

### 8. 创建 Trainer

```python
trainer = Trainer(
    model=self.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=self.compute_metrics
)
```

这里可以这样理解：

```text
model：要训练的 BERT 分类模型
args：训练参数
train_dataset：训练集
eval_dataset：验证集
compute_metrics：评估指标函数
```

### 9. 开始训练

```python
trainer.train()
```

这一句内部会自动做很多事：

```text
取 batch 数据
模型前向传播
计算 loss
反向传播
更新参数
验证集评估
保存 checkpoint
```

所以你不需要手写：

```python
loss.backward()
optimizer.step()
```

### 10. 保存模型

```python
self.save_model()
```

会调用：

```python
def save_model(self):
    self.model.save_pretrained(self.model_path)
    self.tokenizer.save_pretrained(self.model_path)
```

保存两个东西：

```text
模型参数
tokenizer 配置
```

保存后，下次可以直接加载：

```python
BertForSequenceClassification.from_pretrained(self.model_path)
```

---

## 九、compute_metrics：计算准确率

代码：

```python
def compute_metrics(self, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}
```

这个函数用于验证阶段。

假设模型输出：

```python
logits = [
    [0.2, 2.8],
    [3.1, 0.4],
    [0.6, 2.2],
    [2.5, 0.7]
]
```

含义是：

```text
第 1 条：通用知识分数 0.2，专业咨询分数 2.8
第 2 条：通用知识分数 3.1，专业咨询分数 0.4
第 3 条：通用知识分数 0.6，专业咨询分数 2.2
第 4 条：通用知识分数 2.5，专业咨询分数 0.7
```

执行：

```python
predictions = np.argmax(logits, axis=-1)
```

得到：

```python
predictions = [1, 0, 1, 0]
```

如果真实标签是：

```python
labels = [1, 0, 0, 0]
```

对比：

```text
预测：[1, 0, 1, 0]
真实：[1, 0, 0, 0]
```

正确 3 条，所以准确率：

```python
accuracy = 0.75
```

返回：

```python
{"accuracy": 0.75}
```

---

## 十、evaluate_model：评估模型

代码：

```python
def evaluate_model(self, texts, labels):
```

这个方法主要是训练结束后，用验证集看模型效果。

它会做：

```text
1. 对验证文本进行 tokenizer 编码
2. 封装成 Dataset
3. 用 Trainer 预测
4. 得到预测类别
5. 输出分类报告
6. 输出混淆矩阵
```

核心代码：

```python
predictions = trainer.predict(dataset)
pred_labels = np.argmax(predictions.predictions, axis=-1)
```

意思是：

```text
模型先输出 logits，再用 argmax 转成类别编号。
```

然后：

```python
classification_report(
    true_labels,
    pred_labels,
    target_names=["通用知识", "专业咨询"]
)
```

会输出类似：

```text
              precision    recall  f1-score   support

        通用知识       0.95      0.93      0.94       500
        专业咨询       0.94      0.96      0.95       500

    accuracy                           0.95      1000
```

大概含义：

```text
precision：预测为这个类别的样本里，有多少是真的
recall：真实属于这个类别的样本里，有多少被找出来
f1-score：precision 和 recall 的综合指标
support：这个类别有多少条样本
```

---

## 十一、predict_category：上线预测

这是最重要的方法。

代码：

```python
def predict_category(self, query):
```

它的输入是一句话：

```python
query = "JAVA课程费用多少？"
```

返回是：

```text
通用知识
```

或者：

```text
专业咨询
```

### 第一步：判断模型是否加载

```python
if self.model is None:
    logger.error("模型未训练或加载")
    return "通用知识"
```

意思是：

```text
如果模型没有加载成功，默认返回通用知识。
```

### 第二步：tokenizer 编码

```python
encoding = self.tokenizer(
    query,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)
```

比如输入：

```text
JAVA课程费用多少？
```

得到：

```python
encoding = {
    "input_ids": tensor([[101, 8179, 4923, 3621, 6589, 5508, 1914, 2208, 8043, 102]]),
    "attention_mask": tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    "token_type_ids": tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
}
```

注意这里是二维：

```text
[1, 序列长度]
```

这个 `1` 表示：

```text
现在只预测 1 条问题。
```

### 第三步：放到设备上

```python
encoding = {k: v.to(self.device) for k, v in encoding.items()}
```

意思是：

```text
把 input_ids、attention_mask、token_type_ids 都移动到 CPU/GPU 上。
```

### 第四步：模型推理

```python
with torch.no_grad():
    outputs = self.model(**encoding)
```

`torch.no_grad()` 的意思是：

```text
预测阶段不需要计算梯度，可以节省内存和计算。
```

模型输出中最重要的是：

```python
outputs.logits
```

比如：

```python
outputs.logits = tensor([[0.8, 3.2]])
```

含义是：

```text
通用知识分数：0.8
专业咨询分数：3.2
```

### 第五步：取最大分数

```python
prediction = torch.argmax(outputs.logits, dim=1).item()
```

因为：

```text
3.2 > 0.8
```

所以：

```python
prediction = 1
```

### 第六步：返回中文类别

```python
return "专业咨询" if prediction == 1 else "通用知识"
```

所以最终返回：

```text
专业咨询
```

---

## 十二、完整预测流程示例

输入：

```python
query = "JAVA课程费用多少？"
```

经过 tokenizer：

```text
[CLS] java 课 程 费 用 多 少 ？ [SEP]
```

变成数字：

```python
{
    "input_ids": tensor([[101, 8179, 4923, 3621, 6589, 5508, 1914, 2208, 8043, 102]]),
    "attention_mask": tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    "token_type_ids": tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
}
```

模型输出：

```python
logits = tensor([[0.8, 3.2]])
```

取最大值：

```python
prediction = 1
```

转中文：

```python
"专业咨询"
```

完整链路：

```text
JAVA课程费用多少？
    ↓
tokenizer
    ↓
input_ids / attention_mask / token_type_ids
    ↓
BERT 分类模型
    ↓
logits = [0.8, 3.2]
    ↓
argmax = 1
    ↓
专业咨询
```

---

## 十三、`__main__` 部分在干什么？

代码：

```python
if __name__ == "__main__":
    classifier = QueryClassifier(model_path="bert_query_classifier")

    test_queries = [
        "AI学科的课程大纲是什么",
        "JAVA课程费用多少？",
        "5*9等于多少？",
        "AI培训有哪些老师？"
    ]

    for query in test_queries:
        category = classifier.predict_category(query)
        print(f"查询: {query} -> 分类: {category}")
```

这部分是测试代码。

它做了三件事：

```text
1. 创建 QueryClassifier 对象
2. 准备几条测试问题
3. 循环预测每条问题的类别
```

输出可能是：

```text
查询: AI学科的课程大纲是什么 -> 分类: 专业咨询
查询: JAVA课程费用多少？ -> 分类: 专业咨询
查询: 5*9等于多少？ -> 分类: 通用知识
查询: AI培训有哪些老师？ -> 分类: 专业咨询
```

注意，这里训练代码被注释掉了：

```python
# classifier.train_model(data_file='../classify_data/model_generic_5000.json')
```

所以当前运行时，不会训练模型，只会：

```text
加载已有模型 → 预测测试问题
```

如果你想重新训练，才需要取消这行注释。
