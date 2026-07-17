# BERT 意图识别 QueryClassifier 完整教程

# 学习目标

学完本教程后，你需要掌握：

1. BERT 意图识别模块在 RAG 系统中的作用。
2. `QueryClassifier` 类中每个函数的作用。
3. 每个函数输入什么、处理什么、返回什么。
4. `tokenizer`、`encodings`、`Dataset`、`Trainer`、`logits` 的含义。
5. 如何用示例数据走完整训练和预测流程。

---

# 一、整体作用

这段代码实现的是一个 **BERT 问题意图分类器**。

它做的事情是：

```text
输入：用户问题
输出：问题类别
```

当前只分两类：

```python
self.label_map = {
    "通用知识": 0,
    "专业咨询": 1
}
```

也就是：

```text
通用知识 → 0
专业咨询 → 1
```

例如：

```text
5*9等于多少？
→ 通用知识

JAVA课程费用多少？
→ 专业咨询
```

在 RAG 系统中的作用：

```text
用户问题
    ↓
BERT 意图识别
    ↓
通用知识 → 直接让大模型回答
专业咨询 → 进入知识库检索 / RAG 检索
```

---

# 二、代码整体结构

这段代码主要分为 13 个部分：

```text
1. 导入依赖包
2. 处理项目路径
3. 定义 QueryClassifier 类
4. __init__ 初始化
5. load_model 加载模型
6. save_model 保存模型
7. preprocess_data 数据预处理
8. create_dataset 创建 Dataset
9. train_model 训练模型
10. compute_metrics 计算指标
11. evaluate_model 评估模型
12. predict_category 预测类别
13. __main__ 主程序测试
```

核心流程：

```text
训练阶段：
训练数据 → tokenizer 编码 → Dataset → Trainer 训练 → 保存模型

预测阶段：
用户问题 → tokenizer 编码 → BERT 分类模型 → logits → argmax → 类别结果
```

---

# 三、导入依赖包

源码：

```python
import json
import os
import torch
import sys
from base.logger import logger
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
```

## 1. `json`

作用：读取 JSON 格式训练数据。

训练数据示例：

```json
{"query": "JAVA课程费用多少？", "label": "专业咨询"}
```

代码中使用：

```python
json.loads(value)
```

把一行 JSON 字符串转换成 Python 字典。

---

## 2. `os`

作用：处理路径、判断文件或目录是否存在。

例如：

```python
os.path.exists(self.model_path)
```

判断模型目录是否存在。

---

## 3. `torch`

作用：PyTorch 深度学习框架。

本代码中主要用于：

```python
torch.device(...)
torch.no_grad()
torch.argmax(...)
torch.tensor(...)
torch.utils.data.Dataset
```

---

## 4. `sys`

作用：修改 Python 模块搜索路径。

例如：

```python
sys.path.insert(0, project_root)
```

让项目根目录可以被 Python 正确识别。

---

## 5. `logger`

作用：记录日志。

例如：

```python
logger.info("开始训练 BERT 模型...")
logger.error("数据集文件不存在")
```

---

## 6. `numpy`

作用：数组计算。

本代码中主要用于：

```python
np.argmax(logits, axis=-1)
```

从模型输出分数中找出最大值的位置。

---

## 7. Transformers 相关对象

| 对象 | 作用 |
|---|---|
| `BertTokenizer` | 把中文文本转换成 BERT 能识别的数字 |
| `BertForSequenceClassification` | BERT 分类模型 |
| `Trainer` | HuggingFace 官方训练器 |
| `TrainingArguments` | 训练参数配置类 |

---

## 8. scikit-learn 相关函数

| 函数 | 作用 |
|---|---|
| `train_test_split` | 划分训练集和验证集 |
| `classification_report` | 输出分类报告 |
| `confusion_matrix` | 输出混淆矩阵 |

---

# 四、项目路径处理

源码：

```python
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(os.path.abspath(current_dir)))
project_root = os.path.abspath(os.path.dirname(os.path.abspath(rag_qa_path)))
sys.path.insert(0, project_root)
```

## 作用

这几行代码用于计算项目目录路径，并把项目根目录加入 Python 搜索路径。

假设当前文件路径是：

```text
D:\uv\edurag\rag_qa\core\query_classifier.py
```

那么：

```python
current_dir
```

表示：

```text
D:\uv\edurag\rag_qa\core
```

```python
rag_qa_path
```

表示：

```text
D:\uv\edurag\rag_qa
```

```python
project_root
```

表示：

```text
D:\uv\edurag
```

最后：

```python
sys.path.insert(0, project_root)
```

表示把项目根目录加入模块搜索路径，这样才能正常导入：

```python
from base.logger import logger
```

---

# 五、QueryClassifier 类整体作用

源码：

```python
class QueryClassifier:
```

这个类封装了 BERT 意图识别的完整流程。

它负责：

```text
模型加载
数据预处理
Dataset 构建
模型训练
模型评估
模型预测
```

可以理解为：

```text
QueryClassifier = BERT 问题分类器
```

---

# 六、函数一：`__init__` 初始化函数

源码：

```python
def __init__(self, model_path='models/bert_query_classifier'):
   # 加载bert
    self.pre_trained_model_path = f'{rag_qa_path}/models/bert-base-chinese'
    # 模型训练以后保存的位置
    self.model_path = model_path
    # 加载 BERT 分词器
    self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_model_path)
    # 初始化模型
    self.model = None
    # 确定设备（GPU 或 CPU）
    self.device =  torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu" )
    # 记录设备信息
    logger.info(f"使用设备: {self.device}")
    # 定义标签映射
    self.label_map = {"通用知识": 0, "专业咨询": 1}
    # 加载模型
    self.load_model()
```

## 1. 函数作用

`__init__` 是初始化函数。

当执行：

```python
classifier = QueryClassifier(model_path="bert_query_classifier")
```

系统会自动调用 `__init__()`。

它的作用是：

```text
创建分类器对象时，准备好 tokenizer、模型路径、设备、标签映射和模型对象。
```

---

## 2. 参数说明

```python
model_path='models/bert_query_classifier'
```

表示训练好的分类模型保存位置。

如果你写：

```python
classifier = QueryClassifier(model_path="bert_query_classifier")
```

那么：

```python
self.model_path = "bert_query_classifier"
```

---

## 3. 设置基础 BERT 模型路径

```python
self.pre_trained_model_path = f'{rag_qa_path}/models/bert-base-chinese'
```

作用：指定本地基础中文 BERT 模型目录。

这个目录存放的是原始 BERT 模型，不是训练好的意图识别模型。

---

## 4. 设置训练后模型路径

```python
self.model_path = model_path
```

作用：指定意图识别模型保存位置。

---

## 5. 加载 tokenizer

```python
self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_model_path)
```

作用：加载 BERT 分词器。

BERT 不能直接处理中文字符串，需要 tokenizer 把文本转换成数字。

示例：

```text
JAVA课程费用多少？
```

先变成 token：

```text
[CLS] java 课 程 费 用 多 少 ？ [SEP]
```

再变成数字：

```python
[101, 8179, 4923, 3621, 6589, 5508, 1914, 2208, 8043, 102]
```

---

## 6. 初始化模型变量

```python
self.model = None
```

作用：先定义一个空模型变量，后面由 `load_model()` 加载真正模型。

---

## 7. 设置运行设备

```python
self.device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.mps.is_available()
    else "cpu"
)
```

选择逻辑：

```text
有 NVIDIA GPU → cuda
有苹果芯片 GPU → mps
否则 → cpu
```

Windows 环境通常是：

```text
cuda 或 cpu
```

---

## 8. 设置标签映射

```python
self.label_map = {"通用知识": 0, "专业咨询": 1}
```

作用：把中文标签转换成数字标签。

```text
通用知识 → 0
专业咨询 → 1
```

模型只能学习数字标签，不能直接学习中文标签。

---

## 9. 自动加载模型

```python
self.load_model()
```

作用：初始化时自动调用 `load_model()`。

如果已有训练好的模型，就加载已有模型。

如果没有训练好的模型，就初始化新模型。

---

## 10. 示例执行过程

执行：

```python
classifier = QueryClassifier(model_path="bert_query_classifier")
```

内部流程：

```text
1. 设置基础模型路径
2. 设置分类模型路径
3. 加载 tokenizer
4. 初始化 self.model = None
5. 判断使用 cuda / mps / cpu
6. 设置 label_map
7. 调用 load_model()
```

---

# 七、函数二：`load_model` 加载模型

源码：

```python
def load_model(self):
    # 检查模型路径是否存在
    if os.path.exists(self.model_path):
        # 加载预训练模型
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        # 将模型移到指定设备
        self.model.to(self.device)
        # 记录加载成功的日志
        logger.info(f"加载模型: {self.model_path}")
    else:
        # 初始化新模型
        self.model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
        # 将模型移到指定设备
        self.model.to(self.device)
        # 记录初始化模型的日志
        logger.info("初始化新 BERT 模型")
```

## 1. 函数作用

```text
加载已经训练好的模型，或者创建新的 BERT 二分类模型。
```

---

## 2. 情况一：模型路径存在

判断代码：

```python
if os.path.exists(self.model_path):
```

如果目录存在，例如：

```text
bert_query_classifier/
```

说明模型之前训练过。

执行：

```python
self.model = BertForSequenceClassification.from_pretrained(self.model_path)
```

作用：从本地模型目录加载训练好的分类模型。

然后：

```python
self.model.to(self.device)
```

作用：把模型移动到 CPU 或 GPU 上。

---

## 3. 情况二：模型路径不存在

如果模型路径不存在，执行：

```python
self.model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2
)
```

作用：加载基础中文 BERT，并加一个二分类头。

```python
num_labels=2
```

表示两个类别：

```text
0 → 通用知识
1 → 专业咨询
```

---

## 4. 返回结果

这个函数没有显式 `return`。

但是它会给：

```python
self.model
```

赋值。

执行后：

```python
self.model
```

就是可用的 BERT 分类模型。

---

# 八、函数三：`save_model` 保存模型

源码：

```python
def save_model(self):
    """保存模型"""
    self.model.save_pretrained(self.model_path)
    self.tokenizer.save_pretrained(self.model_path)
    logger.info(f"模型保存至: {self.model_path}")
```

## 1. 函数作用

```text
保存训练好的模型和 tokenizer。
```

---

## 2. 保存模型

```python
self.model.save_pretrained(self.model_path)
```

作用：保存模型参数和模型配置。

---

## 3. 保存 tokenizer

```python
self.tokenizer.save_pretrained(self.model_path)
```

作用：保存分词器词表和配置。

为什么 tokenizer 也要保存？

```text
训练和预测必须使用同一个 tokenizer。
如果 tokenizer 不一致，同一句话可能会被编码成不同数字，模型效果会出问题。
```

---

## 4. 保存后的目录示例

```text
bert_query_classifier/
    config.json
    model.safetensors
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
```

---

# 九、函数四：`preprocess_data` 数据预处理

源码：

```python
def preprocess_data(self, texts, labels):
    """预处理数据为 BERT 输入格式"""
    encodings = self.tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    return encodings, [self.label_map[label] for label in labels]
```

## 1. 函数作用

```text
把原始文本和中文标签转换成 BERT 可以训练的数据格式。
```

它做两件事：

```text
1. 文本 → tokenizer → 数字输入
2. 中文标签 → label_map → 0 / 1
```

---

## 2. 输入示例

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

---

## 3. tokenizer 编码

```python
encodings = self.tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)
```

参数解释：

| 参数 | 含义 |
|---|---|
| `texts` | 文本列表 |
| `truncation=True` | 文本超过最大长度时截断 |
| `padding=True` | 同一个 batch 内自动补齐长度 |
| `max_length=128` | 最大长度为 128 |
| `return_tensors="pt"` | 返回 PyTorch tensor |

---

## 4. 编码后结果示例

```python
encodings = {
    "input_ids": tensor([
        [101, 8179, 4923, 3621, 6589, 5508, 1914, 2208, 8043, 102],
        [101, 126, 115, 130, 4638, 754, 1914, 2208, 8043, 102]
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

---

## 5. `input_ids`

表示每个 token 的数字编号。

例如：

```text
[CLS] java 课 程 费 用 多 少 ？ [SEP]
```

可能对应：

```python
[101, 8179, 4923, 3621, 6589, 5508, 1914, 2208, 8043, 102]
```

---

## 6. `attention_mask`

表示哪些位置是真实内容，哪些位置是 padding 补齐内容。

规则：

```text
1 → 真实 token
0 → padding token
```

例如：

```text
[CLS] java 课 程 [SEP] [PAD] [PAD]
```

对应：

```python
attention_mask = [1, 1, 1, 1, 1, 0, 0]
```

---

## 7. `token_type_ids`

表示每个 token 属于第几个句子。

你的任务是单句分类，只输入一个问题，所以通常全是 0。

```python
token_type_ids = [0, 0, 0, 0, 0]
```

如果是两个句子，可能会有 0 和 1：

```text
[CLS] 问题 [SEP] 答案 [SEP]
  0    0     0    1    1
```

---

## 8. 标签转换

源码：

```python
[self.label_map[label] for label in labels]
```

输入：

```python
labels = ["专业咨询", "通用知识"]
```

根据：

```python
self.label_map = {
    "通用知识": 0,
    "专业咨询": 1
}
```

转换为：

```python
[1, 0]
```

---

## 9. 函数返回值

```python
return encodings, [self.label_map[label] for label in labels]
```

返回两个东西：

```text
1. encodings：文本编码结果
2. 数字标签列表
```

示例返回：

```python
(
    {
        "input_ids": tensor(...),
        "attention_mask": tensor(...),
        "token_type_ids": tensor(...)
    },
    [1, 0]
)
```

---

# 十、函数五：`create_dataset` 创建 Dataset

源码：

```python
def create_dataset(self, encodings, labels):
    """创建 PyTorch 数据集"""

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    return Dataset(encodings, labels)
```

## 1. 函数作用

```text
把 tokenizer 编码结果和数字标签封装成 PyTorch Dataset。
```

Trainer 不能直接读取普通的：

```python
encodings
labels
```

它需要一个标准 Dataset。

---

## 2. 输入示例

```python
encodings = {
    "input_ids": tensor([
        [101, 8179, 4923, 102],
        [101, 126, 115, 102]
    ]),
    "attention_mask": tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]),
    "token_type_ids": tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
}

labels = [1, 0]
```

---

## 3. 内部类 Dataset

```python
class Dataset(torch.utils.data.Dataset):
```

表示创建一个自定义数据集类，并继承 PyTorch 官方 Dataset。

Trainer 能识别这种格式。

---

## 4. `__init__`

```python
def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels
```

作用：保存所有编码结果和标签。

---

## 5. `__len__`

```python
def __len__(self):
    return len(self.labels)
```

作用：返回数据集中有多少条样本。

如果：

```python
labels = [1, 0]
```

则：

```python
len(dataset)
```

返回：

```python
2
```

---

## 6. `__getitem__`

```python
def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.encodings.items()}
    item["labels"] = torch.tensor(self.labels[idx])
    return item
```

作用：根据下标取出一条训练样本。

---

## 7. 代入 `idx=0`

执行：

```python
dataset[0]
```

等价于：

```python
__getitem__(0)
```

这行：

```python
item = {key: val[0] for key, val in self.encodings.items()}
```

等价于：

```python
item = {
    "input_ids": self.encodings["input_ids"][0],
    "attention_mask": self.encodings["attention_mask"][0],
    "token_type_ids": self.encodings["token_type_ids"][0]
}
```

得到：

```python
item = {
    "input_ids": tensor([101, 8179, 4923, 102]),
    "attention_mask": tensor([1, 1, 1, 1]),
    "token_type_ids": tensor([0, 0, 0, 0])
}
```

再执行：

```python
item["labels"] = torch.tensor(self.labels[0])
```

因为：

```python
self.labels[0] = 1
```

所以最终得到：

```python
{
    "input_ids": tensor([101, 8179, 4923, 102]),
    "attention_mask": tensor([1, 1, 1, 1]),
    "token_type_ids": tensor([0, 0, 0, 0]),
    "labels": tensor(1)
}
```

---

## 8. 代入 `idx=1`

执行：

```python
dataset[1]
```

返回：

```python
{
    "input_ids": tensor([101, 126, 115, 102]),
    "attention_mask": tensor([1, 1, 1, 1]),
    "token_type_ids": tensor([0, 0, 0, 0]),
    "labels": tensor(0)
}
```

---

## 9. 函数返回值

```python
return Dataset(encodings, labels)
```

返回 Dataset 对象。

后面可以直接交给 Trainer：

```python
train_dataset = self.create_dataset(train_encodings, train_labels)
```

---

# 十一、函数六：`train_model` 训练模型

源码：

```python
def train_model(self, data_file="training_dataset_hybrid_5000.json"):
```

## 1. 函数作用

完整训练 BERT 意图分类模型。

它包含：

```text
读取数据
拆出文本和标签
划分训练集和验证集
预处理数据
创建 Dataset
配置训练参数
创建 Trainer
训练模型
保存模型
评估模型
```

---

## 2. 检查数据文件

```python
if not os.path.exists(data_file):
    logger.error(f"数据集文件 {data_file} 不存在")
    raise FileNotFoundError(f"数据集文件 {data_file} 不存在")
```

作用：如果训练文件不存在，直接报错。

---

## 3. 读取训练数据

```python
with open(data_file, "r", encoding="utf-8") as f:
    data = [json.loads(value) for value in f.readlines()]
```

假设训练文件内容：

```json
{"query": "JAVA课程费用多少？", "label": "专业咨询"}
{"query": "5*9等于多少？", "label": "通用知识"}
{"query": "AI培训有哪些老师？", "label": "专业咨询"}
{"query": "太阳为什么会发光？", "label": "通用知识"}
```

读取后：

```python
data = [
    {"query": "JAVA课程费用多少？", "label": "专业咨询"},
    {"query": "5*9等于多少？", "label": "通用知识"},
    {"query": "AI培训有哪些老师？", "label": "专业咨询"},
    {"query": "太阳为什么会发光？", "label": "通用知识"}
]
```

---

## 4. 拆出问题和标签

```python
texts = [item["query"] for item in data]
labels = [item["label"] for item in data]
```

得到：

```python
texts = [
    "JAVA课程费用多少？",
    "5*9等于多少？",
    "AI培训有哪些老师？",
    "太阳为什么会发光？"
]

labels = [
    "专业咨询",
    "通用知识",
    "专业咨询",
    "通用知识"
]
```

---

## 5. 划分训练集和验证集

```python
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
```

作用：把数据分为训练集和验证集。

```text
test_size=0.2 表示 20% 作为验证集，80% 作为训练集。
random_state=42 表示固定随机结果。
```

如果有 5000 条数据：

```text
训练集：4000 条
验证集：1000 条
```

---

## 6. 预处理训练集和验证集

```python
train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
```

这一步完成：

```text
文本 → input_ids / attention_mask / token_type_ids
中文标签 → 0 / 1
```

例如：

```python
train_texts = [
    "JAVA课程费用多少？",
    "AI培训有哪些老师？"
]

train_labels = [
    "专业咨询",
    "专业咨询"
]
```

处理后：

```python
train_encodings = {
    "input_ids": tensor(...),
    "attention_mask": tensor(...),
    "token_type_ids": tensor(...)
}

train_labels = [1, 1]
```

---

## 7. 创建 Dataset

```python
train_dataset = self.create_dataset(train_encodings, train_labels)
val_dataset = self.create_dataset(val_encodings, val_labels)
```

得到：

```text
train_dataset：训练集 Dataset
val_dataset：验证集 Dataset
```

一条样本结构：

```python
{
    "input_ids": tensor([...]),
    "attention_mask": tensor([...]),
    "token_type_ids": tensor([...]),
    "labels": tensor(1)
}
```

---

## 8. 设置训练参数

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

参数解释：

| 参数 | 含义 |
|---|---|
| `output_dir` | 模型检查点保存目录 |
| `num_train_epochs=3` | 训练 3 轮 |
| `per_device_train_batch_size=8` | 每个设备每次训练 8 条数据 |
| `per_device_eval_batch_size=8` | 每个设备每次评估 8 条数据 |
| `warmup_steps=500` | 前 500 步学习率逐渐升高 |
| `weight_decay=0.01` | 权重衰减，防止过拟合 |
| `logging_dir` | 日志保存目录 |
| `logging_steps=10` | 每 10 步记录一次日志 |
| `evaluation_strategy="epoch"` | 每轮结束评估一次 |
| `save_strategy="epoch"` | 每轮结束保存一次 |
| `load_best_model_at_end=True` | 训练结束后加载最佳模型 |
| `save_total_limit=1` | 最多保存 1 个检查点 |
| `metric_for_best_model="eval_loss"` | 用验证损失判断最佳模型 |
| `fp16=False` | 不使用半精度训练 |

---

## 9. 创建 Trainer

```python
trainer = Trainer(
    model=self.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=self.compute_metrics
)
```

参数解释：

| 参数 | 作用 |
|---|---|
| `model` | 要训练的 BERT 分类模型 |
| `args` | 训练参数配置 |
| `train_dataset` | 训练数据 |
| `eval_dataset` | 验证数据 |
| `compute_metrics` | 验证指标计算函数 |

---

## 10. 开始训练

```python
logger.info("开始训练 BERT 模型...")
trainer.train()
```

`trainer.train()` 内部会自动完成：

```text
1. 从 train_dataset 中取 batch 数据
2. 把 input_ids、attention_mask、labels 送入模型
3. 模型输出预测结果
4. 根据 labels 计算 loss
5. 反向传播
6. 更新模型参数
7. 按 epoch 进行验证和保存
```

---

## 11. 保存模型

```python
self.save_model()
```

保存：

```text
模型参数
tokenizer 配置
```

---

## 12. 评估模型

```python
self.evaluate_model(val_texts, val_labels)
```

使用验证集评估模型表现。

注意：

```text
这里的 val_labels 已经在 preprocess_data 后变成数字标签。
```

---

## 13. 函数返回值

`train_model()` 没有显式返回值。

它的结果是：

```text
模型被训练
模型被保存
评估结果写入日志
```

---

# 十二、函数七：`compute_metrics` 计算指标

源码：

```python
def compute_metrics(self, eval_pred):
    """计算评估指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}
```

## 1. 函数作用

```text
在验证集上计算模型准确率。
```

---

## 2. 输入数据

`eval_pred` 是 Trainer 传进来的评估结果。

里面有：

```python
logits, labels = eval_pred
```

含义：

```text
logits：模型输出的类别分数
labels：真实标签
```

---

## 3. 示例数据

```python
logits = [
    [0.8, 3.2],
    [4.1, 0.6],
    [0.5, 2.7],
    [3.5, 0.2]
]

labels = [1, 0, 1, 0]
```

表格理解：

| 样本 | 通用知识分数 | 专业咨询分数 | 真实标签 |
|---|---:|---:|---:|
| 第1条 | 0.8 | 3.2 | 1 |
| 第2条 | 4.1 | 0.6 | 0 |
| 第3条 | 0.5 | 2.7 | 1 |
| 第4条 | 3.5 | 0.2 | 0 |

---

## 4. 获取预测类别

```python
predictions = np.argmax(logits, axis=-1)
```

表示每一行取最大值位置。

结果：

```python
predictions = [1, 0, 1, 0]
```

解释：

```text
[0.8, 3.2] → 最大位置 1 → 专业咨询
[4.1, 0.6] → 最大位置 0 → 通用知识
[0.5, 2.7] → 最大位置 1 → 专业咨询
[3.5, 0.2] → 最大位置 0 → 通用知识
```

---

## 5. 计算准确率

```python
accuracy = (predictions == labels).mean()
```

对比：

```python
predictions = [1, 0, 1, 0]
labels      = [1, 0, 1, 0]
```

全部正确，所以：

```python
accuracy = 1.0
```

如果：

```python
predictions = [1, 0, 1, 0]
labels      = [1, 1, 1, 0]
```

只有 3 个正确，则：

```python
accuracy = 0.75
```

---

## 6. 返回值

```python
return {"accuracy": accuracy}
```

返回示例：

```python
{"accuracy": 1.0}
```

或：

```python
{"accuracy": 0.75}
```

---

# 十三、函数八：`evaluate_model` 模型评估

源码：

```python
def evaluate_model(self, texts, labels):
    """评估模型性能"""
    # 仅对 texts 进行分词，labels 已为数字
    encodings = self.tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    dataset = self.create_dataset(encodings, labels)

    trainer = Trainer(model=self.model)
    predictions = trainer.predict(dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = labels  # 直接使用数字标签

    logger.info("分类报告:")
    logger.info(classification_report(
        true_labels,
        pred_labels,
        target_names=["通用知识", "专业咨询"]
    ))
    logger.info("混淆矩阵:")
    logger.info(confusion_matrix(true_labels, pred_labels))
```

## 1. 函数作用

```text
用验证数据评估模型效果，输出分类报告和混淆矩阵。
```

---

## 2. 输入示例

```python
texts = [
    "JAVA课程费用多少？",
    "5*9等于多少？",
    "AI培训有哪些老师？",
    "太阳为什么会发光？"
]

labels = [1, 0, 1, 0]
```

---

## 3. 对文本编码

```python
encodings = self.tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)
```

得到：

```python
encodings = {
    "input_ids": tensor(...),
    "attention_mask": tensor(...),
    "token_type_ids": tensor(...)
}
```

---

## 4. 创建 Dataset

```python
dataset = self.create_dataset(encodings, labels)
```

生成验证集 Dataset。

---

## 5. 使用 Trainer 预测

```python
trainer = Trainer(model=self.model)
predictions = trainer.predict(dataset)
```

作用：用当前模型对验证集进行预测。

预测结果中的核心数据是：

```python
predictions.predictions
```

它就是模型输出的 logits。

示例：

```python
predictions.predictions = [
    [0.8, 3.2],
    [4.1, 0.6],
    [0.4, 2.9],
    [3.7, 0.5]
]
```

---

## 6. logits 转类别

```python
pred_labels = np.argmax(predictions.predictions, axis=-1)
```

得到：

```python
pred_labels = [1, 0, 1, 0]
```

---

## 7. 真实标签

```python
true_labels = labels
```

例如：

```python
true_labels = [1, 0, 1, 0]
```

---

## 8. 分类报告

```python
classification_report(
    true_labels,
    pred_labels,
    target_names=["通用知识", "专业咨询"]
)
```

输出示例：

```text
              precision    recall  f1-score   support

        通用知识       1.00      1.00      1.00         2
        专业咨询       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

指标解释：

| 指标 | 含义 |
|---|---|
| `precision` | 预测为该类别的数据中，有多少是真的 |
| `recall` | 真实属于该类别的数据中，有多少被预测出来 |
| `f1-score` | precision 和 recall 的综合指标 |
| `support` | 该类别真实样本数量 |

---

## 9. 混淆矩阵

```python
confusion_matrix(true_labels, pred_labels)
```

输出示例：

```python
[[2, 0],
 [0, 2]]
```

表格理解：

| 真实 \ 预测 | 通用知识 | 专业咨询 |
|---|---:|---:|
| 通用知识 | 2 | 0 |
| 专业咨询 | 0 | 2 |

表示：

```text
2 条通用知识预测正确
2 条专业咨询预测正确
没有预测错误
```

---

# 十四、函数九：`predict_category` 预测类别

源码：

```python
def predict_category(self, query):
    # 检查模型是否加载
    if self.model is None:
        # 模型未加载，记录错误
        logger.error("模型未训练或加载")
        # 默认返回通用知识
        return "通用知识"
    # 对查询进行编码
    encoding = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
    # 将编码移到指定设备
    encoding = {k: v.to(self.device) for k, v in encoding.items()}
    # 不计算梯度，进行预测
    with torch.no_grad():
        # 获取模型输出
        outputs = self.model(**encoding)
        # 获取预测结果
        prediction = torch.argmax(outputs.logits, dim=1).item()
    # 根据预测结果返回类别
    return "专业咨询" if prediction == 1 else "通用知识"
```

## 1. 函数作用

```text
输入一个用户问题，返回它的类别。
```

---

## 2. 输入示例

```python
query = "JAVA课程费用多少？"
```

---

## 3. 检查模型是否加载

```python
if self.model is None:
    logger.error("模型未训练或加载")
    return "通用知识"
```

作用：如果模型没有加载成功，则默认返回“通用知识”。

---

## 4. tokenizer 编码

```python
encoding = self.tokenizer(
    query,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)
```

输入：

```text
JAVA课程费用多少？
```

经过 tokenizer：

```text
[CLS] java 课 程 费 用 多 少 ？ [SEP]
```

编码结果：

```python
encoding = {
    "input_ids": tensor([[101, 8179, 4923, 3621, 6589, 5508, 1914, 2208, 8043, 102]]),
    "attention_mask": tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    "token_type_ids": tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
}
```

这里是二维：

```text
[1, 序列长度]
```

其中：

```text
1 表示当前只预测 1 条问题。
```

---

## 5. 移动到设备

```python
encoding = {k: v.to(self.device) for k, v in encoding.items()}
```

作用：把 `input_ids`、`attention_mask`、`token_type_ids` 移动到 CPU 或 GPU 上。

---

## 6. 关闭梯度

```python
with torch.no_grad():
```

作用：预测阶段不需要训练参数，也不需要计算梯度。

这样可以减少内存消耗，提高推理效率。

---

## 7. 模型推理

```python
outputs = self.model(**encoding)
```

作用：把编码后的问题输入 BERT 分类模型。

模型输出中最重要的是：

```python
outputs.logits
```

---

## 8. logits 示例

模型可能输出：

```python
outputs.logits = tensor([[0.8, 3.2]])
```

含义：

| 类别编号 | 类别名称 | 分数 |
|---|---|---:|
| 0 | 通用知识 | 0.8 |
| 1 | 专业咨询 | 3.2 |

---

## 9. 取最大分数的类别

```python
prediction = torch.argmax(outputs.logits, dim=1).item()
```

通常：

```python
outputs.logits.shape = torch.Size([1, 2])
```

含义：

```text
1 → 当前有 1 条问题
2 → 每条问题有 2 个类别分数
```

`dim=1` 表示在类别维度上找最大值。

因为：

```text
3.2 > 0.8
```

所以：

```python
prediction = 1
```

---

## 10. 返回中文类别

```python
return "专业咨询" if prediction == 1 else "通用知识"
```

因为：

```python
prediction = 1
```

所以返回：

```text
专业咨询
```

---

## 11. 再代入一个通用知识例子

输入：

```python
query = "5*9等于多少？"
```

模型输出可能是：

```python
outputs.logits = tensor([[4.1, 0.5]])
```

含义：

| 类别编号 | 类别名称 | 分数 |
|---|---|---:|
| 0 | 通用知识 | 4.1 |
| 1 | 专业咨询 | 0.5 |

最大分数在位置 0：

```python
prediction = 0
```

最终返回：

```text
通用知识
```

---

# 十五、主程序：`if __name__ == "__main__"`

源码：

```python
if __name__ == "__main__":
    # 初始化分类器
    classifier = QueryClassifier(model_path="bert_query_classifier")

    # 训练模型
    # classifier.train_model(data_file='../classify_data/model_generic_5000.json')
    # 示例预测
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

## 1. 作用

```text
用于直接运行当前 Python 文件时进行测试。
```

---

## 2. 初始化分类器

```python
classifier = QueryClassifier(model_path="bert_query_classifier")
```

执行后：

```text
加载 tokenizer
加载模型
设置设备
设置标签映射
```

---

## 3. 训练代码被注释

```python
# classifier.train_model(data_file='../classify_data/model_generic_5000.json')
```

前面有 `#`，表示当前不会执行训练。

如果取消注释：

```python
classifier.train_model(data_file='../classify_data/model_generic_5000.json')
```

就会开始训练模型。

---

## 4. 测试问题

```python
test_queries = [
    "AI学科的课程大纲是什么",
    "JAVA课程费用多少？",
    "5*9等于多少？",
    "AI培训有哪些老师？"
]
```

---

## 5. 循环预测

```python
for query in test_queries:
    category = classifier.predict_category(query)
    print(f"查询: {query} -> 分类: {category}")
```

每次取一个问题，调用：

```python
predict_category(query)
```

得到类别。

---

## 6. 输出示例

```text
查询: AI学科的课程大纲是什么 -> 分类: 专业咨询
查询: JAVA课程费用多少？ -> 分类: 专业咨询
查询: 5*9等于多少？ -> 分类: 通用知识
查询: AI培训有哪些老师？ -> 分类: 专业咨询
```

---

# 十六、完整训练流程总结

```text
训练数据文件
    ↓
读取 JSON Lines
    ↓
拆出 query 和 label
    ↓
train_test_split 划分训练集 / 验证集
    ↓
preprocess_data
    ↓
query → input_ids / attention_mask / token_type_ids
label → 0 / 1
    ↓
create_dataset
    ↓
Dataset 样本
    ↓
Trainer
    ↓
训练模型
    ↓
save_model
    ↓
保存模型和 tokenizer
```

---

# 十七、完整预测流程总结

```text
用户问题
    ↓
predict_category
    ↓
tokenizer 编码
    ↓
input_ids / attention_mask / token_type_ids
    ↓
移动到 CPU / GPU
    ↓
BERT 分类模型
    ↓
outputs.logits
    ↓
torch.argmax(dim=1)
    ↓
prediction
    ↓
返回中文类别
```

---

# 十八、核心数据流总结

## 1. 原始训练数据

```json
{"query": "JAVA课程费用多少？", "label": "专业咨询"}
```

## 2. 拆分后

```python
text = "JAVA课程费用多少？"
label = "专业咨询"
```

## 3. tokenizer 后

```python
{
    "input_ids": tensor([[101, 8179, 4923, ..., 102]]),
    "attention_mask": tensor([[1, 1, 1, ..., 1]]),
    "token_type_ids": tensor([[0, 0, 0, ..., 0]])
}
```

## 4. 标签转换后

```python
"专业咨询" → 1
```

## 5. Dataset 单条样本

```python
{
    "input_ids": tensor([...]),
    "attention_mask": tensor([...]),
    "token_type_ids": tensor([...]),
    "labels": tensor(1)
}
```

## 6. 模型输出

```python
outputs.logits = tensor([[0.8, 3.2]])
```

## 7. argmax 后

```python
prediction = 1
```

## 8. 返回结果

```text
专业咨询
```

---

# 十九、常见易混点

## 1. tokenizer 是模型吗？

不是。

```text
tokenizer 只负责文本转数字。
```

---

## 2. Dataset 是数据库吗？

不是。

```text
Dataset 是训练数据包装格式。
```

---

## 3. Trainer 是模型吗？

不是。

```text
Trainer 是训练器，负责训练模型。
```

---

## 4. logits 是最终结果吗？

不是。

```text
logits 是模型输出的类别分数。
```

还需要用 `argmax` 取最大值位置。

---

## 5. `dim=1` 为什么是类别维度？

因为：

```python
outputs.logits.shape = [样本数量, 类别数量]
```

例如：

```python
torch.Size([1, 2])
```

其中：

```text
第0维：样本数量
第1维：类别数量
```

所以要在类别维度上取最大值：

```python
torch.argmax(outputs.logits, dim=1)
```

---

# 二十、课堂练习

## 练习1

已知：

```python
self.label_map = {
    "通用知识": 0,
    "专业咨询": 1
}
```

问题：

```text
“专业咨询”对应数字几？
```

答案：

```text
1
```

---

## 练习2

已知模型输出：

```python
outputs.logits = tensor([[0.6, 3.4]])
```

问题：预测类别是什么？

答案：

```text
专业咨询
```

解释：

```text
3.4 最大，位置是 1，1 对应专业咨询。
```

---

## 练习3

已知模型输出：

```python
outputs.logits = tensor([[4.2, 0.8]])
```

问题：预测类别是什么？

答案：

```text
通用知识
```

解释：

```text
4.2 最大，位置是 0，0 对应通用知识。
```

---

## 练习4

已知：

```python
attention_mask = [1, 1, 1, 1, 0, 0]
```

问题：最后两个 0 表示什么？

答案：

```text
最后两个位置是 padding 补齐内容，不需要模型关注。
```

---

## 练习5

已知：

```python
dataset[0] = {
    "input_ids": tensor([101, 8179, 4923, 102]),
    "attention_mask": tensor([1, 1, 1, 1]),
    "token_type_ids": tensor([0, 0, 0, 0]),
    "labels": tensor(1)
}
```

问题：`labels = tensor(1)` 表示什么？

答案：

```text
这条样本的真实类别是专业咨询。
```

---

# 二十一、最终总结

这个 BERT 意图识别模块可以用一句话总结：

```text
它是一个基于 BERT 的问题二分类器，用于判断用户问题属于“通用知识”还是“专业咨询”。
```

训练阶段：

```text
读取 query 和 label
    ↓
query 通过 tokenizer 转成数字
    ↓
label 通过 label_map 转成 0 / 1
    ↓
封装成 Dataset
    ↓
交给 Trainer 训练
    ↓
保存模型
```

预测阶段：

```text
输入用户问题
    ↓
tokenizer 编码
    ↓
BERT 分类模型输出 logits
    ↓
argmax 取最大分数位置
    ↓
返回通用知识或专业咨询
```

最核心记忆链路：

```text
文本 → tokenizer → input_ids → BERT → logits → argmax → 类别
```
