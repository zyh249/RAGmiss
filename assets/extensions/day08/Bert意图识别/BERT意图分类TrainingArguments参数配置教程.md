# BERT 意图分类 TrainingArguments 参数配置教程

## 1. 场景说明

本教程针对如下任务：

> 输入用户问题 `query`，判断它属于 **通用知识** 还是 **专业咨询**。

这是一个典型的 **BERT 文本二分类微调任务**。

你的标签映射如下：

```python
self.label_map = {
    "通用知识": 0,
    "专业咨询": 1
}
```

训练数据格式类似：

```json
{"query": "感冒了应该吃什么药？", "label": "专业咨询"}
{"query": "中国的首都是哪里？", "label": "通用知识"}
```

核心训练参数集中在：

```python
TrainingArguments(...)
```

企业中配置这些参数时，通常要结合以下因素：

```text
1. 数据量大小
2. 模型大小
3. CPU / GPU 环境
4. GPU 显存大小
5. 是否容易过拟合
6. 是否追求训练速度
7. 是否追求模型稳定性
```

---

# 2. 你当前代码中的参数

你当前的配置如下：

```python
training_args = TrainingArguments(
    output_dir="./bert_result",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./bert_logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    save_strategy="epoch",
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    fp16=False
)
```

这套配置可以训练，但是在企业项目中还可以进一步优化：

```text
1. learning_rate 没有显式设置
2. warmup_steps=500 对 5000 条数据可能偏大
3. 没有设置 gradient_accumulation_steps
4. 没有启用 compute_metrics
5. metric_for_best_model 只能基于 eval_loss
6. 没有设置 report_to="none"
7. 没有设置 seed，实验复现性一般
```

---

# 3. output_dir：模型输出目录

## 3.1 作用

```python
output_dir="./bert_result"
```

用于保存训练过程中的：

```text
1. checkpoint
2. 训练状态
3. 中间模型
4. 最终模型相关文件
```

## 3.2 企业中如何设置

本地简单测试可以写：

```python
output_dir="./bert_result"
```

企业项目建议写得更清晰：

```python
output_dir="./outputs/query_classifier_bert"
```

如果要做多组实验，可以写：

```python
output_dir="./outputs/query_classifier_bert_lr2e-5_bs16_epoch3"
```

## 3.3 推荐规则

| 场景 | 推荐配置 |
|---|---|
| 本地测试 | `./bert_result` |
| 企业项目 | `./outputs/query_classifier_bert` |
| 多版本训练 | `./outputs/query_classifier_bert_v1` |
| 实验对比 | `./outputs/bert_lr2e-5_bs16_epoch3` |

---

# 4. num_train_epochs：训练轮数

## 4.1 作用

```python
num_train_epochs=3
```

表示整个训练集被模型完整学习几遍。

例如有 5000 条数据：

```text
epoch=1：5000 条数据训练 1 遍
epoch=3：5000 条数据训练 3 遍
epoch=5：5000 条数据训练 5 遍
```

## 4.2 企业中如何设置

| 数据量 | 推荐 epoch |
|---|---:|
| 1000 条以内 | 5 到 10 |
| 1000 到 1 万条 | 3 到 5 |
| 1 万到 10 万条 | 2 到 4 |
| 10 万条以上 | 1 到 3 |

你的数据大约是 5000 条，所以推荐：

```python
num_train_epochs=3
```

如果验证集效果还在提升，可以试：

```python
num_train_epochs=5
```

但是不建议一上来设置 20 或 30，因为 BERT 小数据微调很容易过拟合。

---

# 5. per_device_train_batch_size：训练 batch size

## 5.1 作用

```python
per_device_train_batch_size=8
```

表示每张 GPU 或 CPU 每一步训练时处理多少条样本。

例如：

```text
batch_size=8
表示模型一次看 8 条 query。
```

这个参数主要受 **显存大小** 影响。

---

## 5.2 CPU 配置

如果没有 GPU，只用 CPU：

```python
per_device_train_batch_size=2
```

或者：

```python
per_device_train_batch_size=4
```

CPU 上不建议设置太大，否则训练很慢。

---

## 5.3 4GB GPU 配置

例如 GTX 1650 4GB：

```python
per_device_train_batch_size=4
```

如果显存不足：

```python
per_device_train_batch_size=2
```

---

## 5.4 6GB 到 8GB GPU 配置

例如 RTX 2060 6GB、RTX 3060 Laptop、RTX 2070 8GB：

```python
per_device_train_batch_size=8
```

如果显存充足，可以尝试：

```python
per_device_train_batch_size=16
```

---

## 5.5 12GB GPU 配置

例如 RTX 3060 12GB：

```python
per_device_train_batch_size=16
```

---

## 5.6 16GB GPU 配置

例如 T4 16GB、RTX 4080 Laptop 16GB：

```python
per_device_train_batch_size=16
```

或者：

```python
per_device_train_batch_size=32
```

---

## 5.7 24GB GPU 配置

例如 RTX 3090、RTX 4090、A10：

```python
per_device_train_batch_size=32
```

或者：

```python
per_device_train_batch_size=64
```

---

# 6. gradient_accumulation_steps：梯度累积

## 6.1 作用

你当前没有设置该参数，默认是：

```python
gradient_accumulation_steps=1
```

它的作用是：

```text
显存不够时，用多次小 batch 模拟一个大 batch。
```

公式：

```text
有效 batch_size = per_device_train_batch_size × gradient_accumulation_steps × GPU 数量
```

例如：

```python
per_device_train_batch_size=4
gradient_accumulation_steps=4
```

等价于：

```text
有效 batch_size = 4 × 4 = 16
```

也就是说，显存一次只能放 4 条数据，但是模型累计 4 次梯度之后再更新一次参数，效果接近一次训练 16 条数据。

---

## 6.2 企业推荐配置

| 设备 | batch_size | gradient_accumulation_steps | 有效 batch |
|---|---:|---:|---:|
| CPU | 2 | 4 | 8 |
| 4GB GPU | 4 | 4 | 16 |
| 6GB GPU | 8 | 2 | 16 |
| 8GB GPU | 8 | 2 | 16 |
| 12GB GPU | 16 | 1 | 16 |
| 16GB GPU | 16 | 2 | 32 |
| 24GB GPU | 32 | 1 | 32 |

对于 BERT 二分类任务，推荐有效 batch 控制在：

```text
16 到 32
```

---

# 7. per_device_eval_batch_size：验证 batch size

## 7.1 作用

```python
per_device_eval_batch_size=8
```

表示验证模型时，每次处理多少条验证集数据。

验证阶段不需要反向传播，所以显存压力比训练小。

## 7.2 推荐配置

| 设备 | 推荐值 |
|---|---:|
| CPU | 8 |
| 4GB GPU | 8 |
| 6GB GPU | 16 |
| 8GB GPU | 16 |
| 12GB GPU | 32 |
| 16GB GPU | 32 |
| 24GB GPU | 64 |

如果你有 GPU，可以从下面开始：

```python
per_device_eval_batch_size=16
```

---

# 8. learning_rate：学习率

## 8.1 作用

学习率控制模型参数每次更新的幅度。

学习率太大：

```text
模型震荡，loss 不稳定，甚至训练失败。
```

学习率太小：

```text
训练很慢，loss 下降不明显。
```

## 8.2 BERT 微调常用学习率

| 学习率 | 特点 |
|---|---|
| `1e-5` | 很稳，但训练慢 |
| `2e-5` | 企业常用，稳定 |
| `3e-5` | 常用，速度和效果平衡 |
| `5e-5` | 学得快，但小数据容易不稳定 |
| `1e-4` | 对 BERT 微调偏大，一般不建议 |

你的数据量约 5000 条，推荐：

```python
learning_rate=2e-5
```

如果 loss 下降太慢，可以尝试：

```python
learning_rate=3e-5
```

---

# 9. warmup_steps / warmup_ratio：学习率预热

## 9.1 作用

你的代码中：

```python
warmup_steps=500
```

作用是让学习率从较小值逐渐升高，避免训练刚开始就剧烈震荡。

## 9.2 为什么 warmup_steps=500 可能偏大

假设你有 5000 条数据，训练集占 80%，则训练集约 4000 条。

如果：

```python
per_device_train_batch_size=8
num_train_epochs=3
```

每个 epoch 的 step 数约为：

```text
4000 / 8 = 500 step
```

总 step 约为：

```text
500 × 3 = 1500 step
```

如果设置：

```python
warmup_steps=500
```

相当于总训练过程的三分之一都在预热，可能偏大。

## 9.3 企业推荐写法

更推荐使用比例：

```python
warmup_ratio=0.1
```

表示总训练步数的 10% 用于预热。

## 9.4 推荐规则

| 总训练 steps | 推荐 warmup_steps |
|---:|---:|
| 500 | 50 |
| 1000 | 100 |
| 1500 | 100 到 150 |
| 5000 | 500 |
| 10000 | 1000 |

你的任务建议使用：

```python
warmup_ratio=0.1
```

---

# 10. weight_decay：权重衰减

## 10.1 作用

```python
weight_decay=0.01
```

用于防止模型过拟合。

简单理解：

```text
不要让模型参数变得过大。
```

## 10.2 企业推荐配置

| 场景 | 推荐值 |
|---|---:|
| 小数据，容易过拟合 | 0.01 |
| 中等数据 | 0.01 |
| 大数据 | 0.001 到 0.01 |
| 不想加正则 | 0 |

你的任务推荐保留：

```python
weight_decay=0.01
```

---

# 11. logging_steps：日志打印频率

## 11.1 作用

```python
logging_steps=10
```

表示每训练 10 个 step 打印一次日志。

## 11.2 推荐配置

| 数据规模 | logging_steps |
|---|---:|
| 5000 条以内 | 10 或 20 |
| 1 万到 10 万条 | 50 或 100 |
| 10 万条以上 | 100 或 500 |

你的数据约 5000 条，可以设置：

```python
logging_steps=20
```

如果想看得更细，可以保留：

```python
logging_steps=10
```

---

# 12. evaluation_strategy：验证策略

## 12.1 作用

```python
evaluation_strategy="epoch"
```

表示每个 epoch 结束后，在验证集上评估一次。

## 12.2 常见取值

```python
evaluation_strategy="no"
evaluation_strategy="steps"
evaluation_strategy="epoch"
```

## 12.3 推荐配置

| 场景 | 推荐配置 |
|---|---|
| 小数据 | `"epoch"` |
| 中大型数据 | `"steps"` |
| 快速调试 | `"steps"` + `eval_steps=50` |
| 正式训练 | `"epoch"` 或 `"steps"` |

你的任务推荐：

```python
evaluation_strategy="epoch"
```

如果想更快观察验证效果，可以改成：

```python
evaluation_strategy="steps"
eval_steps=100
```

---

# 13. save_strategy：保存策略

## 13.1 作用

```python
save_strategy="epoch"
```

表示每个 epoch 结束后保存一次 checkpoint。

## 13.2 推荐原则

如果使用：

```python
load_best_model_at_end=True
```

建议让：

```python
evaluation_strategy
save_strategy
```

保持一致。

例如：

```python
evaluation_strategy="epoch"
save_strategy="epoch"
```

或者：

```python
evaluation_strategy="steps"
eval_steps=100
save_strategy="steps"
save_steps=100
```

---

# 14. save_total_limit：checkpoint 保留数量

## 14.1 作用

```python
save_total_limit=1
```

表示最多保留 1 个 checkpoint，避免磁盘占用过大。

## 14.2 企业推荐配置

| 场景 | 推荐值 |
|---|---:|
| 本地磁盘小 | 1 |
| 正式实验 | 2 或 3 |
| 大模型训练 | 2 |
| 需要回滚 | 3 到 5 |

建议改成：

```python
save_total_limit=2
```

这样既不会占用太多磁盘，也能保留一定回滚空间。

---

# 15. load_best_model_at_end：训练结束加载最佳模型

## 15.1 作用

```python
load_best_model_at_end=True
```

表示训练结束后，不一定使用最后一轮模型，而是自动加载验证集效果最好的模型。

例如：

```text
第 1 轮：效果一般
第 2 轮：效果最好
第 3 轮：开始过拟合
```

开启该参数后，最终会加载第 2 轮的模型。

企业项目中建议保留：

```python
load_best_model_at_end=True
```

---

# 16. metric_for_best_model：选择最佳模型的指标

## 16.1 当前写法

```python
metric_for_best_model="eval_loss"
```

表示根据验证集 loss 最低选择最佳模型。

这可以使用，但分类任务中通常更建议使用：

```python
metric_for_best_model="f1"
```

前提是你需要配置：

```python
compute_metrics=self.compute_metrics
```

## 16.2 为什么企业中推荐 F1

如果数据不平衡，只看 accuracy 可能有问题。

例如：

```text
通用知识：4500 条
专业咨询：500 条
```

如果模型全部预测为“通用知识”，准确率也可能达到 90%，但对“专业咨询”完全没有识别能力。

因此二分类任务更推荐关注：

```text
precision
recall
f1
confusion_matrix
```

推荐配置：

```python
metric_for_best_model="f1"
greater_is_better=True
```

---

# 17. fp16：半精度训练

## 17.1 作用

```python
fp16=False
```

表示是否使用 FP16 半精度训练。

FP16 的好处：

```text
1. 更省显存
2. 训练速度更快
```

但是 CPU 不能使用 FP16。

## 17.2 推荐配置

| 设备 | fp16 | 说明 |
|---|---|---|
| CPU | False | CPU 不使用 fp16 |
| GTX 10 系 | False 或谨慎尝试 | 可能不稳定 |
| GTX 16 系 | 可以尝试 True | 不稳定就关闭 |
| RTX 20 系 | True | 推荐开启 |
| RTX 30 系 | True | 推荐开启 |
| RTX 40 系 | True | 推荐开启 |
| T4 | True | 推荐开启 |
| A10 | True | 推荐开启 |
| A100 | 可用 bf16 | 更推荐 bf16 |
| H100 | 可用 bf16 | 更推荐 bf16 |

通用写法：

```python
fp16=torch.cuda.is_available()
```

如果是 CPU，会自动是 False。

---

# 18. dataloader_num_workers：数据加载线程数

## 18.1 作用

```python
dataloader_num_workers=0
```

表示数据加载时使用多少个子进程。

## 18.2 Windows 推荐

Windows 本地训练建议先设置：

```python
dataloader_num_workers=0
```

原因是 Windows 多进程数据加载有时容易出现兼容问题。

## 18.3 Linux 服务器推荐

如果在 Linux GPU 服务器上训练，可以设置：

```python
dataloader_num_workers=2
```

或者：

```python
dataloader_num_workers=4
```

---

# 19. report_to：日志上报工具

## 19.1 作用

有些环境中，Trainer 可能会尝试接入 wandb、tensorboard 等工具。

如果只是本地训练，建议关闭外部上报：

```python
report_to="none"
```

这样可以减少不必要的报错和干扰。

---

# 20. seed：随机种子

## 20.1 作用

```python
seed=42
```

用于保证实验可复现。

如果不设置，每次划分数据、初始化参数、训练过程可能略有不同。

企业实验中建议固定：

```python
seed=42
```

---

# 21. CPU 配置模板

适合：

```text
没有 CUDA
只想跑通训练流程
训练速度要求不高
```

```python
training_args = TrainingArguments(
    output_dir="./outputs/query_classifier_bert",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs/query_classifier_bert",
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
    dataloader_num_workers=0,
    report_to="none",
    seed=42
)
```

---

# 22. 4GB GPU 配置模板

适合：

```text
GTX 1650 4GB
低显存笔记本 GPU
```

```python
training_args = TrainingArguments(
    output_dir="./outputs/query_classifier_bert",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs/query_classifier_bert",
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    dataloader_num_workers=0,
    report_to="none",
    seed=42
)
```

如果显存不足，改成：

```python
per_device_train_batch_size=2
gradient_accumulation_steps=8
```

---

# 23. 6GB 到 8GB GPU 配置模板

适合：

```text
RTX 2060 6GB
RTX 3060 Laptop
RTX 2070 8GB
RTX 4060 Laptop 8GB
```

```python
training_args = TrainingArguments(
    output_dir="./outputs/query_classifier_bert",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs/query_classifier_bert",
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    dataloader_num_workers=0,
    report_to="none",
    seed=42
)
```

有效 batch size：

```text
8 × 2 = 16
```

---

# 24. 12GB GPU 配置模板

适合：

```text
RTX 3060 12GB
RTX 4070 12GB
```

```python
training_args = TrainingArguments(
    output_dir="./outputs/query_classifier_bert",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs/query_classifier_bert",
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    dataloader_num_workers=2,
    report_to="none",
    seed=42
)
```

---

# 25. 16GB GPU 配置模板

适合：

```text
Tesla T4 16GB
RTX 4080 Laptop 16GB
A4000 16GB
```

```python
training_args = TrainingArguments(
    output_dir="./outputs/query_classifier_bert",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs/query_classifier_bert",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    dataloader_num_workers=2,
    report_to="none",
    seed=42
)
```

有效 batch size：

```text
16 × 2 = 32
```

---

# 26. 24GB GPU 配置模板

适合：

```text
RTX 3090
RTX 4090
A10
A5000
```

```python
training_args = TrainingArguments(
    output_dir="./outputs/query_classifier_bert",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs/query_classifier_bert",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    dataloader_num_workers=4,
    report_to="none",
    seed=42
)
```

---

# 27. 企业推荐版：加入 F1 指标

## 27.1 添加导入

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
```

## 27.2 在 QueryClassifier 类中添加方法

```python
def compute_metrics(self, eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
```

## 27.3 修改 Trainer

```python
trainer = Trainer(
    model=self.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=self.compute_metrics,
)
```

## 27.4 修改最佳模型指标

```python
metric_for_best_model="f1"
greater_is_better=True
```

---

# 28. 最终推荐 TrainingArguments

这是当前项目最推荐的企业版配置：

```python
training_args = TrainingArguments(
    output_dir="./outputs/query_classifier_bert",

    # 基础训练参数
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,

    # 优化参数
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # 日志与评估
    logging_dir="./logs/query_classifier_bert",
    logging_steps=20,
    evaluation_strategy="epoch",

    # 保存策略
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,

    # 根据 F1 选择最佳模型
    metric_for_best_model="f1",
    greater_is_better=True,

    # 混合精度
    fp16=torch.cuda.is_available(),

    # Windows 本地训练建议先用 0
    dataloader_num_workers=0,

    # 避免 wandb 等外部日志工具干扰
    report_to="none",

    # 固定随机种子
    seed=42
)
```

配套 Trainer：

```python
trainer = Trainer(
    model=self.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=self.compute_metrics,
)
```

---

# 29. 调参顺序

企业中不要所有参数一起乱改，建议按照以下顺序调整。

## 29.1 第一步：先保证能跑

```python
num_train_epochs=1
per_device_train_batch_size=2
fp16=False
```

先把完整流程跑通。

---

## 29.2 第二步：解决显存问题

如果出现：

```text
CUDA out of memory
```

优先减小：

```python
per_device_train_batch_size
```

例如：

```text
16 -> 8 -> 4 -> 2
```

然后增加：

```python
gradient_accumulation_steps
```

例如：

```python
gradient_accumulation_steps=4
```

---

## 29.3 第三步：观察 loss 是否下降

如果 loss 不下降，可以尝试增大学习率：

```python
learning_rate=3e-5
```

如果 loss 波动很大，可以减小学习率：

```python
learning_rate=1e-5
```

---

## 29.4 第四步：观察验证集 F1

如果训练集效果好，验证集效果差，说明可能过拟合。

可以减少 epoch：

```python
num_train_epochs=2
```

或者保持：

```python
weight_decay=0.01
```

不要盲目增加训练轮数。

---

## 29.5 第五步：看类别是否不平衡

如果“专业咨询”经常识别错，说明可能类别不平衡。

这时不要只看 accuracy，要看：

```text
precision
recall
f1
confusion_matrix
```

---

# 30. 最小记忆版

```text
CPU：batch=2，累积=4，fp16=False
4GB GPU：batch=4，累积=4，fp16=True
8GB GPU：batch=8，累积=2，fp16=True
12GB GPU：batch=16，累积=1，fp16=True
16GB GPU：batch=16，累积=2，fp16=True
24GB GPU：batch=32，累积=1，fp16=True

学习率：2e-5
训练轮数：3
预热比例：warmup_ratio=0.1
权重衰减：weight_decay=0.01
最佳模型指标：优先使用 f1
```

---

# 31. 当前项目建议配置

针对你的数据：

```text
数据量：约 5000 条
任务类型：BERT 二分类
最大长度：128
标签数量：2
```

建议：

```python
num_train_epochs=3
per_device_train_batch_size=8
per_device_eval_batch_size=16
gradient_accumulation_steps=2
learning_rate=2e-5
warmup_ratio=0.1
weight_decay=0.01
fp16=torch.cuda.is_available()
metric_for_best_model="f1"
greater_is_better=True
report_to="none"
seed=42
```

也就是说，你当前最应该修改的是：

```python
learning_rate=2e-5
warmup_ratio=0.1
gradient_accumulation_steps=2
per_device_eval_batch_size=16
save_total_limit=2
report_to="none"
compute_metrics=self.compute_metrics
metric_for_best_model="f1"
greater_is_better=True
```
