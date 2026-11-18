Fine-tuning in Practice.py代码完整演示了在深度学习和自然语言处理（NLP）领域中，使用 **PyTorch** 和 **Hugging Face Transformers** 库对 BERT 模型进行**全流程优化与训练**的经典步骤。

代码涵盖了从模型微调、层级冻结、超参数优化、性能评估、推理加速（量化）到知识蒸馏的6个核心板块。以下是针对代码中各个模块的详细功能解析：

---

### 一、 整体初始化与数据准备

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)

texts = [
    'I love this product, it works perfectly!',
    'Worst experience ever, completely disappointed.',
]
labels = torch.tensor([1, 0])
inputs = tokenizer(
    texts, return_tensors='pt', padding=True, truncation=True, max_length=32
)

```

* **分词器（Tokenizer）与模型加载**：加载了预训练的 `bert-base-uncased`（不区分大小写的基础版 BERT）。`BertForSequenceClassification` 是带有分类头（Classification Head）的 BERT 模型，`num_labels=2` 表示这是一个**二分类任务**（如情感分析：正面/负面）。
* **数据编码**：将两句简短的英文文本转换为模型所需的张量格式。`padding=True` 确保长度一致，`truncation=True` 截断超过限制的文本，`max_length=32` 限制最大长度。

---

### 二、 层级冻结（Layer Freezing）

```python
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False

```

* **原理与作用**：BERT-base 模型通常包含 12 个 Transformer 编码器层。这里通过切片操作 `[:6]` 将**前 6 层**的参数冻结（`requires_grad = False`）。
* **目的**：前几层主要负责提取底层的通用语言特征（如语法、词性），在下游任务中通常不需要改变。冻结它们可以**显著减少显存占用、加快训练速度**，并防止在小数据集上发生过拟合。

---

### 三、 超参数调整与微调训练

```python
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5,
    weight_decay=0.01,
)

model.train()
for epoch in range(2):
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

```

* **参数过滤**：使用 `filter(lambda p: p.requires_grad, ...)` 只把需要求导更新的参数（即后 6 层和分类头）传给优化器，避免为冻结参数分配梯度缓存。
* **优化器选择**：采用 `AdamW`（带权重衰减的 Adam），这是 Transformer 模型微调的标配优化器，学习率设为经典的 `2e-5`。
* **训练循环**：进行 2 个周期的微调。每次前向传播获取损失 `loss`，然后通过 `loss.backward()` 反向传播计算梯度，最后用 `optimizer.step()` 更新权重。

---

### 四、 模型评估（Evaluation）

```python
def evaluate_model(model, inputs, labels):
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)

        y_true = labels.numpy()
        y_pred = predictions.numpy()

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return acc, precision, recall, f1

```

* **评估模式**：`model.eval()` 关闭 Dropout 和 BatchNorm 的训练特性。`with torch.no_grad()` 禁用梯度计算，节省内存并加快推理。
* **指标计算**：通过 `torch.argmax` 获取概率最高的类别索引作为预测值。利用 `scikit-learn` 计算了四项核心分类指标：
* **Accuracy（准确率）**：整体预测正确的比例。
* **Precision（精确率）**：预测为正类的样本中实际为正类的比例。
* **Recall（召回率）**：实际为正类的样本中被正确预测出来的比例。
* **F1 Score（F1分数）**：精确率和召回率的调和平均数。
* *注：`zero_division=0` 用于防止在样本极少或极端情况下分母为 0 时报错。*



---

### 五、 推理优化说明（Quantization）

```python
print(
    'Note: quantize_dynamic on Hugging Face models may require torch.fx or Hugging Face Optimum.'
)

```

* **代码意图**：原脚本尝试使用 PyTorch 原生的动态量化（`quantize_dynamic`）将线性层的权重转换为 `qint8`（8位整型），以减小模型体积并加速 CPU 推理。
* **实际注意事项**：直接对复杂的 Hugging Face 变压器模型结构使用原生动态量化容易引发兼容性报错。在实际工程项目中，推荐使用专用的 **Hugging Face Optimum** 库或 Intel IPEX 库来进行模型量化。

---

### 六、 知识蒸馏（Knowledge Distillation）

```python
model.eval()
with torch.no_grad():
    teacher_logits = model(**inputs).logits

student_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    teacher_probs = torch.nn.functional.softmax(
        teacher_logits / temperature, dim=-1
    )
    student_probs = torch.nn.functional.log_softmax(
        student_logits / temperature, dim=-1
    )
    return (
        torch.nn.functional.kl_div(
            student_probs, teacher_probs, reduction='batchmean'
        )
        * (temperature**2)
    )

```

* **教师模型与学生模型**：
* **教师模型（Teacher）**：这里将前面微调好的大模型作为教师模型，并用 `with torch.no_grad()` 锁死其参数和计算图，提取其输出的软标签（`teacher_logits`）。
* **学生模型（Student）**：重新初始化一个未经微调的全新 BERT 作为学生模型，准备通过模仿教师模型的输出来加速训练。


* **蒸馏损失函数（Distillation Loss）**：
* 引入了 **Temperature（温度系数，代码中为 2.0）**。通过对 logits 除以温度，平滑概率分布，使学生模型不仅能学到“哪个是对的”，还能学到“错的选项之间有什么相似性”（暗含的丰富信息）。
* 使用 **KL 散度（`kl_div`）** 来衡量学生输出分布与教师输出分布之间的差距，最后乘以 `temperature ** 2` 进行梯度幅度的缩放补偿，这是经典的 **Hinton 知识蒸馏** 标准实现。
---


### 七、 运行结果
<img width="344" height="48" alt="d52c754a-ffef-42d5-b4b7-6c88a858e8fe" src="https://github.com/user-attachments/assets/b5ecef4c-834b-41c5-ae5a-a1ce6f75bd60" />
