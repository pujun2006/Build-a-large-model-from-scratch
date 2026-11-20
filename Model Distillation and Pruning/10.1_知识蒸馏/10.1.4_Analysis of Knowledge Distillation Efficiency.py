import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# 1. 加载数据集并选择子集
dataset = load_dataset("stanfordnlp/imdb")
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))

# 2. 加载对应的分词器（这里使用 BERT 的分词器处理两个模型均可）
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 数据预处理函数：将文本转换为 Token ID
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# 对训练集和测试集进行分词，并将原文字段移除
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. 定义准确率计算函数（解决 KeyError 问题）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    return {"accuracy": acc}

# 初始化教师模型（BERT-base）和学生模型（DistilBERT）
teacher_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
student_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 训练参数设置（使用更新后的 eval_strategy）
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True if torch.cuda.is_available() else False,  # 推荐开启混合精度加速
)

# 创建并执行训练和评估的函数
def train_and_evaluate(model, train_data, test_data, model_name=""):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics  # 传入评估指标函数
    )

    # 记录训练时间
    start_train = time.time()
    trainer.train()
    train_time = time.time() - start_train

    # 记录推理/评估时间
    start_eval = time.time()
    metrics = trainer.evaluate()
    eval_time = time.time() - start_eval

    # 计算模型参数量（百万单位）
    model_size = sum(p.numel() for p in model.parameters()) / 1e6

    # 输出结果
    print(f"\n--- {model_name} 模型 ---")
    print(f"训练时间: {train_time:.2f} 秒")
    print(f"推理评估时间: {eval_time:.2f} 秒")
    print(f"验证集损失: {metrics['eval_loss']:.4f}")
    print(f"验证集准确率: {metrics['eval_accuracy']:.4f}")
    print(f"模型参数量: {model_size:.2f}M\n")

# 依次训练和评估教师模型与学生模型
train_and_evaluate(teacher_model, train_dataset, test_dataset, model_name="教师模型 (BERT-base)")
train_and_evaluate(student_model, train_dataset, test_dataset, model_name="学生模型 (DistilBERT)")
