import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    RobertaForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# 1. 加载SST-2数据集并选择子集
dataset = load_dataset("nyu-mll/glue", "sst2")
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
test_dataset = dataset["validation"].shuffle(seed=42).select(range(500))

# 2. 加载分词器并对数据进行预处理（SST-2 的文本字段名为 "sentence"）
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["sentence", "idx"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["sentence", "idx"])

# 3. 定义准确率计算函数（解决 KeyError 问题）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    return {"accuracy": acc}

# 4. 初始化教师模型 (RoBERTa-large) 和学生模型 (DistilRoBERTa)
# 注意：distilroberta-base 属于 RoBERTa 架构，应使用 RobertaForSequenceClassification
teacher_model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2)
student_model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)

# 5. 训练参数设置
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True if torch.cuda.is_available() else False,  # 推荐开启混合精度加速
)

# 6. 创建Trainer函数
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

    # 记录推理时间
    start_eval = time.time()
    metrics = trainer.evaluate()
    eval_time = time.time() - start_eval

    # 模型大小（百万参数量）
    model_size = sum(p.numel() for p in model.parameters()) / 1e6

    # 输出结果
    print(f"\n--- {model_name} 模型 ---")
    print(f"训练时间: {train_time:.2f} 秒")
    print(f"推理时间: {eval_time:.2f} 秒")
    print(f"验证集损失: {metrics['eval_loss']:.4f}")
    print(f"验证集准确率: {metrics['eval_accuracy']:.4f}")
    print(f"模型参数量: {model_size:.2f}M\n")

# 7. 依次训练和评估教师模型与学生模型
train_and_evaluate(teacher_model, train_dataset, test_dataset, model_name="教师模型 (RoBERTa-large)")
train_and_evaluate(student_model, train_dataset, test_dataset, model_name="学生模型 (DistilRoBERTa)")