from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import torch.nn.functional as F
import numpy as np

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_model_name = "bert-base-uncased"
student_model_name = "distilbert-base-uncased"

# 加载教师模型和分词器，并移动到对应设备
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name, num_labels=2)
teacher_model.to(device)
teacher_model.eval()
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# 加载学生模型和分词器
student_model = AutoModelForSequenceClassification.from_pretrained(student_model_name, num_labels=2)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# 加载数据集
dataset = load_dataset("stanfordnlp/imdb")


# 预处理数据：不仅要分词，还要将 'label' 重命名/映射为 'labels' 以适配 Trainer
def preprocess_data(examples):
    result = teacher_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    result['labels'] = examples['label']  # 适配 Trainer 的 labels 键名
    return result


# 建议处理时移除原文字段，避免传给模型时多余
encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=["text", "label"])


def get_teacher_logits(batch, device):
    # 将 batch 数据移动到与教师模型相同的设备上
    inputs = {k: v.to(device) for k, v in batch.items() if k in teacher_tokenizer.model_input_names}
    with torch.no_grad():
        outputs = teacher_model(**inputs)
    return outputs.logits


# 完善蒸馏损失函数：结合软标签损失 (KL Divergence) 和硬标签损失 (Cross Entropy)
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0):
    # 1. 软标签损失
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kd_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    # 2. 硬标签损失（真实标签交叉熵）
    ce_loss = F.cross_entropy(student_logits, labels)

    # 加权组合
    return alpha * kd_loss + (1 - alpha) * ce_loss


class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 从 inputs 中安全取出 labels 并从 inputs 中弹出，防止传给模型的 forward 报错
        labels = inputs.pop("labels")

        # 学生模型输出
        outputs = model(**inputs)
        student_logits = outputs.logits

        # 获取教师模型输出（需要确保在同一设备）
        teacher_logits = get_teacher_logits(inputs, teacher_model.device)

        # 计算综合蒸馏损失
        loss = distillation_loss(student_logits, teacher_logits, labels)

        return (loss, outputs) if return_outputs else loss


# 设置训练参数（使用更新后的 eval_strategy）
training_args = TrainingArguments(
    output_dir="./distilled_model",
    eval_strategy="epoch",  # 旧版本为 evaluation_strategy
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    fp16=True if torch.cuda.is_available() else False,  # 推荐开启混合精度加速
)

# 初始化训练器
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
)

# 开始训练
trainer.train()
