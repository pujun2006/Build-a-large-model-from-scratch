from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.quantization import quantize_dynamic
from transformers import BertForSequenceClassification, BertTokenizer

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)

# 1. 数据准备：清洗和分割数据集
texts = [
    'I love this product, it works perfectly!',
    'Worst experience ever, completely disappointed.',
]
labels = torch.tensor([1, 0])
inputs = tokenizer(
    texts, return_tensors='pt', padding=True, truncation=True, max_length=32
)

# 2. 层级冻结：冻结前6层
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False

# 3. 超参数调整：只将需要训练的参数（requires_grad=True）传入优化器
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5,
    weight_decay=0.01,
)

# 微调训练
model.train()
for epoch in range(2):
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')


# 4. 模型评估
def evaluate_model(model, inputs, labels):
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)

        y_true = labels.numpy()
        y_pred = predictions.numpy()

        acc = accuracy_score(y_true, y_pred)
        # 加上 zero_division=0 防止样本极少时指标计算报错
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return acc, precision, recall, f1


acc, precision, recall, f1 = evaluate_model(model, inputs, labels)
print(
    f'Accuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}'
)

# 5. 推理优化提示：
# 注意：直接对 Hugging Face Transformer 模型使用 torch.quantize_dynamic 容易报错。
# 实际生产中建议使用 Hugging Face Optimum 库来进行量化。
print(
    'Note: quantize_dynamic on Hugging Face models may require torch.fx or Hugging Face Optimum.'
)

# 6. 蒸馏（使用 torch.no_grad() 切断教师模型梯度，避免计算图被重复使用报错）
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


optimizer_stu = torch.optim.AdamW(student_model.parameters(), lr=2e-5)
student_model.train()
for epoch in range(2):
    optimizer_stu.zero_grad()
    student_outputs = student_model(**inputs)
    loss = distillation_loss(student_outputs.logits, teacher_logits)
    loss.backward()
    optimizer_stu.step()
    print(f'Epoch {epoch + 1}, Distillation Loss: {loss.item():.4f}')