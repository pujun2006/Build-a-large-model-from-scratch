import os
import time
import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

# 1. 加载预训练的 BERT 模型和 Tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. 加载 IMDb 数据集
dataset = load_dataset("stanfordnlp/imdb")
train_dataset = dataset["train"].map(
    lambda e: tokenizer(e['text'], truncation=True, padding='max_length'),
    batched=True
)
test_dataset = dataset["test"].map(
    lambda e: tokenizer(e['text'], truncation=True, padding='max_length'),
    batched=True
)

# 3. 训练参数设置
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1
)


# 4. 评估指标函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 运行前测试
eval_result = trainer.evaluate()
print(f"Initial Model Accuracy: {eval_result['eval_accuracy']:.2f}")


# 5. 【已适配 transformers 5.17.0】手动裁剪指定层的注意力头
def prune_attention_heads(model, heads_to_prune):
    total_heads = model.config.num_attention_heads

    for layer_idx, heads in heads_to_prune.items():
        if not heads:
            continue
        if layer_idx < 0 or layer_idx >= len(model.bert.encoder.layer):
            raise IndexError(f"层索引 {layer_idx} 超出当前编码器层数范围。")

        layer = model.bert.encoder.layer[layer_idx]
        attn = layer.attention.self
        out_proj = layer.attention.output.dense

        current_q_dim = attn.query.out_features
        if current_q_dim % total_heads != 0:
            raise ValueError(
                f"第 {layer_idx} 层当前 Q 维度 ({current_q_dim}) 无法被 num_attention_heads ({total_heads}) 整除。"
            )
        head_dim = current_q_dim // total_heads

        prune_set = set(heads)
        keep_heads = [h for h in range(total_heads) if h not in prune_set]
        if not keep_heads:
            raise ValueError(f"第 {layer_idx} 层不能移除全部注意力头。")

        keep_indices = []
        for h in keep_heads:
            start = h * head_dim
            keep_indices.extend(range(start, start + head_dim))

        device = attn.query.weight.device
        dtype = attn.query.weight.dtype
        keep_idx_tensor = torch.tensor(keep_indices, dtype=torch.long, device=device)

        # 裁剪 Q, K, V 的输出维度 (dim=0)
        for attr_name in ("query", "key", "value"):
            old_proj = getattr(attn, attr_name)
            new_proj = torch.nn.Linear(
                old_proj.in_features,
                len(keep_indices),
                bias=(old_proj.bias is not None),
                device=device,
                dtype=dtype,
            )
            new_proj.weight.data = old_proj.weight.data.index_select(0, keep_idx_tensor)
            if old_proj.bias is not None:
                new_proj.bias.data = old_proj.bias.data.index_select(0, keep_idx_tensor)
            setattr(attn, attr_name, new_proj)

        # 裁剪 attention.output.dense 的输入维度 (dim=1)，保留完整 bias
        new_out_dense = torch.nn.Linear(
            len(keep_indices),
            out_proj.out_features,
            bias=(out_proj.bias is not None),
            device=device,
            dtype=dtype,
        )
        new_out_dense.weight.data = out_proj.weight.data.index_select(1, keep_idx_tensor)
        if out_proj.bias is not None:
            new_out_dense.bias.data = out_proj.bias.data.clone()
        layer.attention.output.dense = new_out_dense

        # 若模块上存在对应属性，同步更新为该层实际保留的头数与维度
        if hasattr(attn, "num_attention_heads"):
            attn.num_attention_heads = len(keep_heads)
        if hasattr(attn, "all_head_size"):
            attn.all_head_size = len(keep_indices)


# 剪枝掉第3层的第2和第3个注意力头
prune_attention_heads(model, {3: [2, 3]})


# 6. 层剪枝：移除不重要的 Transformer 层，并同步更新 config
def prune_layers(model, prune_layer_nums):
    layers = [layer for i, layer in enumerate(model.bert.encoder.layer) if i not in prune_layer_nums]
    model.bert.encoder.layer = torch.nn.ModuleList(layers)
    # 同步更新模型的隐藏层总数配置
    model.config.num_hidden_layers = len(layers)


# 移除第11层以减少模型深度
prune_layers(model, prune_layer_nums=[11])

# 剪枝后再次进行评估
eval_result_pruned = trainer.evaluate()
print(f"Accuracy after Pruning: {eval_result_pruned['eval_accuracy']:.2f}")


# 7. 模型大小与推理性能评估函数
def evaluate_model_metrics(model, test_dataset, device="cuda"):
    model.to(device)

    # 评估模型物理文件大小
    model_path = "pruned_bert_model.pth"
    torch.save(model.state_dict(), model_path)
    model_size = os.path.getsize(model_path) / 1e6  # MB
    print(f'Model Size after Pruning: {model_size:.2f} MB')

    # 单次前向推理耗时评估（抽样测试）
    model.eval()
    dummy_input = tokenizer("This is a test sentence for inference speed evaluation.", return_tensors="pt").to(device)

    # 预热 (Warm-up)
    with torch.no_grad():
        for _ in range(10):
            _ = model(**dummy_input)

    # 正式计时
    start_time = time.time()
    num_iters = 50
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(**dummy_input)

    avg_inference_time = (time.time() - start_time) / num_iters * 1000  # 毫秒 (ms)
    print(f'Average Inference Time per forward pass: {avg_inference_time:.2f} ms')


# 执行评估
device = "cuda" if torch.cuda.is_available() else "cpu"
evaluate_model_metrics(model, test_dataset, device=device)