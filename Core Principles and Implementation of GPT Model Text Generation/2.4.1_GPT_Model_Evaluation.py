import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 1. 文本定义
generated_text = "The quick brown fox jumps over the lazy dog"
reference_text = "The quick brown fox jumps over the lazy dog"


# 2. 计算 BLEU
def calculate_bleu(generated, reference):
    # sentence_bleu 接收分词后的列表
    gen_tokens = generated.split()
    ref_tokens = [reference.split()]  # 可以有多个参考句
    return sentence_bleu(ref_tokens, gen_tokens, smoothing_function=SmoothingFunction().method1)


# 3. 计算 ROUGE
def calculate_rouge(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # rouge-scorer 直接接收原始字符串
    return scorer.score(reference, generated)


# 输出 BLEU 与 ROUGE
print("BLEU分数:", calculate_bleu(generated_text, reference_text))
rouge_scores = calculate_rouge(generated_text, reference_text)
print("ROUGE分数:")
for key, score in rouge_scores.items():
    print(f"  {key}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1={score.fmeasure:.4f}")


# 4. 正确的 GPT 困惑度计算类
class GPTPerplexityCalculator(nn.Module):
    def __init__(self, model):
        super(GPTPerplexityCalculator, self).__init__()
        self.model = model

    def forward(self, input_ids):
        # input_ids shape: [batch_size, seq_len]
        outputs = self.model(input_ids)  # [batch_size, seq_len, vocab_size]

        # 1. 错位对齐 (Shift)
        # Shift logits: 预测第 1 ~ seq_len 个位置，所以舍弃最后一个位置的预测值
        shift_logits = outputs[:, :-1, :].contiguous()
        # Shift labels: 目标是第 1 ~ seq_len 个位置的真实 Token，所以舍弃第 0 个位置
        shift_labels = input_ids[:, 1:].contiguous()

        # 2. 展平进行 CrossEntropy 计算
        # [batch_size * (seq_len - 1), vocab_size] vs [batch_size * (seq_len - 1)]
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # 3. 困惑度 PPL = exp(loss)
        perplexity = torch.exp(loss)
        return perplexity


# 5. 模拟模型与数据验证
class SimpleGPTModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SimpleGPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.fc(x)
        return x


vocab_size = 1000
embed_size = 256
model = SimpleGPTModel(vocab_size, embed_size)

# 支持多样本 batch 验证 (例如 batch_size=2, seq_len=10)
input_data = torch.randint(0, vocab_size, (2, 10))

perplexity_calculator = GPTPerplexityCalculator(model)
perplexity = perplexity_calculator(input_data)
print("模型困惑度 (PPL):", perplexity.item())


import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 参考文本和生成文本
reference_text = "近年来，随着人工智能技术的不断发展，深度学习和自然语言处理逐渐成为热门的研究领域，特别是在文本生成方面，GPT（Generative Pre-trained Transformer）模型表现出色，能够生成具有连贯性和逻辑性的长文本。"
generated_text = "随着人工智能的发展，深度学习和自然语言处理成为研究的热点。GPT模型在文本生成任务中表现优异，能够生成连贯的长文本。"


# ----------------------------------------------------
# 1. 修正后的 BLEU 分数计算（增加中文分词处理）
# ----------------------------------------------------
def calculate_bleu(generated, reference):
    # 使用 jieba 进行中文分词
    ref_tokens = list(jieba.cut(reference))
    gen_tokens = list(jieba.cut(generated))

    # BLEU 要求 reference 是一个候选列表的列表
    bleu_score = sentence_bleu(
        [ref_tokens], gen_tokens,
        smoothing_function=SmoothingFunction().method1
    )
    return bleu_score


# ----------------------------------------------------
# 2. 修正后的 ROUGE 分数计算（按词/字添加空格拼接）
# ----------------------------------------------------
def calculate_rouge(generated, reference):
    # rouge-score 默认用空格切词，因此我们需要手动加空格拼接中文词/字
    ref_spaced = " ".join(jieba.cut(reference))
    gen_spaced = " ".join(jieba.cut(generated))

    # 中文计算时设置 use_stemmer=False
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(ref_spaced, gen_spaced)
    return scores


print("--- 1. 计算文本重合度指标 ---")
bleu_score = calculate_bleu(generated_text, reference_text)
print(f"BLEU 分数: {bleu_score:.4f}")

rouge_scores = calculate_rouge(generated_text, reference_text)
print("ROUGE 分数:")
for key, score in rouge_scores.items():
    print(f"  {key}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1={score.fmeasure:.4f}")


# ----------------------------------------------------
# 3. 修正后的困惑度（Perplexity）计算类
# ----------------------------------------------------
class SimpleGPTModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SimpleGPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.fc(x)
        return x


class GPTPerplexityCalculator(nn.Module):
    def __init__(self, model, vocab_size):
        super(GPTPerplexityCalculator, self).__init__()
        self.model = model
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        # input_ids 形状: [Batch, Seq_Len]
        outputs = self.model(input_ids)  # 形状: [Batch, Seq_Len, Vocab_Size]

        # Shift 机制对齐：用前 T-1 个预测后 T-1 个 Token
        shift_logits = outputs[:, :-1, :].contiguous().view(-1, self.vocab_size)
        shift_labels = input_ids[:, 1:].contiguous().view(-1)

        # 计算交叉熵损失
        loss = F.cross_entropy(shift_logits, shift_labels)
        perplexity = torch.exp(loss)
        return perplexity


print("\n--- 2. 计算模型困惑度 (PPL) ---")
vocab_size = 3000
embed_size = 512
model = SimpleGPTModel(vocab_size, embed_size)
input_data = torch.randint(0, vocab_size, (1, 100))

perplexity_calculator = GPTPerplexityCalculator(model, vocab_size)
perplexity = perplexity_calculator(input_data)
print("模型困惑度 (Perplexity):", round(perplexity.item(), 2))