# 导入所需库
from transformers import BertTokenizer, GPT2Tokenizer

def main():
    print("=== 1. 初始化分词器 ===")
    # 加载 BERT 和 GPT-2 分词器
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 【工程最佳实践】GPT-2 原生没有定义 pad_token，需显式将其设置为 eos_token，
    # 避免在 padding 时触发警告或产生不规范的填充 ID。
    if gpt_tokenizer.pad_token is None:
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    # 定义示例句子集：情感分析任务中的句子
    sentences = [
        "I love this movie! It's absolutely amazing.",
        "The plot was terrible and the acting was worse.",
        "An enjoyable experience with outstanding visuals.",
        "I wouldn't recommend it to anyone.",
        "A brilliant and heartwarming story that everyone should watch."
    ]

    print("\n=== 2. 批量处理与编码对比 ===")
    # 对每个句子使用 BERT 和 GPT-2 分词器进行分词和编码
    for i, sentence in enumerate(sentences):
        print(f"\n--------------------------------------------------")
        print(f"句子 {i+1}: {sentence}")

        # ==================== BERT 处理部分 ====================
        bert_tokens = bert_tokenizer.tokenize(sentence)
        bert_encoded = bert_tokenizer(
            sentence,
            padding="max_length",
            max_length=12,
            truncation=True,
            return_tensors="pt"
        )
        print("\n[BERT]")
        print("  - 分词结果 (Tokens):", bert_tokens)
        print("  - 编码后的 ID (input_ids):", bert_encoded['input_ids'].tolist())
        print("  - 注意力掩码 (Attention Mask):", bert_encoded['attention_mask'].tolist())

        # ==================== GPT-2 处理部分 ====================
        gpt_tokens = gpt_tokenizer.tokenize(sentence)
        gpt_encoded = gpt_tokenizer(
            sentence,
            padding="max_length",
            max_length=12,
            truncation=True,
            return_tensors="pt"
        )
        print("\n[GPT-2]")
        print("  - 分词结果 (Tokens):", gpt_tokens)
        print("  - 编码后的 ID (input_ids):", gpt_encoded['input_ids'].tolist())
        print("  - 注意力掩码 (Attention Mask):", gpt_encoded['attention_mask'].tolist())

if __name__ == "__main__":
    main()
