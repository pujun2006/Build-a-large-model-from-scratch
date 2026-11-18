from transformers import BertTokenizer, GPT2Tokenizer

# 初始化分词器
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 【最佳实践】为没有默认 pad_token 的 GPT-2 设置 pad_token
if gpt_tokenizer.pad_token is None:
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

text = "Artificial Intelligence is transforming the world."

# 编码测试
bert_encoded = bert_tokenizer(text, padding="max_length", max_length=12, truncation=True, return_tensors="pt")
gpt_encoded = gpt_tokenizer(text, padding="max_length", max_length=12, truncation=True, return_tensors="pt")

print("BERT 编码:", bert_encoded['input_ids'])
print("GPT2 编码:", gpt_encoded['input_ids'])