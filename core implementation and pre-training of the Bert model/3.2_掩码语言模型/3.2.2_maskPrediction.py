from transformers import BertTokenizer, BertForMaskedLM
import torch

# 初始化BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# 定义带有掩码的输入句子
text = "In the context of global economic growth, the demand for technology companies' [MASK] is rapidly increasing."

# 将输入句子编码为ID序列，并识别掩码位置
input_ids = tokenizer.encode(text, return_tensors="pt")
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# 前向传播预测掩码位置的词汇分布
with torch.no_grad():
    output = model(input_ids)
    logits = output.logits

# 获取掩码位置的预测结果，并取概率最高的词
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
predicted_word = tokenizer.decode(predicted_token_id)

# 将预测词插入句子中，生成完整标题
final_sentence = text.replace("[MASK]", predicted_word)
print("生成的标题:", final_sentence)
