import re
import pandas as pd
from collections import Counter, defaultdict

# 示例数据
data = {
    'text': [
        "自然语言处理是人工智能的一个重要领域。",
        "数据增强技术可以扩展数据集，提高模型的泛化能力。",
        "深度学习和机器学习是现代人工智能的核心技术。",
    ]
}
df = pd.DataFrame(data)

# 【修改处】定义分词函数：保持每个文本的空格分隔字符序列作为一个整体存储
def get_vocab(texts):
    vocab = {}
    for text in texts:
        # 将文本转换为以空格分隔的字符序列作为字典的键，初始频次为 1
        word = ' '.join(list(text))
        vocab[word] = vocab.get(word, 0) + 1
    return vocab

# 获取最频繁的字对
def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

# BPE分词合并
def merge_vocab(vocab, pair):
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        w_out = p.sub(''.join(pair), word)
        new_vocab[w_out] = vocab[word]
    return new_vocab

# BPE分词主函数
def bpe_tokenize(texts, num_merges):
    vocab = get_vocab(texts)
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(vocab, best)
    return vocab

# 使用BPE进行分词
bpe_vocab = bpe_tokenize(df['text'], num_merges=10)

# 输出结果
print("分词结果：")
for word in bpe_vocab:
    print(f"{word}: {bpe_vocab[word]}")
