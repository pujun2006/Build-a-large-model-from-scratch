import nltk
import random
import re
from nltk.corpus import wordnet as wn

# 下载所需资源（确保首次运行成功）
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 示例文本：莎士比亚《哈姆雷特》片段
text_data = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;"""

# 1. 分词与数据预处理
def preprocess_text(text):
    words = re.findall(r'\b\w+\b', text.lower())  # 将文本分割成单词
    return words

tokenized_text = preprocess_text(text_data)
print("分词结果前20个:", tokenized_text[:20])

# 获取同义词辅助函数（处理下划线）
def get_synonyms(word):
    synonyms = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')  # 将 WordNet 中的下划线转为空格
            if name != word and ' ' not in name:  # 这里为了简便，暂时只取单词同义词
                synonyms.append(name)
    return list(set(synonyms))

# 2. 同义词替换 (Synonym Replacement, SR)
def synonym_replacement(words, n=5):
    new_words = words[:]
    # 避免 words 长度小于 n
    n = min(n, len(new_words))
    for _ in range(n):
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        synonyms = get_synonyms(word)
        if synonyms:
            new_words[idx] = random.choice(synonyms)
    return new_words

augmented_text_synonym = synonym_replacement(tokenized_text, n=5)
print("同义词替换结果:", " ".join(augmented_text_synonym[:30]))

# 3. 随机插入 (Random Insertion, RI)
def random_insertion(words, n=5):
    new_words = words[:]
    for _ in range(n):
        if not new_words:
            break
        word = random.choice(new_words)
        synonyms = get_synonyms(word)
        if synonyms:
            insert_pos = random.randint(0, len(new_words))
            new_words.insert(insert_pos, random.choice(synonyms))
    return new_words

augmented_text_insertion = random_insertion(tokenized_text, n=5)
print("随机插入结果:", " ".join(augmented_text_insertion[:30]))

# 4. 随机删除 (Random Deletion, RD)
def random_deletion(words, p=0.2):
    if len(words) <= 1:
        return words
    # 保证至少保留一个词，防止返回空列表
    new_words = [word for word in words if random.uniform(0, 1) > p]
    if len(new_words) == 0:
        return [random.choice(words)]
    return new_words

augmented_text_deletion = random_deletion(tokenized_text, p=0.2)
print("随机删除结果:", " ".join(augmented_text_deletion[:30]))