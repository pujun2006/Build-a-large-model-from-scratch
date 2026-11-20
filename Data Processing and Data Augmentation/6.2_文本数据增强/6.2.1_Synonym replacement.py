import random
import jieba
import pandas as pd

# 示例数据集
data = {
    'text': [
        '自然语言处理是人工智能的一个重要领域，它涉及人机交互中的语言理解和生成。',
        '机器学习和深度学习的进步推动了自然语言处理的发展。',
        '同义词替换是一种数据增强技术，能在保持语义不变的情况下扩展数据集。',
    ]
}
df = pd.DataFrame(data)

# 简单的中文同义词典示例（实际项目中可使用更完整的“同义词词林”或词向量模型）
synonym_dict = {
    '重要': ['关键', '核心', '主要'],
    '发展': ['前进', '进展', '演进'],
    '保持': ['维持', '维系', '坚持'],
    '扩展': ['扩充', '扩大', '拓宽'],
    '领域': ['范畴', '板块', '界别'],
}


# 中文同义词替换函数
def synonym_replacement_cn(sentence, n=1):
  # 使用 jieba 进行分词
  words = list(jieba.cut(sentence))
  new_words = words[:]

  # 筛选出词典中存在的词的索引
  candidate_indices = [i for i, word in enumerate(words) if word in synonym_dict]

  replaced = 0
  # 循环条件增加 candidate_indices 非空判断，防止找不到可替换词时陷入死循环
  while replaced < n and candidate_indices:
    idx = random.choice(candidate_indices)
    candidate_indices.remove(idx)  # 移除已选过的索引，避免重复选中

    word = words[idx]
    synonyms = synonym_dict[word]

    if synonyms:
      new_words[idx] = random.choice(synonyms)
      replaced += 1

  # 将分词拼接回完整句子
  return ''.join(new_words)


# 定义增强函数，对数据集每一行进行同义词替换
def augment_text(df, num_replacements=1):
  augmented_texts = []

  for text in df['text']:
    augmented_text = synonym_replacement_cn(text, num_replacements)
    augmented_texts.append(augmented_text)

  df['augmented_text'] = augmented_texts
  return df


# 增强数据集，指定每条句子替换1个词
df = augment_text(df, num_replacements=1)

# 显示增强结果
print('原始文本：')
for i, t in enumerate(df['text']):
  print(f'[{i}] {t}')

print('\n增强后的文本：')
for i, t in enumerate(df['augmented_text']):
  print(f'[{i}] {t}')
