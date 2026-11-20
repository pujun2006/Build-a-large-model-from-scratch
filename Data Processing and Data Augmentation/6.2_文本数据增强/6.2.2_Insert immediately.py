import random
import jieba
import jieba.posseg as pseg  # 引入词性标注模块
import pandas as pd

# 示例数据集
data = {
    'text': [
        '自然语言处理是人工智能的一个重要领域，它涉及人机交互中的语言理解和生成。',
        '机器学习和深度学习的进步推动了自然语言处理的发展。',
        '数据增强技术可以通过多种方式扩展文本数据，以提升模型的泛化能力。',
    ]
}
df = pd.DataFrame(data)

# 中文同义词典示例
synonym_dict = {
    '重要': ['关键', '核心'],
    '发展': ['前进', '进展'],
    '提升': ['提高', '增强'],
    '扩展': ['扩充', '扩大'],
    '技术': ['方法', '手段'],
}


# 优化的中文随机插入函数（基于词性过滤，减少严重语病）
def random_insertion_cn(sentence, n=1):
  # 使用 jieba 进行带词性的分词
  words_with_flag = list(pseg.cut(sentence))
  words = [word for word, flag in words_with_flag]
  new_words = words[:]

  inserted = 0
  attempts = 0  # 防止死循环的安全计数器

  while inserted < n and attempts < 20:
    attempts += 1
    # 随机选择一个词
    word_idx = random.randint(0, len(words) - 1)
    word = words[word_idx]

    # 检查该词是否有同义词
    if word in synonym_dict:
      synonym = random.choice(synonym_dict[word])

      # 随机选择一个合法的插入位置（避免插在句首或句尾太奇怪的地方，这里简化为全随机）
      insert_pos = random.randint(0, len(new_words))
      new_words.insert(insert_pos, synonym)
      inserted += 1

  return ''.join(new_words)


# 定义随机插入增强函数
def augment_text_with_insertion(df, num_insertions=1):
  augmented_texts = []
  for text in df['text']:
    augmented_text = random_insertion_cn(text, num_insertions)
    augmented_texts.append(augmented_text)

  df['augmented_text'] = augmented_texts
  return df


# 运行随机插入增强（每条语句尝试插入1个同义词）
df = augment_text_with_insertion(df, num_insertions=1)

# 显示结果
print('原始文本：')
for i, t in enumerate(df['text']):
  print(f'[{i}] {t}')

print('\n随机插入增强后的文本：')
for i, t in enumerate(df['augmented_text']):
  print(f'[{i}] {t}')
