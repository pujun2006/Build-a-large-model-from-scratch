import random
import jieba
import pandas as pd

# 示例数据集
data = {
    'text': [
        '自然语言处理是人工智能的一个重要领域，它涉及人机交互中的语言理解和生成。',
        '机器学习和深度学习的进步推动了自然语言处理的发展。',
        '数据增强技术通过扩展文本数据集，提升模型的泛化能力。',
    ]
}
df = pd.DataFrame(data)


# 1. 修复后的中文随机删除函数
def random_deletion_cn(sentence, p=0.2):
  words = list(jieba.cut(sentence))
  if len(words) <= 1:
    return sentence

  new_words = []
  for word in words:
    # 随机决定是否保留该词
    if random.uniform(0, 1) > p:
      new_words.append(word)

  # 若所有词都被删光了，随机保留原分词列表中的一个词
  if len(new_words) == 0:
    return random.choice(words)

  return ''.join(new_words)


# 2. 修复后的中文随机顺序交换函数
def random_swap_cn(sentence, n=1):
  words = list(jieba.cut(sentence))
  if len(words) < 2:
    return sentence  # 词数少于2个无法交换

  new_words = words[:]
  for _ in range(n):
    idx1, idx2 = random.sample(range(len(new_words)), 2)
    # 交换两个随机位置的词语
    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

  return ''.join(new_words)


# 定义数据增强函数
def augment_text_with_delete_swap(df, delete_prob=0.2, swap_times=1):
  augmented_texts_delete = []
  augmented_texts_swap = []

  for text in df['text']:
    # 进行随机删除
    augmented_text_delete = random_deletion_cn(text, delete_prob)
    augmented_texts_delete.append(augmented_text_delete)

    # 进行随机顺序交换
    augmented_text_swap = random_swap_cn(text, swap_times)
    augmented_texts_swap.append(augmented_text_swap)

  df['augmented_text_delete'] = augmented_texts_delete
  df['augmented_text_swap'] = augmented_texts_swap
  return df


# 增强数据集
df = augment_text_with_delete_swap(df, delete_prob=0.2, swap_times=1)

# 显示增强结果
print('原始文本：')
print(df['text'])
print('\n随机删除后的文本：')
print(df['augmented_text_delete'])
print('\n随机顺序交换后的文本：')
print(df['augmented_text_swap'])
