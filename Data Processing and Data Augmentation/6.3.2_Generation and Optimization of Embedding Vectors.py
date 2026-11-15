import pandas as pd
import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端
import matplotlib.pyplot as plt
import jieba

# 指定字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 示例数据集
data = {
    'text': [
        "自然语言处理是人工智能的重要分支。",
        "数据增强可以提高模型的泛化能力。",
        "深度学习和机器学习是现代人工智能的核心。",
        "词向量能表示词汇的语义信息。",
    ]
}
df = pd.DataFrame(data)

# 分词处理
def preprocess_text(text):
    words = jieba.lcut(text)
    return [w for w in words if w.strip() and w not in ['。', '，', '、']]

df['tokenized'] = df['text'].apply(preprocess_text)

# 训练 Word2Vec
model = Word2Vec(sentences=df['tokenized'], vector_size=10, window=3, min_count=1, workers=1)

# 提取词向量并降维
words = list(model.wv.index_to_key)
word_vectors = np.array([model.wv[word] for word in words])

n_comp = min(2, len(words), len(word_vectors[0]))
pca = PCA(n_components=n_comp)
word_vectors_2d = pca.fit_transform(word_vectors)

# 绘制图表
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    x = word_vectors_2d[i, 0]
    y = word_vectors_2d[i, 1] if n_comp > 1 else 0
    plt.scatter(x, y)
    plt.annotate(word, (x, y), fontsize=12)

plt.title("2D PCA后的词嵌入可视化")
plt.xlabel("PCA 元素 1")
plt.ylabel("PCA 元素 2")
plt.grid(True)

# 【修改处】将 plt.show() 改为保存到当前脚本所在的目录下
plt.savefig('embedding_plot.png', dpi=300, bbox_inches='tight')
print("图表已成功保存为 'embedding_plot.png'，请在项目文件夹中查看！")