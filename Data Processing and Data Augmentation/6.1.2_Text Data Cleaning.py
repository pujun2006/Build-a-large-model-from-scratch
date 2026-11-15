import re
import pandas as pd
from zhon.hanzi import punctuation as zh_punctuation

# 示例数据集
data = {
    'text': [
        "这是一段包含HTML标签的文本。<html>标签内容</html>此外，文本中还有特殊字符：#、@、$以及一些无用信息。",
        "这里是包含重复词汇的示例。比如，比如，这里有一些重复的词汇需要去除。",
        "这是一段含有停用词的文本。停用词包括：和、是、的、了、不等。",
        "  此文本包含了前后空格以及多余的空白     间隔。   ",
        "此文本中包含英文标点符号，例如: ! @ # $, 需要将这些符号进行处理。"
    ]
}
df = pd.DataFrame(data)

# 中文停用词表
stop_words = set(["和", "是", "的", "了", "不", "等"])


# 定义数据清洗函数
def clean_text(text):
    # 1. 去除HTML标签
    text = re.sub(r'<.*?>', '', text)

    # 2. 去除特殊字符（保留汉字、字母、数字和空格）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)

    # 3. 去除中文标点符号
    text = re.sub(f'[{zh_punctuation}]', '', text)

    # 4. 去除前后空格
    text = text.strip()

    # 5. 删除多余空格
    text = re.sub(r'\s+', ' ', text)

    # 6. 去除停用词
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # 7. 去除重复词汇
    words = text.split()
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    text = ' '.join(unique_words)

    return text


# 应用清洗函数
df['cleaned_text'] = df['text'].apply(clean_text)

# 显示清洗结果
print("清洗前的文本：")
print(df['text'])
print("\n清洗后的文本：")
print(df['cleaned_text'])
