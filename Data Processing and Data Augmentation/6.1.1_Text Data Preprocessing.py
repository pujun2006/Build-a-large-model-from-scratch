import re
import pandas as pd

# 示例数据集
data = {
    'text': [
        "  Hello world! This is an example text with numbers 123 and symbols #, @, and so on.   ",
        "Here's another sentence: it contains HTML tags like <html> and & symbols.",
        "Numbers, punctuations...and STOP words are in this sentence!"
    ]
}
df = pd.DataFrame(data)


# 定义文本数据预处理函数
def preprocess_text(text):
    # 1. 去除前后空格
    text = text.strip()

    # 2. 转换为小写
    text = text.lower()

    # 3. 去除HTML标签
    text = re.sub(r'<.*?>', '', text)

    # 4. 去除数字
    text = re.sub(r'\d+', '', text)

    # 5. 去除特殊字符（保留字母和空格）
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 6. 去除多余空格
    text = re.sub(r'\s+', ' ', text)

    return text


# 应用预处理函数
df['processed_text'] = df['text'].apply(preprocess_text)

# 显示预处理结果
print("预处理前的英文文本：")
print(df['text'])
print("\n预处理后的英文文本：")
print(df['processed_text'])

# 示例中文长文本
data = {
    'text': [
        """
        在自然语言处理领域，数据预处理是不可或缺的重要步骤。对于中文文本数据来说，预处理显得尤为重要，因为中文文本中可能包含各种特殊字符和标点符号，
        以及不同格式的编码，比如全角字符和半角字符等。在实际应用中，通常需要对原始文本进行预处理，以提升模型的学习效果。
        本文将介绍一种常见的中文文本预处理流程，涵盖去除标点符号、规范化空格、统一大小写、处理特殊字符等操作。
        例如，"你好，世界！"这句话包含了中文标点符号和空格，若直接输入模型，可能会影响模型的训练效果。
        本示例将一步步展示如何将原始文本转化为格式统一的输入，确保文本具备高质量的输入特性。

        首先，去除文本中的前后空格，确保输入文本的规范性。接下来，将所有字符转换为小写，以消除大小写的干扰。其次，
        使用正则表达式去除文本中的标点符号，例如句号、逗号、引号等，保留文字的核心信息，确保模型不受到噪声的干扰。此外，
        去除不必要的空格，确保每个词之间只存在一个空格，提升文本的整洁度。
        最后，文本中可能还包含某些特殊字符或乱码，需要进行进一步清理。经过这些步骤，文本数据将更加清晰，有助于提升自然语言处理模型的表现。
        例如，“这个文本中包含数字12345678和标点符号#，&，%等”，需要将这些无关的符号进行处理。
        通过规范化处理，文本将具备更好的输入效果。

        在实际应用中，经过预处理后的文本将能够更加准确地被模型理解，同时也能有效提升模型的训练效率。
        本次示例将展示具体的文本清洗步骤和代码实现，帮助理解文本预处理的重要性和具体操作。
        """
    ]
}
df = pd.DataFrame(data)


# 定义中文文本预处理函数
def preprocess_chinese_text(text):
    # 1. 去除前后空格
    text = text.strip()

    # 2. 转换为小写
    text = text.lower()

    # 3. 去除标点符号和特殊字符（保留汉字、字母和数字）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)

    # 4. 处理全角空格转换为半角空格
    text = re.sub(r'\u3000', ' ', text)

    # 5. 去除多余空格
    text = re.sub(r'\s+', ' ', text)

    return text


# 应用预处理函数
df['processed_text'] = df['text'].apply(preprocess_chinese_text)

# 显示预处理结果
print("预处理前的中文文本：")
print(df['text'][0])
print("\n预处理后的中文文本：")
print(df['processed_text'][0])
