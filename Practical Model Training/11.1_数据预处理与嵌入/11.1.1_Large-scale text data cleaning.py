import re

text_data = [
    "Hello, world! This is an example text with <html> tags.",
    "Python is great!!! $$$",
    "Hello, world!    ",
    "Machine learning is fascinating! 😊",
    "Deep learning <b>models</b> are powerful."
]


def clean_text(text, remove_punctuation=False):
    # 1. 去除HTML标签
    text = re.sub(r'<.*?>', '', text)

    if remove_punctuation:
        # 如果需要彻底清洗掉所有标点和特殊符号
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    else:
        # 如果需要保留基本标点（如.,!?-），只清除无用的特殊符号/表情符号
        text = re.sub(r'[^\w\s.,!?]', '', text)

    # 2. 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# 步骤 1：先进行基础清洗（保留标点，以便后续分句）
semi_cleaned_data = [clean_text(text, remove_punctuation=False) for text in text_data]

# 步骤 2：去重（保留原有顺序）
unique_data = list(dict.fromkeys(semi_cleaned_data))

print("Data after removing duplicates:")
for text in unique_data:
    print(text)
print("-" * 40)


# 步骤 3：按句号或标点切分文本
def split_sentences(text):
    # 使用正则表达式按 . ! ? 切分句子，并保留标点或直接切分
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


split_data = [split_sentences(text) for text in unique_data]

print("\nData after splitting:")
for idx, sentences in enumerate(split_data):
    print(f"Text {idx + 1}:")
    for sentence in sentences:
        print(f" - {sentence}")
    print()


# 步骤 4：标注规则
def label_text(text):
    text_lower = text.lower()
    if "machine learning" in text_lower:
        return "Technology"
    elif "python" in text_lower:
        return "Programming"
    elif "world" in text_lower:
        return "General"
    else:
        return "Other"


labeled_data = [{"text": text, "label": label_text(text)} for text in unique_data]

print("\nLabeled Data:")
for item in labeled_data:
    print(f"Text: {item['text']}, Label: {item['label']}")
