import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 下载停用词
import nltk

nltk.download('stopwords')

# 初始化词形还原器和停用词
lemma = WordNetLemmatizer()
stopwords_en = set(stopwords.words("english"))


# 函数：移除表情符号
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# 读取CSV文件
df = pd.read_csv('train.csv')


# 函数：预处理
def preprocess(tweet):
    sentence = ''
    words = [word.lower() for word in tweet.split() if 'http' not in word]  # 移除 URL
    for word in words:
        # 移除嵌入式特殊字符（例如 #earthquake）
        word = re.sub(r'[^\w\s]', '', word)
        # 移除数字
        word = re.sub(r'\d+', '', word)
        # 移除停用词和标点符号
        if word not in stopwords_en and word.isalnum():
            # 词形还原
            word = lemma.lemmatize(word, pos='v')
            # 移除表情符号
            word = remove_emoji(word)
            # 构建处理后的句子
            sentence = sentence + word + ' '
    return sentence.strip()


# 应用预处理函数
df['text'] = df['text'].apply(lambda s: preprocess(s))

# 保存结果到txt文件
with open('data/preprocessed_data.txt', 'w', encoding='utf-8') as file:
    for index, row in df.iterrows():
        file.write(f"{row['target']}\t####\t{row['text']}\n")
