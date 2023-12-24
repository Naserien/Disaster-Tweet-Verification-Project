import pandas as pd
import pickle as pkl
from tqdm import tqdm
import os
import nltk

# 进度条初始化
tqdm.pandas()

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_english_vocab(sentences):
    # 下载英文停用词
    nltk.download('stopwords')
    english_stopwords = set(nltk.corpus.stopwords.words('english'))

    # 去除特殊字符
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }

    sentences = sentences.apply(lambda x: clean_special_chars(x, punct, punct_mapping))

    # 提取数组
    sentences = sentences.progress_apply(lambda x: x.split()).values

    # 去除停用词和特殊字符
    sentences = [
        [word for word in sentence if word.lower() not in english_stopwords and word not in {'#', '####', '1', '0'}] for
        sentence in sentences]

    vocab_dic = {}
    for sentence in tqdm(sentences, disable=False):
        for word in sentence:
            try:
                vocab_dic[word] += 1
            except KeyError:
                vocab_dic[word] = 1

    vocab_list = sorted([_ for _ in vocab_dic.items()], key=lambda x: x[1], reverse=True)[:MAX_VOCAB_SIZE]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

    return vocab_dic


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    return text


def build_dataset(config):
    df = pd.read_csv(config.train_path, encoding='utf-8', sep=';', header=None)
    sentences = df.iloc[:, 0].apply(lambda x: x.lower())
    vocab = build_english_vocab(sentences)
    pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"词典======== {vocab}")


if __name__ == "__main__":
    class Config():
        def __init__(self):
            self.vocab_path = './english_vocab.pkl'  # 保存英文词典的路径
            self.train_path = './data/preprocessed_data.txt'


    build_dataset(Config())
