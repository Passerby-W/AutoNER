import csv
import os
import pickle
import config
from vocabulary import Vocabulary
import json

"""
if file suffix in ['.txt', '.tsv', '.bio', '.bioes'], every line should follow the format below:
word word word\tlabel label label

if file suffix is '.jsonl', every line should contain 2 keys('data', 'label'), 
data contains a sentence, label contains tuples(begin index, end index + 1, entity name)
example likes below:
{"data": "Eat 8 grams of aspirin every day.", "label": [[1, 3, "Dose"], [4, 5, "Medicine"]]}
"""


def transfer_jsonl(in_path, bioes=True):
    words, tags = [], []
    with open(in_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = json.loads(line.strip())
            if bioes:
                word_list, tag_list = jsonl_to_bioes(line)
            else:
                word_list, tag_list = jsonl_to_bio(line)
            words.append(word_list)
            tags.append(tag_list)
    return words, tags


def transfer_txt(in_path, bioes=True):
    words, tags = [], []
    with open(in_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            word, tag = line.split('\t')
            word_list = word.split()
            tag_list = tag.split() if bioes else bioes_to_bio(tag.split())
            words.append(word_list)
            tags.append(tag_list)
    return words, tags


def jsonl_to_bioes(record):
    text = record['data']
    labels = record['label']

    # Convert text to a word list
    words = list(text)

    # Initialize label list to 'O'
    tags = ['O'] * len(words)

    # Traverse the annotation information and mark the corresponding words as BIOES labels
    for label in labels:
        start, end, label_type = label
        if end - start == 1:
            tags[start] = 'S-' + label_type
        else:
            tags[start] = 'B-' + label_type
            tags[start + 1:end - 1] = ['I-' + label_type] * (end - start - 2)
            tags[end - 1] = 'E-' + label_type

    # 返回BIOES标签列表
    return words, tags


def jsonl_to_bio(record):
    text = record['data']
    labels = record['label']

    # Convert text to a word list
    words = list(text)

    # Initialize label list to 'O'
    tags = ['O'] * len(words)

    # Traverse the annotation information and mark the corresponding words as BIOES labels
    for label in labels:
        start, end, label_type = label
        tags[start] = 'B-' + label_type
        tags[start + 1:end] = ['I-' + label_type] * (end - start - 1)

    # 返回BIO标签列表
    return words, tags


def bioes_to_bio(bioes):
    # Initialize a new label sequence
    bio = bioes
    for i, tag in enumerate(bioes):
        if tag.startswith("E-"):
            # Convert the "E -" label to the "I -" label
            bio[i] = tag.replace("E-", "I-")
        elif tag.startswith("S-"):
            # Convert the "S -" label to the "B -" label
            bio[i] = tag.replace("S-", "B-")
    return bio


def split_zh_en(text):
    # 分割字符串并在中英文之间添加空格
    words = []
    temp_word = ''

    for char in text:
        # 如果是中文字符，直接添加到words列表中
        if '\u4e00' <= char <= '\u9fa5':
            if temp_word:
                words.extend(temp_word.split())
                temp_word = ''
            words.append(char)
        # 如果是英文字符或数字，拼接到temp_word字符串中
        elif char.isalnum():
            temp_word += char
        # 如果是其他字符，将temp_word添加到words列表中，并清空temp_word
        else:
            if temp_word:
                words.extend(temp_word)
                temp_word = ''
            words.append(char)

    # 如果temp_word还有剩余，添加到words列表中
    if temp_word:
        words.extend(temp_word)

    # 输出分割结果
    return words


class PrePrecessing:
    def __init__(self):
        self.vocab = None
        self.current_path = os.getcwd()
        self.train_data_path = config.train_file_path
        self.dev_data_path = config.dev_file_path
        self.bioes = config.bioes_format
        self.save_path = config.save_path
        self.tags = config.tags
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.max_vocab_size = config.max_vocab_size
        self.min_word_freq = config.min_word_freq

    def get_train_dataset(self):
        sentences = []
        if self.train_data_path.endswith('.jsonl'):
            words, tags = transfer_jsonl(self.train_data_path, self.bioes)
            for row in words:
                sentences.append(''.join(row))
            self.vocab = Vocabulary(sentences, max_vocab_size=self.max_vocab_size, min_word_freq=self.min_word_freq)
            with open(self.save_path + '/vocab.pkl', 'wb') as f:
                pickle.dump(self.vocab, f)
        else:
            words, tags = transfer_txt(self.train_data_path, self.bioes)
            for row in words:
                sentences.append(''.join(row))
            self.vocab = Vocabulary(sentences, max_vocab_size=self.max_vocab_size, min_word_freq=self.min_word_freq)
            with open(self.save_path + '/vocab.pkl', 'wb') as f:
                pickle.dump(self.vocab, f)
        dataset = []
        for i, j in zip(words, tags):
            dataset.append((i, j))
        return dataset

    def get_dev_dataset(self):
        if self.dev_data_path:
            if self.dev_data_path.endswith('.jsonl'):
                words, tags = transfer_jsonl(self.dev_data_path, self.bioes)
            else:
                words, tags = transfer_txt(self.dev_data_path, self.bioes)
            dataset = []
            for i, j in zip(words, tags):
                dataset.append((i, j))
            return dataset
        else:
            return ''

    def get_word_to_ix(self):
        return self.vocab.word_to_index

    def get_tag_to_ix(self):
        tag_to_ix = {'O': 0, "<START>": 1, "<STOP>": 2}
        for tag in self.tags:
            if self.bioes:
                tag_to_ix['B-' + tag] = len(tag_to_ix)
                tag_to_ix['I-' + tag] = len(tag_to_ix)
                tag_to_ix['E-' + tag] = len(tag_to_ix)
                tag_to_ix['S-' + tag] = len(tag_to_ix)
            else:
                tag_to_ix['B-' + tag] = len(tag_to_ix)
                tag_to_ix['I-' + tag] = len(tag_to_ix)
        return tag_to_ix

    def get_ix_to_tag(self):
        tag_to_ix = self.get_tag_to_ix()
        return {v: k for k, v in tag_to_ix.items()}


if __name__ == '__main__':
    p = PrePrecessing()
    train = p.get_train_dataset()
    for data in train:
        print(data)
    dev = p.get_dev_dataset()
    for data in dev:
        print(data)
