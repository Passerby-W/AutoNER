# AutoNER
auto model for ner

该包可以帮助给定文件做命名体识别。

## 输入

目前支持两种文件格式输入

1. jsonl文件，其内部格式为：`{"data": "我喜欢吃汉堡和薯条。", "label": [[4, 6, 'food'], [7, 9, 'food']]}`

2. bmes，bio，txt文件。

   如bio格式为：`我 喜 欢 吃 汉 堡 和 薯 条 。\tO O O O B-food I-food O B-food I-food O`

   上面其他格式类似。

## 输出

jsonl文件，其内部格式为：`{"data": "我喜欢吃汉堡和薯条。", "label": [[4, 6, 'food'], [7, 9, 'food']]}`

## 训练

在config.py文件中更改相关参数。然后运行train.py得到模型model.pth和训练数据的字典vocab.pkl。

## 解码

使用decode.py的decode方法，输入为单个句子。

## 参数解释

embed_dim：输入token的embedding size

feature_dim：bilstm的输出size

max_vocab_size：字典的最大元素大小

min_word_freq：字典内元素的最低的词频

tags：entity的类型。输入为列表，例如 ['food', 'drink']

epoch：训练周期

lr：学习率

weight_decay：权重衰减

train_file_path：训练集路径

dev_file_path：开发集路径

bioes_format：是否使用bioes标签  # bioes if True else bio

save_path：保存的目录
