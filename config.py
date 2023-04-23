# MODEL
embed_dim = 300
feature_dim = 100

# VOCAB
max_vocab_size = 1000000
min_word_freq = 1

# TAG
tags = ['disease', 'symptom', 'checkup', 'organ', 'medicine', 'food', 'department']

# TRAIN
epoch = 3
lr = 0.01
weight_decay = 1e-5

# INPUT
train_file_path = 'example/train.tsv'
dev_file_path = ''
bioes_format = False  # bioes if True else bio

# OUTPUT DIR
save_path = 'example'

