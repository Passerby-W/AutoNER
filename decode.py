import torch

import config
import pickle
from model import BiLSTM_CRF
from preprocessing import PrePrecessing

with open(config.save_path + '/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

p = PrePrecessing()
word_to_ix = vocab.word_to_index
tag_to_ix = p.get_tag_to_ix()
ix_to_tag = p.get_ix_to_tag()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, config.embed_dim, config.feature_dim).to(device)
model.load_state_dict(torch.load(config.save_path + '/model.pth'))


def decode(sentence):
    res = []
    with torch.no_grad():
        sentence_in = torch.tensor([word_to_ix.get(w, 0) for w in sentence], dtype=torch.long)
        _, predicted_tags = model(sentence_in)
        res.extend(ix_to_tag[tag.item()] for tag in predicted_tags)
    return res


s = '桂枝三两、炙甘草二两、生姜三两、大枣十二枚、牡蛎五两、蜀漆三两、龙骨四两'
res = decode(s)
for i in zip(list(s), res):
    print(i)
