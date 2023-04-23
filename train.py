from preprocessing import PrePrecessing
from model import BiLSTM_CRF
import config
import torch
from torch import optim
from tqdm import tqdm

# params
p = PrePrecessing()
train_dataset = p.get_train_dataset()
dev_dataset = p.get_dev_dataset()
word_to_ix = p.get_word_to_ix()
tag_to_ix = p.get_tag_to_ix()
ix_to_tag = p.get_ix_to_tag()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, config.embed_dim, config.feature_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def bio_tags_to_entities(tags):
    """
    将BIO标记转换为实体边界。
    :param tags: 包含BIO标记的列表
    :return: 包含实体边界的列表
    """
    entities = []
    entity = None
    for i, tag in enumerate(tags):
        if tag[0] == "B":
            if entity is not None:
                entities.append(entity)
            entity = {"type": tag[2:], "start": i, "end": i}
        elif tag[0] == "I":
            if entity is not None and entity["type"] == tag[2:]:
                entity["end"] = i
            else:
                entity = None
        else:
            if entity is not None:
                entities.append(entity)
            entity = None
    if entity is not None:
        entities.append(entity)
    return entities


def bioes_tags_to_entities(tags):
    """
    将BIOES标记转换为实体边界。
    :param tags: 包含BIOES标记的列表
    :return: 包含实体边界的列表
    """
    entities = []
    entity = None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if entity is not None:
                entities.append(entity)
            entity = {"type": tag[2:], "start": i, "end": i}
        elif tag.startswith("I-"):
            if entity is not None and entity["type"] == tag[2:]:
                entity["end"] = i
            else:
                entity = None
        elif tag.startswith("E-"):
            if entity is not None and entity["type"] == tag[2:]:
                entity["end"] = i
                entities.append(entity)
            entity = None
        elif tag.startswith("S-"):
            if entity is not None:
                entities.append(entity)
            entity = {"type": tag[2:], "start": i, "end": i}
            entities.append(entity)
            entity = None
        else:
            if entity is not None:
                entities.append(entity)
            entity = None
    if entity is not None:
        entities.append(entity)
    return entities


def entity_f1_score(true_entities_list, pred_entities_list):
    """
    计算实体级别的F1分数。
    :param true_entities_list: 真实的实体边界列表的列表
    :param pred_entities_list: 预测的实体边界列表的列表
    :return: 实体级别的F1分数
    """
    tp = 0  # 真正例
    fp = 0  # 假正例
    fn = 0  # 假反例
    for true_entities, pred_entities in zip(true_entities_list, pred_entities_list):
        true_entities = set(tuple(entity.items()) for entity in true_entities)
        pred_entities = set(tuple(entity.items()) for entity in pred_entities)
        for true_entity in true_entities:
            if true_entity in pred_entities:
                tp += 1
                pred_entities.remove(true_entity)
            else:
                fn += 1
        fp += len(pred_entities)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fp > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1


def evaluate_and_save_best_model(model):
    true_entities_list, pred_entities_list = [], []

    evaluate_dataset = dev_dataset if dev_dataset else train_dataset
    with torch.no_grad():
        for data in evaluate_dataset:
            sentence, tags = data
            sentence_in = torch.tensor([word_to_ix.get(w, 0) for w in sentence], dtype=torch.long)
            _, predicted_tags = model(sentence_in)

            predicted_tags = [ix_to_tag[tag.item()] for tag in predicted_tags]

            if config.bioes_format:
                true_entities_list.append(bioes_tags_to_entities(tags))
                pred_entities_list.append(bioes_tags_to_entities(predicted_tags))
            else:
                true_entities_list.append(bio_tags_to_entities(tags))
                pred_entities_list.append(bio_tags_to_entities(predicted_tags))

    return entity_f1_score(true_entities_list, pred_entities_list)


# Train
best_f1_score = 0
for epoch in range(config.epoch):
    for data in tqdm(train_dataset, desc=f"Training Epoch {epoch + 1}"):
        sentence, tags = data
        model.zero_grad()

        sentence_in = torch.tensor([word_to_ix.get(w, 0) for w in sentence], dtype=torch.long)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        _, sentence_out = model(sentence_in)

        loss = model.neg_log_likelihood(sentence_in, targets)

        loss.backward()
        optimizer.step()

    f1 = evaluate_and_save_best_model(model)
    if dev_dataset:
        tqdm.write(f"Current F1 on dev_set：{f1:.4f}")
    else:
        tqdm.write(f"Current F1 on train_set：{f1:.4f}")
    if f1 > best_f1_score:
        best_f1_score = f1
        torch.save(model.state_dict(), config.save_path + '/model.pth')
    if dev_dataset:
        tqdm.write(f"Best F1 on dev_set: {best_f1_score:.4f}")
    else:
        tqdm.write(f"Best F1 on train_set: {best_f1_score:.4f}")
