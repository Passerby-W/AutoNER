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
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1


true_entities_list = [
    [{"type": "PER", "start": 1, "end": 2}, {"type": "LOC", "start": 4, "end": 6}, {"type": "ORG", "start": 8, "end": 8}],
    [{"type": "PER", "start": 1, "end": 2}, {"type": "LOC", "start": 4, "end": 6}],
    [{"type": "ORG", "start": 0, "end": 2}, {"type": "LOC", "start": 4, "end": 6}]
]
pred_entities_list = [
    [{"type": "PER", "start": 1, "end": 2}, {"type": "LOC", "start": 4, "end": 5}, {"type": "ORG", "start": 8, "end": 8}],
    [{"type": "PER", "start": 1, "end": 2}, {"type": "LOC", "start": 4, "end": 5}],
    [{"type": "ORG", "start": 1, "end": 2}]]

print(entity_f1_score(true_entities_list, pred_entities_list))
# tags = ["O", "B-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-LOC", "O", "B-ORG"]
# entities = bio_tags_to_entities(tags)
# print(entities)
