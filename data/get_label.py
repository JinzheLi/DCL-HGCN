import os, json
from tqdm import tqdm

def get_labels_dict(data_dir):
    labels_set = set()
    for flag in ["train", "dev", "test"]:
        data_path = os.path.join(data_dir, flag + '.json')
        if not os.path.exists(data_path):
            continue

        with open(data_path, "r", encoding='utf-8') as fh:
            word_json = json.load(fh)
        for d in tqdm(word_json):
            labels_set.add(d["relation"])
    save_path = os.path.join(data_dir, "label.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(list(labels_set), f, ensure_ascii=False)

def get_dep_type(data_dir):
    type_set = set()
    for flag in ["train", "dev", "test"]:
        data_path = os.path.join(data_dir, flag + '.json')
        if not os.path.exists(data_path):
            continue
        with open(data_path, "r", encoding='utf-8') as fh:
            word_json = json.load(fh)
        for d in tqdm(word_json):
            for deprel in d["stanford_deprel"]:
                type_set.add(deprel)
    save_path = os.path.join(data_dir, "dep_type.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(list(type_set), f, ensure_ascii=False)

def get_pos_type(data_dir):
    pos_set = set()
    for flag in ["train", "dev", "test"]:
        data_path = os.path.join(data_dir, flag + '.json')
        if not os.path.exists(data_path):
            continue
        with open(data_path, "r", encoding='utf-8') as fh:
            word_json = json.load(fh)
        for d in tqdm(word_json):
            for pos in d["stanford_pos"]:
                pos_set.add(pos)
    save_path = os.path.join(data_dir, "pos_type.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(list(pos_set), f, ensure_ascii=False)

def get_ner_type(data_dir):
    ner_set = set()
    for flag in ["train", "dev", "test"]:
        data_path = os.path.join(data_dir, flag + '.json')
        if not os.path.exists(data_path):
            continue
        with open(data_path, "r", encoding='utf-8') as fh:
            word_json = json.load(fh)
        for d in tqdm(word_json):
            for ner in d["stanford_ner"]:
                ner_set.add(ner)
    save_path = os.path.join(data_dir, "ner_type.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(list(ner_set), f, ensure_ascii=False)

# get_labels_dict("tacred")
# get_dep_type("tacred")
# get_pos_type("tacred")
# get_ner_type("tacrev")
