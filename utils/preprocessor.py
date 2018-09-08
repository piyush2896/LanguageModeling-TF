import nltk
import os

def text_to_ids(word_id_map, txt):
    words = txt.split()
    ids = []
    for word in words:
        if word.lower() not in word_id_map:
            ids.append(len(word_id_map))
        else:
            ids.append(word_id_map[word.lower()])
    return ids

def reverse_kv_pairs(word_id_map):
    return dict(zip(word_id_map.values(), word_id_map.keys()))

def ids_to_text(word_id_map, ids):
    words = []
    id_word_map = reverse_kv_pairs(word_id_map)
    for id in ids:
        if id == len(id_word_map):
            words.append('<UNK>')
        else:
            words.append(id_word_map[id])
    return ' '.join(words)

def sort_seq_len(src, dest):
    if os.path.isfile(dest):
        return
    with open(dest, 'w+') as fw:
        with open(src, 'r') as fr:
            lines = fr.readlines()
            lines = sorted(lines, key=lambda x: len(x))
        for line in lines:
            fw.write(line)
