import numpy as np
import os
def get_dict(file):
    d = {}
    d_list = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            d[cur[0]] = cur[1]
            d_list.append(cur[0])
    return d, d_list


def save_list(l, file):
    with open(file, 'w') as f:
        for cur in l:
            f.write(cur + '\n')

def evaluate(w_embed, d_embed):
    count1 = 0
    count3 = 0
    count10 = 0
    # 计算每一个embedding的模
    norm1 = np.linalg.norm(w_embed, axis=-1,keepdims=True)
    norm2 = np.linalg.norm(d_embed, axis=-1,keepdims=True)

    w_embed_norm = w_embed / norm1
    d_embed_norm = d_embed / norm2

    l = len(w_embed)
    cos_matrix = np.dot(w_embed_norm, d_embed_norm.T)
    rank = 0
    for i in range(l):
        index_sort = np.argsort(-cos_matrix[i])
        index = np.where(index_sort == i)[0]
        if index <= 0:
            count1 += 1
            count3 += 1
            count10 += 1
        elif index <= 2:
            count3 += 1
            count10 += 1
        elif index <= 9:
            count10 += 1
        rank += 1 / (index + 1)
    mrr = rank / l
    print('Hit 1:', count1 / l)
    print('Hit 3:', count3 / l)
    print('Hit 10:', count10 / l)
    print('MRR:', mrr)
    return count1 / l
