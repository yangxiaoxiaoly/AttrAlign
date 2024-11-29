import torch
import json
from tqdm import tqdm
import heapq
import re
import Levenshtein
import time
import numpy as np

start = time.perf_counter()
lang = "data/D_Y_15K_V1"
la = 'jape'
data = []
k = 10
p = 10500
theta = 0.99
lam = 0.5
with open(lang + '/test_ids', 'r') as f:
    for line in f.readlines():
        t = tuple(line.split('\t'))
        if len(t) != 2:
            print('===')
        data.append(t)



emb_path = lang + '/' + la +'_ent_embeds.npy'

# with open(emb_path, 'r', encoding='utf-8') as f:

embedding_list = np.load(emb_path)

embedding_list = torch.tensor(embedding_list)


def load_triples(path):
    tr = []
    with open(path, 'r') as f:
        for line in f.readlines():
            t = tuple(line.split())
            if (len(t) != 3):
                print('===')
            tr.append((int(t[0]), int(t[1]), int(t[2])))
    return tr

def write2file(name, data):
    with open(name, 'w') as f:
        for d in data:
            f.write(str(d) + "\n")
    print('save'+ lang +' to file success...')

def write2dic(name, dic):
    with open(name, 'w') as f:
        for key in dic:
            pros = str(key)
            for pro in dic[key]:
                pros = pros + "/" + str(pro).strip("\n")
            f.write(pros + "\n")
    print('save'+ lang +' to file success...')


def load_ent_attr(path):
    ent_attr = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip("\n")
            t = tuple(line.split("#"))
            ent_attr[int(t[0])] = set([i.split('/')[-1] for i in t[1:]])
    return ent_attr


def load_attr_v(path):
    ent_attr_v = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip("\n")
            t = tuple(line.split("#"))
            ent_attr_v[int(t[0])] = set(t[1:])
    return ent_attr_v


def load_ent(path):
    ent = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            t = tuple(line.split())
            ent.append(t)
    return ent

def ent_to_dic(ent):
    ent_dic = {}
    for e in ent:
        ent_dic[int(e[0])] = e[1]
    return ent_dic




test_pairs = data
pair_num = len(test_pairs)
print(la)
print("number of test_pairs")
print(pair_num)



test_l = [int(x[0]) for x in test_pairs]
test_r = [int(x[1]) for x in test_pairs]


test_l_t = torch.tensor(test_l)
test_r_t = torch.tensor(test_r)


emb_l = embedding_list[test_l_t]
emb_r = embedding_list[test_r_t]

S = torch.cdist(emb_l, emb_r, p=1)
candi_id = S.topk(k, largest=False)[1]
# print(candi_id)
candi_ent = test_r_t[candi_id]
# print(candi_ent)
top1 = S.topk(1, largest=False)[1]
ent_top1 = test_r_t[top1]
# print(ent_top1)



# 候选选择的准确率
H10 = (candi_id == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
H1 = (top1 == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
# print((candi_id == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item())
# print(torch.arange(pair_num, device=S.device).view(-1, 1))


'''
for i in candi_id:
    print(test_r_t[i])

'''
print(lang + la + "的结果：")
print("hits@10", H10)
print("hits@1", H1)



#实体id和属性
ent1_attr = load_ent_attr(lang + '/id1_attr') #实体id和属性
ent2_attr = load_ent_attr(lang + '/id2_attr')
#


'''
def load_attr_frevn(path):
    attr_frevn = {}
    with open(lang + path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.split()
            attr_frevn[' '.join(line[:-1])] = int(line[-1])
    return attr_frevn

attr_fre_zh = load_attr_frevn('/attr_fre_zh_trans')
attr_fre_en = load_attr_frevn('/attr_fre_en')
attr_vn_zh = load_attr_frevn('/attr_vn_zh_trans')
attr_vn_en = load_attr_frevn('/attr_vn_en')

def com_p(fre, vn):
    attr_p = {}
    for i in fre:
        attr_p[i] = lam * 1 / fre[i] + (1 - lam) * 1 / vn[i]
    return attr_p


attr_p_zh = com_p(attr_fre_zh, attr_vn_zh)
attr_p_en = com_p(attr_fre_en, attr_vn_en)
'''






def compute(l, r, ent1attr, ent2attr):
    final = []
    max_n = []
    max_p = []
    #由左向右对齐

    for tl in tqdm(l, desc='计算覆盖率'):
        attr_over = []
        attr1 = ent1attr[tl] #左实体的属性
        for tr in r:
            pro_pair = []
            pro_p = []
            for i in set(attr1):
                for j in set(ent2attr[tr]):
                    if Levenshtein.ratio(i.lower(), j.lower()) > theta:
                        pro_pair.append((i,j))
            # for pro in pro_pair:
            #     pro_p.append(0.5*(p_zh[pro[0]] + p_en[pro[1]]))

            # over = len(pro_pair)
            # if len(pro_p) == 0:
            #     over = 0
            # else:
            #     over = max(pro_p)

            over = len(pro_pair)
            # uni = len(set(attr1) | set(ent2attr[tr])) #所有wei属性会偏大，因为未对齐
            # over = len(pro_pair) / uni

            # inter = set(attr1) & set(ent2attr[tr]) #两个实体的共有属性
            # print(inter)
            # uni = len(set(attr1) | set(ent2attr[tr]))
            # if len(inter) == 0:
            #     over = 0
            # else:
            #     over = max([arrtp[i] for i in inter])
            # if min(len(attr1), len(ent2attr[tr])) == 0:
            #     over_ration = 0
            # else:
            #     over_ration = over / min(len(attr1), len(ent2attr[tr]))

            over_ration = round(over, 2)
            # print(over_ration)
            attr_over.append(over_ration)

        id_over = dict(zip(r, attr_over))
        ent_id = sorted(id_over, key=id_over.get, reverse=True)
        final.append(ent_id[:1])
        max_n.append(ent_id[:k])
        # max_p.append((id_over[ent_id[0]], id_over[ent_id[1]], id_over[ent_id[2]], id_over[ent_id[3]], id_over[ent_id[4]]))
        max_p.append((id_over[ent_id[0]], id_over[ent_id[1]], id_over[ent_id[2]]))



        # max_over = max(attr_over)
        # id = attr_over.index(max_over)
        # final.append(r[id])
        #
        # list_over = list(zip())
        # max_n_index = sorted(list_over, key=lambda x: x[1], reverse=True)
        # max_n.append(torch.tensor(max_n_index)[:, 0].tolist()[:k])
    return final, max_n, max_p

final_attr, max_n_attr, max_p_attr = compute(test_l, test_r, ent1_attr, ent2_attr)
# print(torch.tensor(final).view(-1, 1))
# print(max_n)
# print(torch.tensor(final).view(-1, 1))
# print(torch.tensor(max_p))

# write_data_2file(lang + '/attr_fre_zh_trans', attr_fre_zh_trans)
# write_data_2file(lang + '/attr_vn_zh_trans', attr_vn_zh_trans)
write2file(lang + '/' + la + 'final_attr', final_attr)
write2file(lang + '/' + la + 'max_n_attr', max_n_attr)
write2file(lang + '/' + la + 'max_p_attr', max_p_attr)





Hk_attr = (torch.tensor(max_n_attr, dtype=int) == test_r_t[:p].view(-1, 1)).sum().item()/pair_num
# Hk_attr = (max_n == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
H1_attr = (torch.tensor(final_attr) == test_r_t[:p].view(-1, 1)).sum().item()/pair_num

print("相似性阈值", theta)
print("只使用attr的结果：")
print("hits@10", Hk_attr)
print("hits@1", H1_attr)

def interaction(name_ent, attr_ent, maxp):
    inter_ent = []
    l = len(name_ent)
    for i in range(l):
        if maxp[i][0] > maxp[i][1]:
            inter_ent.append(attr_ent[i])
        else:
            inter_ent.append(name_ent[i])
    return inter_ent

inter_ent = interaction(ent_top1.tolist(), final_attr, max_p_attr)
# print(torch.tensor(inter_ent).view(-1, 1))


H1_all = (torch.tensor(inter_ent) == test_r_t[:pair_num].view(-1, 1)).sum().item()/pair_num
print("结合后的hits@1：")
print(H1_all)

end = time.perf_counter()
runTime = end - start

print("运行时间:", runTime)










