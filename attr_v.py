import torch
import json
from tqdm import tqdm
import heapq
import re
import Levenshtein
import time


start = time.perf_counter()

lang = "data/zh_en"
la = 'zh'
data = []
k = 20
p = 10500
theta = 0.99
lam = 0.
with open(lang + '/ref_ent_ids', 'r') as f:
    for line in f.readlines():
        t = tuple(line.split())
        if len(t) != 2:
            print('===')
        data.append(t)

emb_path = lang + '/' + la + '_vectorList.json'
with open(emb_path, 'r', encoding='utf-8') as f:
    embedding_list = torch.tensor(json.load(f))



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
    with open(lang + path, 'r', encoding='UTF-8') as f:
        c = 0
        for line in f.readlines():
            line = line.strip("\n")
            t = tuple(line.split("/"))
            if t[0].isdigit():
                ent_attr[int(t[0])] = []
                for i in t[1:]:
                    ent_attr[int(t[0])].append(i.replace("'", '').replace('[', '').replace(']', '').split(','))
                c += 1
            else:
                ent_attr[c] = []
                for i in t[0:]:
                    ent_attr[c].append(i.replace("'",'').replace('[','').replace(']','').split(','))
                c += 1
            # ent_attr[int(t[0])] = []
            # if len(t) != 0 and t[0].isdigit():
            #     for i in t[1:]:
            #         ent_attr[int(t[0])].append(i.replace("'",'').replace('[','').replace(']','').split(','))
            # if len(t) != 0 and t[0].isdigit() == False:
            #     print(t[0].isdigit())
            #     for i in t[0:]:
            #         ent_attr[c].append(i.replace("'",'').replace('[','').replace(']','').split(','))
    return ent_attr


def load_attr_v(path):
    ent_attr_v = {}
    with open(lang + path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.replace('\\','').replace('\"','').strip('\n').split('/')
            ent_attr_v[line[0]] = line[1:]
    return ent_attr_v


def load_ent(path):
    ent = []
    with open(lang + path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            t = tuple(line.split())
            ent.append(t)
    return ent

def ent_to_dic(ent):
    ent_dic = {}
    for e in ent:
        ent_dic[int(e[0])] = e[1].split("/")[-1]
    return ent_dic


test_pairs = data[:p]
print(la)
pair_num = len(test_pairs)
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
print("只使用实体名的结果：")
print("hits@10", H10)
print("hits@1", H1)

#实体id和属性
# ent1_attr = load_ent_attr('/id1_attr_trans') #实体id和属性
# ent2_attr = load_ent_attr('/id2_attr')

ent1_attr_v = load_ent_attr('/id1_attr_v_'+ la +'_en') #实体id和属性
# print(ent1_attr_v)
ent2_attr_v = load_ent_attr('/id2_attr_v')

def compute(l, r, ent1attr, ent2attr):
    final = []
    max_n = []
    max_p = []
    #由左向右对齐

    for tl in tqdm(l, desc='计算覆盖率'):
        attr_over = []
        attr1 = ent1attr[tl] #左实体的属性和属性值
        for tr in r:
            pro_pair = []
            pro_p = []
            for i in attr1:

                for j in ent2attr[tr]:
                    if Levenshtein.ratio(i[0].lower(), j[0].lower()) > theta:
                        if len(i) > 1 and len(j) > 1 and Levenshtein.ratio(i[1].lower(), j[1].lower()) > theta:
                            pro_pair.append((i[0],j[0]))


            # for pro in pro_pair:
            #     pro_p.append(0.5*(p_zh[pro[0]] + p_en[pro[1]]))


            over = len(pro_pair)
            over_ration = round(over, 2)
            # print(over_ration)
            attr_over.append(over_ration)

        id_over = dict(zip(r, attr_over))
        ent_id = sorted(id_over, key=id_over.get, reverse=True)
        final.append(ent_id[:1])
        max_n.append(ent_id[:k])
        # max_p.append((id_over[ent_id[0]], id_over[ent_id[1]], id_over[ent_id[2]], id_over[ent_id[3]], id_over[ent_id[4]]))
        max_p.append((id_over[ent_id[0]], id_over[ent_id[1]], id_over[ent_id[2]]))

    return final, torch.tensor(max_n, dtype=int), max_p

final_attr_v, max_n_attr_v, max_p_attr_v = compute(test_l, test_r, ent1_attr_v, ent2_attr_v)
# print(torch.tensor(final).view(-1, 1))
# print(max_n)
# print(torch.tensor(final).view(-1, 1))
# print(torch.tensor(max_p))

Hk_attr = (max_n_attr_v == test_r_t[:p].view(-1, 1)).sum().item()/pair_num
# Hk_attr = (max_n == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
H1_attr = (torch.tensor(final_attr_v) == test_r_t[:p].view(-1, 1)).sum().item()/pair_num

print("相似性阈值", theta)
print("使用属性和属性值的结果：")
print("hits@10", Hk_attr)
print("hits@1", H1_attr)

write2file(lang + '/final_attr_v', final_attr_v)
write2file(lang + '/max_n_attr_v', max_n_attr_v)
write2file(lang + '/max_p_attr_v', max_p_attr_v)

def interaction(name_ent, attr_ent, maxp):
    inter_ent = []
    l = len(name_ent)
    for i in range(l):
        if maxp[i][0] > maxp[i][1]:
            inter_ent.append(attr_ent[i])
        else:
            inter_ent.append(name_ent[i])
    return inter_ent

inter_ent = interaction(ent_top1.tolist(), final_attr_v, max_p_attr_v)
# print(torch.tensor(inter_ent).view(-1, 1))


H1_all = (torch.tensor(inter_ent) == test_r_t[:pair_num].view(-1, 1)).sum().item()/pair_num
print("结合后的hits@1：")
print(H1_all)

end = time.perf_counter()
runTime = end - start

print("运行时间:", runTime)


