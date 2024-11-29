import torch
import json
from tqdm import tqdm
import heapq
import re
import numpy as np

model = "RDGCN"
la = "fr"
lang = "dbp15k_valid/"
k = 10
print(lang)

ref_data = []
with open("data/"+la+"_en/" +'/ref_ent_ids', 'r') as f:
    for line in f.readlines():
        t = tuple(line.split())
        if len(t) != 2:
            print('===')
        ref_data.append((int(t[0]), int(t[1])))

data_test = ref_data[:10500]
pair_num = len(data_test)
print("测试数据：")
print(pair_num)
#测试集左右实体
test_l = [int(x[0]) for x in data_test]
test_r = [int(x[1]) for x in data_test]


test_l_t = torch.tensor(test_l)
test_r_t = torch.tensor(test_r)



#左右实体的实体名向量
# emb_l = embedding_list[test_l_t]
# emb_r = embedding_list[test_r_t]
#左右实体的向量
vec_se = np.load(lang + "noval" + model.lower() + la + "vector.npy")

vec_se = torch.tensor(vec_se)

emb_l_g = vec_se[test_l_t]
emb_r_g = vec_se[test_r_t]


#计算距离
S = torch.cdist(emb_l_g, emb_r_g, p=1)
candi_id = S.topk(k, largest=False)[1]
# print(candi_id)
candi_ent = test_r_t[candi_id]
# print("gcnnnnnnnnnnn")
# print(candi_ent[:50])
top1 = S.topk(1, largest=False)[1]
ent_top1 = test_r_t[top1]



# 候选选择的准确率
H10 = (candi_ent == test_r_t.view(-1, 1)).sum().item()/pair_num
H1 = (ent_top1 == test_r_t.view(-1, 1)).sum().item()/pair_num
# print((candi_id == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item())
# print(torch.arange(pair_num, device=S.device).view(-1, 1))
'''
for i in candi_id:
    print(test_r_t[i])

'''
print("只使用" + model + "的结果：")
print("hits@10", H10)
print("hits@1", H1)



#实体id和属性
# ent1_attr = load_ent_attr('/ent1_attr_trans') #实体id和属性
# ent2_attr = load_ent_attr('/ent2_attr_dic')

# attr_trans = load_attr_trans('/attr_trans')
#
# attr_v_zh = load_attr_v('/attr_dic_zh')
# attr_v_en = load_attr_v('/attr_dic_en')
#
# attr_ent1_v = load_attr_v('/attr_ent1_v_dic')
# attr_ent2_v = load_attr_v('/attr_ent2_v_dic')




# entid1_attr_v = id_attr_v(ent1_dic, attr_v_zh)
# entid2_attr_v = id_attr_v(ent2_dic, attr_v_en)
# entid1_attr_v_500_trans = load_attr_v('/entid1_attr_v_500_trans')

# entid1_v = id_attr_v(ent1_dic, attr_ent1_v)
# entid2_v = id_attr_v(ent2_dic, attr_ent2_v)
#加载实体的属性值信息


def load_re_f(path):
    results = []
    with open("data/"+la+"_en/" + path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            # t = tuple(line.strip('\n').replace('[', '').replace(')', '').replace('(', '').replace(']', ''))
            t = line.strip('\n').replace('[', '').replace(')', '').replace('(', '').replace(']', '')
            results.append(int(t))
    return results

def load_re_mn(path):
    results = []
    with open("data/"+la+"_en/"  + path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            # t = tuple(line.strip('\n').replace('[', '').replace(')', '').replace('(', '').replace(']', ''))
            t = line.strip('\n').replace('[', '').replace(')', '').replace('(', '').replace(']', '').replace('tensor', '').split(',')
            results.append([int(i) for i in t])
    return results


# print(final_v)
# print(max_n_v)
# print(max_p_v)
def todic(lis):
    dic = {}
    c = 0
    for i in lis:
        dic[c] = i
        c += 1
    return dic


final_attr = load_re_f("final_attr")
final_v = load_re_f("final_v")
final_attr_v = load_re_f("final_attr_v")

max_n_attr = load_re_mn("max_n_attr")
max_n_v = load_re_mn("max_n_v")
max_n_attr_v = load_re_mn("max_n_attr_v")


max_p_attr = load_re_mn("max_p_attr")
max_p_v = load_re_mn("max_p_v")
max_p_attr_v = load_re_mn("max_p_attr_v")





Hk_attr = (torch.tensor(max_n_attr) == test_r_t.view(-1, 1)).sum().item()/pair_num
H1_attr = (torch.tensor(final_attr) == test_r_t).sum().item()/pair_num



Hk_v = (torch.tensor(max_n_v) == test_r_t.view(-1, 1)).sum().item()/pair_num
H1_v = (torch.tensor(final_v) == test_r_t).sum().item()/pair_num

Hk_attr_v = (torch.tensor(max_n_attr_v) == test_r_t.view(-1, 1)).sum().item()/pair_num
H1_attr_v = (torch.tensor(final_attr_v) == test_r_t).sum().item()/pair_num
print("只使用属性名的结果：")
print("hits@10", Hk_attr)
print("hits@1", H1_attr)

print("只使用属性值的结果：")
print("hits@10", Hk_v)
print("hits@1", H1_v)

print("使用属性名和值的结果：")
print("hits@10", Hk_attr_v)
print("hits@1", H1_attr_v)

def interaction(name_ent, attr_ent, maxp):
    inter_ent = []
    l = len(name_ent)
    for i in range(l):
        if maxp[i][0] > maxp[i][1]:
            inter_ent.append(attr_ent[i])
        else:
            inter_ent.append(name_ent[i][0])
    return inter_ent

inter_ent_attr = interaction(ent_top1.tolist(), final_attr, max_p_attr)
inter_ent_v = interaction(ent_top1.tolist(), final_v, max_p_v)
inter_ent_attr_v = interaction(ent_top1.tolist(), final_attr_v, max_p_attr_v)
# print(torch.tensor(inter_ent).view(-1, 1))









H1_all_attr = (torch.tensor(inter_ent_attr) == test_r_t).sum().item()/pair_num
H1_all_v = (torch.tensor(inter_ent_v) == test_r_t).sum().item()/pair_num
H1_all_attr_v = (torch.tensor(inter_ent_attr_v) == test_r_t).sum().item()/pair_num
print("结合后的hits@1：")
print("只使用属性名的结果：", H1_all_attr)
print("只使用属性值的结果：", H1_all_v)
print("只使用属性名和值的结果：", H1_all_attr_v)




'''

with open(lang +'/ref_ent_ids', 'r') as f:
    for line in f.readlines():
        t = tuple(line.split())
        if len(t) != 2:
            print('===')
        ref_data.append((int(t[0]), int(t[1])))

with open(lang +'/ref_ent_ids', 'r') as f:
    for line in f.readlines():
        t = tuple(line.split())
        if len(t) != 2:
            print('===')
        data.append(t)

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
            f.write(d + "\n")
    print('save'+ lang +' to file success...')

def write2dic(name, dic):
    with open(name, 'w') as f:
        for key in dic:
            pros = str(key)
            for pro in dic[key]:
                pros = pros + "/" + str(pro).strip("\n")
            f.write(pros + "\n")
    print('save'+ lang +' to file success...')

# emb_path = 'zh_en/zh_vectorList.json'
# with open(emb_path, 'r', encoding='utf-8') as f:
#     embedding_list = torch.tensor(json.load(f))

def load_ent_attr(path):
    ent_attr = {}
    with open(lang + path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip("\n")
            t = tuple(line.split("/"))
            ent_attr[int(t[0])] = list(t[1:])
    return ent_attr

def load_attr_trans(path):
    attr_trans = {}
    with open(lang + path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.split()
            attr_trans[line[0]] = line[1]
    return attr_trans


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

def load_re_f(path):
    results = []
    with open(lang + path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            # t = tuple(line.strip('\n').replace('[', '').replace(')', '').replace('(', '').replace(']', ''))
            t = line.strip('\n').replace('[', '').replace(')', '').replace('(', '').replace(']', '')
            results.append(int(t))
    return results

def load_re_mn(path):
    results = []
    with open(lang + path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            # t = tuple(line.strip('\n').replace('[', '').replace(')', '').replace('(', '').replace(']', ''))
            t = line.strip('\n').replace('[', '').replace(')', '').replace('(', '').replace(']', '').replace('tensor', '').split(',')
            results.append([int(i) for i in t])
    return results

def ent_to_dic(ent):
    ent_dic = {}
    for e in ent:
        ent_dic[int(e[0])] = e[1].split("/")[-1]
    return ent_dic

def id_attr_v(ent, attr_v):
    entid_attr_v = {}
    for e in ent:
        if ent[e] in attr_v:
            entid_attr_v[e] = attr_v[ent[e]]
        else:
            entid_attr_v[e] = []
    return entid_attr_v
#加载实体
# ent1 = load_ent('/ent_ids_1')
# ent1_dic = ent_to_dic(ent1)
# ent2 = load_ent('/ent_ids_2')
# ent2_dic = ent_to_dic(ent2)


#加载三元组
# triples1 = load_triples(lang +'/triples_1')
# triples2 = load_triples(lang +'/triples_2')

# print(len(triples1))
# print(len(triples2))



# train_paris = data[:graph]
# test_pairs = data[graph:graph+p]
#测试集
# test_pairs = data[:p]
# pair_num = len(test_pairs)
# print("number of test_pairs")
# print(pair_num)
# print(lang)
'''

