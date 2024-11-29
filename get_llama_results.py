import numpy as np
from Param import *
import torch

def get_llama_results(path):
    data = {}
    with open(path, 'r', encoding='UTF-8') as f:
        c = 0
        for line in f.readlines():
            line = line.strip().replace("[","").replace("]","").replace("\'","").replace(" ","")
            data[c] = line.split(",")
            c += 1
        return data

def get_seg_results(path):
    e = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            e.append(line.strip())
    return e

rerank_result_attr = get_llama_results(fold + lang + "llama3_results_allattr_step_50")

rerank_result_seg = get_seg_results(fold + lang + "new_tem0_llama8b_hard_result_id10")
# print(rerank_result)

hard_list = np.load(fold + lang + "hard_list_10.npy")
hard_topk = np.load(fold + lang + 'hard_top'+ f + '.npy')

attr_hard_rerank_results = []
for re in rerank_result_attr:
    if "Yes" in rerank_result_attr[re]:
        attr_hard_rerank_results.append(hard_topk[re][rerank_result_attr[re].index("Yes")])
    else:
        attr_hard_rerank_results.append(hard_topk[re][0])

seg_hard_rerank_results = []
id_dic = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9}
for i in range(len(hard_list)):
    if rerank_result_seg[i] not in id_dic.keys():
        seg_hard_rerank_results.append(0)
    else:
        id = hard_topk[i][id_dic[rerank_result_seg[i]]]
        seg_hard_rerank_results.append(id)

seg_hard_rerank_results = torch.tensor(seg_hard_rerank_results)


hard_rerank_results = attr_hard_rerank_results
# print(hard_list)
# print(np.array(hard_rerank_results))

# a = np.array([1, 3, 4, 5, 7, 9])
# b = np.array([2, 3, 5, 7, 8, 9])
# pair_num = len(hard_list)

hard_H1 = (torch.tensor(hard_rerank_results) == torch.tensor(hard_list)).sum().item()/len(hard_list)
print("hits@1 in hard list:")
print(lang,hard_H1)



top_1 = np.load(fold + lang + 'top1.npy')

# c = torch.tensor([1,2])
#
# b[c] = torch.tensor([1,2])
# print(b)
top_1[torch.tensor(hard_list)] = torch.tensor(hard_rerank_results).view(-1, 1)

pair_num = 10500
H1 = (torch.tensor(top_1) == torch.arange(pair_num).view(-1, 1)).sum().item()/pair_num
print("hits@1 in all list:")
print(lang,H1)
