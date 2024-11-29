from tqdm import tqdm

lang = "data/zh_en"

ref_data = []
sup_data = []
with open(lang +'/ent_ILLs', 'r', encoding='UTF-8') as f:
    for line in f.readlines():
        t = tuple(line.strip().split())
        if len(t) != 2:
            print('===')
        ref_data.append((t[0], t[1]))
# with open(lang +'/sup_pairs', 'r') as f:
#     for line in f.readlines():
#         t = tuple(line.split())
#         if len(t) != 2:
#             print('===')
#         sup_data.append((int(t[0]), int(t[1])))


def load_triples(path):
    tr = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            t = tuple(line.strip().split())
            if (len(t) != 3):
                print('===')
            tr.append((t[0], t[1], t[2]))
    return tr
data = ref_data

def write2file(name, dat):
    with open(name, 'w', encoding='UTF-8') as f:
        for d in dat:
            f.write(str(d[0]) + "\t" + str(d[1]) + "\n")
    print('save'+ lang +' to file success...')

# write2file(lang + '/ref_ent_ids', data)

triples1 = load_triples(lang +'/zh_rel_triples')
triples2 = load_triples(lang +'/en_rel_triples')

print(len(triples1))
print(len(triples2))

# data = ref_data
overlap = []
for i in tqdm(data, desc='计算覆盖率'):
    adjh1 = [k[2] for k in triples1 if k[0] == i[0]]
    adjt1 = [k[0] for k in triples1 if k[2] == i[0]]
    adj1 = set(adjh1 + adjt1)
    adjh = [k[2] for k in triples2 if k[0] == i[1]]
    adjt = [k[0] for k in triples2 if k[2] == i[1]]
    adj = set(adjh + adjt)
    c = 0
    for e1 in adj1:
        for e2 in adj:
            if (e1, e2) in data:
                c += 1
    over_ration = c / min(len(adj1), len(adj))
    over_ration = round(over_ration, 2)
    if over_ration < 0.3:
        print(i, adj1, adj)
    overlap.append(over_ration)




# over_1 = []
# for i in overlap:
#     if  i  <= 0.5:
#         over_1.append(i)
# r_n = len(over_1) / len(overlap)
# r_n = round(r_n, 3)
# print(r_n)
# over_2 = []
# for i in overlap:
#     if  i  > 0.5:
#         over_2.append(i)
# r_n = len(over_2) / len(overlap)
# r_n = round(r_n, 3)
# print(r_n)
# over_3 = []
# for i in overlap:
#     if  i  == 1:
#         over_3.append(i)
# r_n = len(over_3) / len(overlap)
# r_n = round(r_n, 3)
# print(r_n)
#
# def cop_over(n, over):
#     over_n = []
#     for i in over:
#         if n <= i < n+0.1:
#             over_n.append(i)
#     r_n = len(over_n) / len(over)
#     r_n = round(r_n, 3)
#     return r_n
#
# r_1 = cop_over(0, overlap)
# r_2 = cop_over(0.1, overlap)
# r_3 = cop_over(0.2, overlap)
# r_4 = cop_over(0.3, overlap)
# r_5 = cop_over(0.4, overlap)
# r_6 = cop_over(0.5, overlap)
# r_7 = cop_over(0.6, overlap)
# r_8 = cop_over(0.7, overlap)
# r_9 = cop_over(0.8, overlap)
# r_10 = cop_over(0.9, overlap)
#
# print(r_1, r_2, r_3, r_4,r_5,r_6,r_7,r_8,r_9,r_10)