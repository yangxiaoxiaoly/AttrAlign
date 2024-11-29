import re
import copy
import string
lang = 'data/D_W_15K_V1'
la = 'jape'
lam = 0.5
def load_ent(path):
    ent = []
    with open(lang + path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            t = tuple(line.split('\t'))
            ent.append(t)
    return ent

def ent_to_dic(ent):
    ent_dic = {}
    for e in ent:
        ent_dic[int(e[1])] = e[0]
    return ent_dic

def ent_to_id(ent, attr):
    id_attr_dic = {}
    for e in ent:
        if ent[e] in attr:
            id_attr_dic[e] = attr[ent[e]]
        else:
            id_attr_dic[e] = []
    return id_attr_dic



#加载实体、属性、属性值、类型
def load_attr_triples(path):
    attr_all = []
    with open(lang + path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            t = tuple(line.strip('\n').split('\t'))
            # v_t = re.split("[@ ^^]", ''.join(t[2:]))
            attr_all.append((t[0], t[1], t[2]))
    return attr_all

#处理属性三元组
def process_attr(attr):
    ent_attr_v_dic = {} #实体：（属性、属性值、类型）
    attr_v_dic = {} #属性：属性值
    attr_vn_dic = {} #属性：属性值个数
    attr_fre = {} #属性：出现次数
    ent_v_dic = {}  #实体：属性值
    ent_attr_dic = {}  #实体：属性
    for i in attr:
        if i[0] in ent_attr_v_dic:
            ent_attr_v_dic[i[0]].append([i[1], i[2]])
        else:
            ent_attr_v_dic[i[0]] = [[i[1], i[2]]]
    for i in attr:
        if i[1] in attr_v_dic:
            attr_v_dic[i[1]].append(i[2])
        else:
            attr_v_dic[i[1]] = [i[2]]
    for i in attr_v_dic:
        attr_vn_dic[i] = len(attr_v_dic[i])
    attr_vn_dic = dict(sorted(attr_vn_dic.items(), key=lambda x: x[1]))
    for i in attr:
        if i[1]in attr_fre:
            attr_fre[i[1]] += 1
        else:
            attr_fre[i[1]] = 1
    attr_fre = dict(sorted(attr_fre.items(), key=lambda x: x[1]))
    for i in attr:
        if i[0] in ent_v_dic:
            ent_v_dic[i[0]].append(i[2])
        else:
            ent_v_dic[i[0]] = [i[2]]
    for i in attr:
        if i[0] in ent_attr_dic:
            ent_attr_dic[i[0]].append(i[1])
        else:
            ent_attr_dic[i[0]] = [i[1]]
    return ent_attr_v_dic, attr_vn_dic, attr_fre, ent_v_dic, ent_attr_dic

def write_dic_2file(name, dic):
    with open(name, 'w') as f:
        for key in dic:
            pros = str(key)
            for pro in dic[key]:
                pros = pros + "#" + str(pro)
            f.write(pros + "\n")
    print('save'+ lang +' to file success...')
def write_data_2file(name, dic):
    with open(name, 'w') as f:
        for key in dic:
            f.write(str(key) +' ' + str(dic[key]) + "\n")
    print('save'+ lang +' to file success...')

#实体字典：id：实体
ent1 = load_ent('/' + la + '_kg1_ent_ids')
ent1_dic = ent_to_dic(ent1)
ent2 = load_ent('/' + la + '_kg2_ent_ids')
ent2_dic = ent_to_dic(ent2)


attr_all2 = load_attr_triples('/attr_triples_2') #加载2属性三元组
attr_all1 = load_attr_triples('/attr_triples_1') #加载1属性三元组

ent1_attr_v, attr_vn_1, attr_fre_1, ent1_v, ent1_attr = process_attr(attr_all1)  #生成中文三元组
ent2_attr_v, attr_vn_2, attr_fre_2, ent2_v, ent2_attr = process_attr(attr_all2) #生成英文三元组

id1_attr_v = ent_to_id(ent1_dic, ent1_attr_v)
id2_attr_v = ent_to_id(ent2_dic, ent2_attr_v)

id1_v = ent_to_id(ent1_dic, ent1_v)
id2_v = ent_to_id(ent2_dic, ent2_v)

id1_attr = ent_to_id(ent1_dic, ent1_attr)
id2_attr = ent_to_id(ent2_dic, ent2_attr)








write_dic_2file(lang + '/id1_attr_v', id1_attr_v) #需要翻译的文件
write_dic_2file(lang + '/id2_attr_v', id2_attr_v)
write_dic_2file(lang + '/id1_attr', id1_attr) #需要翻译的文件
write_dic_2file(lang + '/id2_attr', id2_attr)
write_dic_2file(lang + '/id1_v', id1_v) #需要翻译的文件
write_dic_2file(lang + '/id2_v', id2_v)

write_data_2file(lang + '/attr_vn_1', attr_vn_1) #需要翻译的文件
write_data_2file(lang + '/attr_vn_2', attr_vn_2)
write_data_2file(lang + '/attr_fre_1', attr_fre_1) #需要翻译的文件
write_data_2file(lang + '/attr_fre_2', attr_fre_2)


