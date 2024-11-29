# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

from llama import Dialog, Llama
import numpy as np
from tqdm import tqdm

fold = "data/"
lang = "fr_en/"

k = 10
f = str(k)

# def load_pairs(path):
#     pairs = []
#     with open(path, 'r', encoding='UTF-8') as f:
#         for line in f.readlines():
#             t = tuple(line.strip().split('\t'))
#             pairs.append((int(t[0]), int(t[1])))
#     return pairs

# def load_entid(path):
#     e = {}
#     with open(path, 'r', encoding='UTF-8') as f:
#         for line in f.readlines():
#             t = tuple(line.strip().split('\t'))
#             e[int(t[0])] = t[1].split('/')[-1]
#     return e


# def load_ent(path):
#     e1 = []
#     e2 = []
#     with open(path, 'r', encoding='UTF-8') as f:
#         for line in f.readlines():
#             t = tuple(line.strip().split('\t'))
#             e1.append(t[0].split('/')[-1])
#             e2.append(t[1].split('/')[-1])
#     return e1, e2

def write_data_2file(name, data):
    with open(name, 'a', encoding='UTF-8') as f:
        f.write(str(data) + "\n")

# ent1 = load_entid(fold + lang + "ent1_dic")
# ent2 = load_entid(fold + lang + "ent2_dic")
# pairs = load_pairs(fold + lang + "test_links")
# test_pairs = pairs[:10500]

# e1 = [int(p[0]) for p in test_pairs]
# e2 = [int(p[1]) for p in test_pairs]

# e1, e2 = load_ent(fold + lang + "test_links")

def load_ent(path):
    e1 = []
    e2 = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            t = tuple(line.strip().split('\t'))
            e1.append(t[0])
            e2.append(t[1])
    return e1, e2

def load_entid(path):
    e = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            t = tuple(line.strip().split('\t'))
            e[t[1]] = int(t[0])
    return e
def ent_to_dic(ent):
    ent_dic = {}
    i = 0
    for e in ent:
        ent_dic[i] = e.split("/")[-1]
        i += 1
    return ent_dic

ent1, ent2 = load_ent(fold + lang + "test_links")
pair_num = len(ent1)

ent1_dic = ent_to_dic(ent1)
ent2_dic = ent_to_dic(ent2)


hard_list = np.load(fold + lang + "hard_list_10.npy")
hard_topk = np.load(fold + lang + 'hard_top'+ f + '.npy')

def load_attr(path):
    attr_all = {}
    attr_v = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            t = tuple(line.strip().replace("<",'').replace(">",'').split(' '))
            # print(t)
            if t[0].split("/")[-1] not in attr_all:
                attr_all[t[0].split("/")[-1]] = [(t[1].split("/")[-1],t[2].split("^")[0])]
            else:
                attr_all[t[0].split("/")[-1]].append((t[1].split("/")[-1],t[2].split("^")[0]))
            if t[0].split("/")[-1] not in attr_v:
                attr_v[t[0].split("/")[-1]] = [t[2].split("^")[0]]
            else:
                attr_v[t[0].split("/")[-1]].append(t[2].split("^")[0])
        return attr_all, attr_v

ent1_allattr, ent1_v = load_attr(fold + lang + lang.split("_")[0] + "_att_triples")
ent2_allattr, ent2_v = load_attr(fold + lang + "en_att_triples")

def load_attr_name(path):
    name = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            name.append(line.strip())
    return name

attr_name1 = load_attr_name(fold + lang + "all_attr1")
attr_name2 = load_attr_name(fold + lang + "all_attr2")

def cut_dict(dic, name):
    new_dic = {}
    for k in dic:
        new_dic[k] = [item for item in dic[k] if item[0] in name]
    return new_dic

ent1_cutattr = cut_dict(ent1_allattr, attr_name1)
ent2_cutattr = cut_dict(ent2_allattr, attr_name2)

attr_all1 = ent1_cutattr
attr_all2 = ent2_cutattr

# attr_all1 = ent1_allattr
# attr_all2 = ent2_allattr

# attr_all1 = ent1_v
# attr_all2 = ent2_v

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    for i in tqdm(range(len(hard_list)), desc="querying:"):
        # print(ent1[e1[i]])
        # rank_entities = [ent2.get(k) for k in candidate[i]]
        rank_entities = [ent2_dic[k] for k in hard_topk[i]]
        
        # rank_dict = enumerate(rank_entities)
        # print(rank_entities)
        
        answer_all = []
        for can_e in rank_entities:
            
            if ent1_dic[hard_list[i]] in attr_all1:
                attr1 = str(set(attr_all1[ent1_dic[hard_list[i]]]))
            else:
                attr1 = "" 
            if can_e in attr_all2:
                attr2 = str(set(attr_all2[can_e]))
            else:
                attr2 = ""
            
            if len(attr1) > 50:
                attr1 = attr1[:50]
            if len(attr2) > 50:
                attr2 = attr2[:50]
            dialogs: List[Dialog] = [

                [
                  {"role": "user", "content": "I am going to give two entities with attribute information."},
                    {"role": "assistant",
                        "content": "This is the entity: " + str(ent1_dic[hard_list[i]]) + ". And there are its attribute informations, " + attr1 +". This is another entity: " + str(can_e) + ". And its attribute informations, " + attr2 +"."},
                    {"role": "user", "content": " First, compare the entity names, then compare the attribute information. According to the above information determine whether the two entities are describe same entity. Just answer me yes or no." }, 
                    # {"role": "user", "content": "According to the above information determine whether the two entities are the same. Just answer me yes or no." },
                    # Just answer me yes or no.
                    # {"role": "user", "content": "Entity: "+ str(ent1_dic[hard_list[i]]) + "and entity: " + str(can_e) + "are same entity or not. Just answer me yes or no."},
                    # {"role": "assistant",
                    #     "content": "This is the entity: " + str(ent1_dic[hard_list[i]]) + ". And there are its attribute informations, " + attr1 +". This is another entity: " + str(can_e) + ". And its attribute informations, " + attr2 +"."},
                    # {"role": "user", "content": " First, compare the entity names, then compare the attribute information. According to the above information determine whether the two entities are describe same entity. Just answer me yes or no." }, 
                ],
            ]
            results = generator.chat_completion(
                dialogs,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for dialog, result in zip(dialogs, results):
                # for msg in dialog:
                #     print(f"{msg['role'].capitalize()}: {msg['content']}\n")

                answer = f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                
                # print(answer)
                # print(answer.split('\n'))
                # print(
                #     f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                # )
                # print("\n==================================\n")
            answer_all.append(answer.split(":")[-1])
        # print(answer_all)
        write_data_2file(fold + lang + "llama3_results_allattr_step_50", answer_all)


if __name__ == "__main__":
    fire.Fire(main)
