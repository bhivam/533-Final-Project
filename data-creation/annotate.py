import os
import logging
import json
from collections import defaultdict
from tqdm import tqdm
import spacy

nlp = spacy.load("en_core_web_sm")




def breakup_string(str):
    ignore = [","]
    loaded_str = nlp(str)
    broken_string = []
    for wrd in loaded_str:
        if wrd.lemma_ not in ignore:
            broken_string.append(wrd.lemma_)
    return broken_string

compressed_file = json.load(open("compressed_meeting_data.json", "r"))
inital_list = []
compressed_list = []
for input in compressed_file:
    inital_list.extend(input["text_list"])
    compressed_list.extend(input["compressed_list"])

labeled_file = {}
for count, input in enumerate(zip(inital_list, compressed_list)):
    org = breakup_string(input[0])
    comp = breakup_string(input[1])
    prev = 0
    end_index = 0
    org_size = len(org)
    labels = [False] * org_size
    org_set = set(org)
    for word in org:
        org_set.add(word)
    for word in comp:
        flag = False
        for i in range(150):
            right = min(org_size-1, prev+i)
            if (word == org[right]) and not labels[right]:
                labels[right] = True
                if right - prev > 75:
                    prev += 75
                else:
                    prev = right
                flag = True
                break
            right = max(0, prev - i)
            if (word == org[right])  and not labels[right]:
                labels[right] = True
                prev = right
                flag = True
                break
            
    
    retained_words = []
    for index, word in enumerate(org):
        if labels[index]:
            retained_words.append(word)
    retained_string = " ".join(retained_words)
    labeled_file[count] = { "original": input[0], "labels": labels, "text_to_keep": retained_string}

json.dump(labeled_file, open("label_word2.json", "w"), indent=4)









   
        


    