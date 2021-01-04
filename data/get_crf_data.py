# -*- coding: utf-8 -*-
# @Time : 2020/12/30 15:05
# @Author : Jclian91
# @File : get_crf_data.py
# @Place : Yangpu, Shanghai
with open("conll2003.test", "r", encoding="utf-8") as f:
    content = [_.strip() for _ in f.readlines()]

g = open("crf_english_ner.test", "w", encoding="utf-8")
for line in content:
    if line:
        char = line.split()[0]
        pos = line.split()[-2]
        tag = line.split()[-1]
        g.write("{}\t{}\t{}\n".format(char, pos, tag))
    else:
        g.write("\n")

g.close()