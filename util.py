# -*- coding: utf-8 -*-
# @Time : 2020/12/24 13:26
# @Author : Jclian91
# @File : util.py
# @Place : Yangpu, Shanghai

# 数据相关的配置
event_type = "conll2003"

train_file_path = "./data/%s.train" % event_type
test_file_path = "./data/%s.test" % event_type

# 模型相关的配置
BERT_MODEL_DIR = "wwm_uncased_L-24_H-1024_A-16"
MAX_SEQ_LEN = 128   # 输入的文本最大长度
BATCH_SIZE = 32     # 模型训练的BATCH SIZE
EPOCH = 20          # 模型训练的轮次
