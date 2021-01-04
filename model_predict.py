# -*- coding: utf-8 -*-
# @Time : 2020/12/24 13:28
# @Author : Jclian91
# @File : model_predict.py
# @Place : Yangpu, Shanghai
import numpy as np
from pprint import pprint
from keras.models import load_model
from keras_bert import get_custom_objects
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

from util import event_type
from model_train import PreProcessInputData, id_label_dict
from load_data import bert_encode


# 加载训练好的模型
custom_objects = get_custom_objects()
for key, value in {'CRF': CRF, 'crf_loss': crf_loss, 'crf_accuracy': crf_accuracy}.items():
    custom_objects[key] = value
model = load_model("%s_large_ner.h5" % event_type, custom_objects=custom_objects)


# 测试句子
def get_text_predict(text):
    new_text = []
    for word in text.split():
        new_text.extend(bert_encode(word))
    word_labels, seq_types = PreProcessInputData([new_text])

    # 模型预测
    predicted = model.predict([word_labels, seq_types])
    y = np.argmax(predicted[0], axis=1)
    tags = [id_label_dict[_] for _ in y]

    # 输出预测结果
    real_tag = []
    i = 1
    for word in text.split():
        new_word = bert_encode(word)
        if i < len(tags):
            real_tag.append(tags[i])
            i += len(new_word)

    return real_tag


if __name__ == '__main__':
    test_text = "South Africa - 15 - Andre Joubert , 14 - Justin Swart , 13 - Japie Mulder ( Joel Stransky , 48 mins ) 12 - Danie van Schalkwyk , 11 - Pieter Hendriks ; 10 - Henry Honiball , 9 - Joost van der Westhuizen ; 8 - Gary Teichmann ( captain ) , 7 - Andre Venter ( Wayne Fyvie , 75 ) , 6 - Ruben Kruge , 5 - Mark Andrews ( Fritz van Heerden , 39 ) , 4 - Kobus Wiese , 3 - Marius Hurter , 2 - James Dalton , 1 - Dawie Theron ( Garry Pagel , 66 ) ."
    print(get_text_predict(test_text))

