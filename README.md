本项目采用Keras和Keras-bert实现英语序列标注，其中对BERT进行微调。

### 维护者

- jclian91

### 数据集

1. [Conll2003](https://www.clips.uantwerpen.be/conll2003/ner/)

    conll2003.train 14987条数据和conll2003.test 3466条数据，共4种标签：
    
    + [x] LOC
    + [x] PER
    + [x] ORG
    + [x] MISC
    
2. [wnut17](https://noisy-text.github.io/2017/emerging-rare-entities.html)

    wnut17.train 3394条数据和wnut17.test 1009条数据，共6种标签：
    
    + [x] Person
    + [x] Location (including GPE, facility)
    + [x] Corporation
    + [x] Consumer good (tangible goods, or well-defined services)
    + [x] Creative work (song, movie, book, and so on)
    + [x] Group (subsuming music band, sports team, and non-corporate organisations)

### 模型结构

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, None)         0
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, None)         0
__________________________________________________________________________________________________
model_2 (Model)                 multiple             108596736   input_1[0][0]
                                                                 input_2[0][0]                    
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, None, 200)    695200      model_2[1][0]
__________________________________________________________________________________________________
crf_1 (CRF)                     (None, None, 9)      1908        bidirectional_1[0][0]
==================================================================================================
Total params: 109,293,844
Trainable params: 109,293,844
Non-trainable params: 0
```

### 模型效果

- Conll2003

模型参数：uncased_L-12_H-768_A-12, MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
           precision    recall  f1-score   support

      PER     0.9650    0.9577    0.9613      1842
      ORG     0.8889    0.8770    0.8829      1341
     MISC     0.8156    0.8395    0.8274       922
      LOC     0.9286    0.9271    0.9278      1837

micro avg     0.9129    0.9116    0.9123      5942
macro avg     0.9134    0.9116    0.9125      5942
```

BERT模型评估结果对比

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=10.

|模型名称|P|R|F1|
|---|---|---|---|
|BERT-Small|0.8744|0.8859|0.8801|
|BERT-Medium|0.9052|0.9031|0.9041|
|BERT-Base|0.9129|0.9116|0.9123|

[最新SOTA结果的F1值为94.3%.](https://github.com/sebastianruder/NLP-progress/blob/master/english/named_entity_recognition.md)

- wnut17

模型参数：uncased_L-12_H-768_A-12, MAX_SEQ_LEN=128, BATCH_SIZE=20, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
             precision    recall  f1-score   support

       work     0.2069    0.0571    0.0896       105
     person     0.6599    0.4830    0.5577       470
    product     0.3333    0.0965    0.1497       114
   location     0.5070    0.4865    0.4966        74
      group     0.1500    0.1538    0.1519        39
corporation     0.1935    0.1765    0.1846        34

  micro avg     0.5328    0.3489    0.4217       837
  macro avg     0.5016    0.3489    0.4033       837
```


### 代码说明

0. 将BERT英语预训练模型放在对应的文件夹下
1. 运行load_data.py，生成类别标签文件label2id.json，注意O标签为0;
2. 所需Python第三方模块参考requirements.txt文档
3. 自己需要分类的数据按照data/conll2003.train和data/conll2003.test的格式准备好
4. 调整模型参数，运行model_train.py进行模型训练
5. 运行model_evaluate.py进行模型评估
6. 运行model_predict.py对新文本进行预测