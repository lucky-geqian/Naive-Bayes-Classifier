# -*- coding:utf-8 -*-
import os
import time
import random
import jieba
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import pickle
from NBC import MakeWordsSet, words_dict

def TextProcessing(folder_path):
    # folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 类间循环
    with open(folder_path, 'r') as f1:
        for line in f1:
            tks = line.split('\t', 1)
            class_list.append(tks[0])
            raw = tks[1].strip()
            word_cut = jieba.cut(raw, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
            data_list.append(word_list)

    # for folder in folder_list:
    #     new_folder_path = os.path.join(folder_path, folder)
    #     files = os.listdir(new_folder_path)
    #     # 类内循环
    #     j = 1
    #     for file in files:
    #         if j > 100:  # 每类text样本数最多100
    #             break
    #         with open(os.path.join(new_folder_path, file), 'r', encoding='UTF-8') as fp:
    #             for raw in fp.readlines():
    #                 raw = raw.strip()
    #                 # print (raw)
    #                 # --------------------------------------------------------------------------------
    #                 # jieba分词
    #                 # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
    #                 word_cut = jieba.cut(raw, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
    #                 word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
    #                 # jieba.disable_parallel() # 关闭并行分词模式
    #                 # print(word_list)
    #                 ## --------------------------------------------------------------------------------
    #                 data_list.append(word_list)
    #                 j += 1
    #     # print(j)
    # random.shuffle(data_list)
    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in data_list:
        for word in word_list:
            if all_words_dict.__contains__(word):
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list))[0]

    return all_words_list, data_list, class_list

def TextFeatures(data_list, feature_words, flag='sklearn'):
    def text_features(text, feature_words):
        text_words = set(text)
        ## -----------------------------------------------------------------------------------
        if flag == 'nltk':
            ## nltk特征 dict
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features
    feature_list = [text_features(text, feature_words) for text in data_list]
    return feature_list

if __name__ == '__main__':
    #文本处理
    folder_path = "F:/github/Naive-Bayes-Classifier/scripts/test_data.txt"
    all_words_list, data_list, class_list = TextProcessing(folder_path)

    # 生成stopwords_set
    stopwords_file = 'F:/github/Naive-Bayes-Classifier/stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    ## 文本特征提取和分类
    # feature_words = words_dict(all_words_list, 0, stopwords_set)
    feature_words = []
    with open('F:/github/Naive-Bayes-Classifier/Database/SogouC/feature_words.txt', 'r') as fb:
        for raw in fb.readlines():
            raw = raw.strip('\n')
            feature_words.append(raw)

    # 加载模型(sklearn)
    with open("F:/github/Naive-Bayes-Classifier/Database/SogouC/model/nbc_classifier.pickle", 'rb') as f:
        classifier = pickle.load(f)
    feature_list = TextFeatures(data_list, feature_words)
    # print(classifier.predict(feature_list))
    test_accuracy = classifier.score(feature_list, class_list)
    print(test_accuracy)
    # for test_feature in feature_list:
    #     # print(test_feature)
    #     for i in data_list:
    #         print(i)
    #         print(classifier.predict(np.asarray(test_feature).reshape(1, -1)))

