# -*- coding: utf-8 -*-
import sys
#sys.path.append('c:\\python27\\lib\\site-packages') # python系统路径
import os
import re
import jieba
import string
import logging
import jieba.analyse
from time import time
import numpy as np
import pandas as pd
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report
le = preprocessing.LabelEncoder()
rekon = re.compile('\s+')
regex = re.compile("[^\u4e00-\u9f5aa-zA-Z0-9]")
stopwords = {}.fromkeys([line.rstrip() for line in open('../config data/stopwords.txt','r')]) # 停用词文件路径
train_path=""
test_path=""
vali_path=""
# train
corpus = [] #train set
fileNameCorpus = []
category_list = [] #train 类别列表
# test
test_corpus=[]
test_fileNameCorpus=[]

test_fileName=[]
train_hashList= []
test_hashList= []
delEStr = string.punctuation + ' ' + string.digits

TrainingConfig = ''

pipe_body = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.8, max_features=4000)),
    ('scaler', MaxAbsScaler()),
    ('chi2', SelectKBest(chi2, k=3700)),
    ('svc', SVC(kernel='linear', gamma=0.005, C=0.5, probability=True))
])
pipe_title = Pipeline([
    ('tfidf', TfidfVectorizer(binary=True, max_features=800)),
    ('mnb', MultinomialNB(alpha=0.0001))
])

## 内部函数
# 文本内容处理
def DocProcess(contents):
    wordsList = []
    contents = contents
    contents = regex.sub(' ',contents) # 只保留中文英文
    contents = rekon.sub(' ',contents) # 多空格变一个空格
    lenCon = len([word for word in jieba.cut(contents, cut_all=False) if word not in stopwords])
    chineseCut = [word for word in jieba.analyse.extract_tags
                  (contents, topK=int(lenCon*0.5), withWeight=False, allowPOS=())
                  if word not in stopwords]
    contentCut = [word for word in chineseCut if ((word != ' ') and not(word.isdigit())) ] # 排除空格和纯数字
    return contentCut
# 文件名处理
def fileNameProcess(contents):
    contents = contents
    contents = contents[:contents.index('.txt')]
    fileName = [word for word in jieba.cut(contents, cut_all=False)]
    return fileName
# 循环导入训练文件

def train_preprocess():
    print("Prepare loading train data")
    category = os.listdir(train_path)
    for cate in category:
        starttime = time();n=0
        filesList = os.listdir(os.path.join(train_path,cate))      # 这个类别内所有文件列表
        for fileName in filesList:
            contents = open(os.path.join(train_path,cate,fileName),'rb').read().decode('utf-8')
            if len(re.sub(r'\n','',contents))<10: # 空文本
                print ('Find '+fileName+' is empty or the length of file is too short, and will be removed!');continue
            if hash(contents) not in train_hashList:
                DocProcessed = DocProcess(contents)
                corpus.append(' '.join(DocProcessed))
                fileNameCorpus.append(' '.join(fileNameProcess(fileName)))
                category_list.append(cate)
                n = n+1
                train_hashList.append(hash(contents))
            else:
                print ('Find '+fileName+' is duplication ，it has been removed!')
        endtime = time()
        print ('训练集密级类别为（%s）有文件有:%d 个。导入用时: %.3f' % (cate,n,endtime-starttime))
    print("loading train set successfully!")

def test_preprocess():
    print("prepare loading test data")
    starttime = time();n = 0
    test_file = os.listdir(test_path)
    for fileName in test_file:
        contents = open(os.path.join(test_path, fileName), 'rb').read().decode('utf-8')
        if len(re.sub(r'\n', '', contents)) < 10:  # 空文本
            print('Find ' + fileName + ' is empty or the length of file is too short, and will be removed!')
            continue
        if hash(contents) not in test_hashList:
            DocProcessed = DocProcess(contents)
            test_corpus.append(' '.join(DocProcessed))
            test_fileNameCorpus.append(' '.join(fileNameProcess(fileName)))
            n = n + 1
            test_hashList.append(hash(contents))
            test_fileName.append(fileName)
        else:
            print('Find ' + fileName + ' is duplication ，it has been removed!')
    endtime = time()
    print('测试数据文件有:%d 个，导入用时: %.3f' % (n, endtime - starttime))
    print("loading test set successfully")
    return test_fileName

# 接口函数
# set training data, input file path, output bool
def SetTrainingData():
    train_path = "../train data" # 训练文件夹路径
    test_path="../test data" #　测试文件夹路径
    return train_path,test_path

# set config, input parameters, output bool
def SetTrainingConfig(TrainingConfig):
    TrainingConfig=TrainingConfig

def StartTraining():
    train_preprocess()
    X = np.asarray(corpus)
    label = le.fit_transform(category_list)
    y = np.asarray(label)
    X_fn = np.asarray(fileNameCorpus)
    # training
    pipe_body.fit(X, y)  # fit body
    pipe_title.fit(X_fn, y)  # fit title
    # save features
    feature = pipe_body.named_steps['tfidf'].get_feature_names()
    feature_names = pd.Series([feature[i] for i in pipe_body.named_steps['chi2'].get_support(indices=True)])
    fh = open('feature.txt', 'w')
    fh.write(','.join(feature_names))
    fh.close()
def GetDocType():
    fileName = test_preprocess()
    X_test = np.asarray(test_corpus)
    X_fn_test = np.asarray(test_fileNameCorpus)
    y_body_proba = pipe_body.predict_proba(X_test)  # predict probality
    y_title_proba = pipe_title.predict_proba(X_fn_test)
    y_proba = y_body_proba * 0.7 + y_title_proba * 0.3
    y_pre = np.argmax(y_proba, axis=1)
    print(y_proba)
    y_pre_label = le.inverse_transform(y_pre)
    result = pd.DataFrame(dict(file_name=fileName, predict=y_pre_label))
    result.to_csv('../prediction data/predict.csv')
# 测试 函数
def GetTrainingData():
    test_path=''
    return 0
    # print('precision,recall,F1-score如下：\n\t 备注： \t0：内部 \t1：机密 \t2：秘密\n')
    # print (classification_report(y, y_pre))

def ClearTraining():
    return 0

if __name__=="__main__" :
    train_path, test_path = SetTrainingData()
    #SetTrainingConfig()
    StartTraining()
    GetDocType()
    #GetTrainingData()
    ClearTraining()
