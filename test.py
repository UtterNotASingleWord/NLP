

# np.random.seed(10)
# import re
# re_tag = re.compile(r'<[^>]+>')
# def rm_tags(text):
#     return re_tag.sub('', text)
# import os
# #
# path = "data/C50/C50train/"
# file_list=[]
# index=0
# for f in os.listdir(path):
#     for file in os.listdir(path+f):
#         txt_path=path+f+"/"+file
#         file_list += [txt_path]
# count = 1
# all_textsOne = []
# all_textsTwo = []
# for fi in file_list:
#     with open(fi, encoding='utf8') as file_input:
#         if count % 2 == 0:
#             all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
#             count = count + 1
#         else:
#             all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
#             count = count + 1
# for i in range(int(file_list/2)):
#     with open(file_list[i], encoding='utf8') as file_input:
#         all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
#     with open(file_list[i+50], encoding='utf8') as file_input:
#         all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
# all_labels = ([1] * int(file_list/2) + [0] * int(file_list/2))
# def read_files(filetype):
#     path = "data/C50/" + filetype
#     file_list = []
#     for f in os.listdir(path):
#         for file in os.listdir(path + f):
#             txt_path = path + f + "/" + file
#             file_list += [txt_path]
#     all_texts=[]
#     for fi in file_list:
#         with open(fi, encoding='utf8') as file_input:
#             all_texts += [rm_tags(" ".join(file_input.readlines()))]
#     all_labels=[]
#     for i in range(50):
#         all_labels+=([i]*50)
#     return all_labels, all_texts
#
# y_trainOrign, train_text = read_files("C50train/")
# y_train=np_utils.to_categorical(y_trainOrign,num_classes=50)
# print(y_train[2499])
# def get_dataC50All(filetype):
#     path = "dataset/C50/"+filetype
#     file_list = []
#     for f in os.listdir(path):
#         for file in os.listdir(path + f):
#             txt_path = path + f + "/" + file
#             file_list += [txt_path]
#     count = 1
#     all_textsOne = []
#     all_textsTwo = []
#     count=0
#     for fi in file_list:
#         with open(fi, encoding='utf8') as file_input:
#             sentence=rm_tags(" ".join(file_input.readlines()))
#             sentenceLine=sentence.strip().split("t")
#             for i in range(len(sentenceLine)):
#                 # words=nltk.word_tokenize(sentenceLine[i].lower())
#                 maxlen=len(sentenceLine[i])
#                 if maxlen>count:
#                    count=maxlen
            # if count % 2 == 0:
            #     all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
            #     count = count + 1
            # else:
            #     all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
            #     count = count + 1
    # print(count)
    # return count
    # for i in range(int(len(file_list) / 2)):
    #     with open(file_list[i], encoding='utf8') as file_input:
    #         all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
    #     with open(file_list[i + 70], encoding='utf8') as file_input:
    #         all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
    # all_labels = ([1] * int(len(file_list)/2) + [0] * int(len(file_list)/2))
    # # print(all_labels)
    # return all_textsOne,all_textsTwo


# train_textOne,train_textTwo=get_dataC50All("C50train/")
# token = Tokenizer(num_words=10000)
# token.fit_on_texts(train_textOne)
# token.fit_on_texts(train_textTwo)
# word2id=token.word_index
# id2word={v:k for k,v in word2id.items()}
# print(id2word)
# get_dataC50All("C50train/")
import csv
import random

import  pandas as pd
# def get_dataC50WritingStyle(filetype):
#     path = filetype
#     all_inputOne = []
#     all_inputTwo = []
#     all_input=[]
#     reader = csv.reader(open(filetype))
#     count=1
#     index=1
#     for i in reader:
#         if count==1:
#             count=count+1
#             continue
#         else:
#             all_input+=[i[2:13]]
#             if index %2==0:
#                 all_inputOne+=[i[2:13]]
#                 index=index+1
#             else:
#                 all_inputTwo+=[i[2:13]]
#                 index=index+1
#     print(len(all_input))
#     for i in range(int(len(all_input))/2):
#         all_inputOne+=[all_input[i]]
#         all_inputTwo+=[all_input[i+50]]
#     all_labels = ([1] * int(len(all_input) / 2) + [0] * int(len(all_input) / 2))
#     randomNum = random.randint(0, 2499)
#     random.seed(randomNum)
#     random.shuffle(all_inputOne)
#     random.seed(randomNum)
#     random.shuffle(all_inputTwo)
#     random.seed(randomNum)
#     random.shuffle(all_labels)
#     return all_labels, all_inputOne,all_inputTwo








#
# get_dataC50WritingStyle("train.csv")
from keras.utils import np_utils


def get_dataC50WritingStyle(filetype):
    path = filetype
    all_input=[]
    all_labels=[]
    reader = csv.reader(open(filetype))
    count=1
    for i in reader:
        if count==1:
            count=count+1
            continue
        else:
            all_input+=[i[2:15]]
    for i in range(50):
        all_labels += ([i] * 50)
    all_labels = np_utils.to_categorical(all_labels, num_classes=50)
    for i in all_labels:
        print(i)
    return all_labels,all_input

get_dataC50WritingStyle("trainData.csv")