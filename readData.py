import codecs
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import  csv
np.random.seed(10)
nlp = StanfordCoreNLP(r'D:\NLPEnglish\stanford-corenlp-full-2018-10-05',lang='en')
import re
import os
stopwordslist = [line.strip() for line in open ('stopwords.txt',encoding='utf-8').readlines()]
# print(stopwordslist)

re_tag = re.compile(r'(\")')
def rm_tags(text):
    return re_tag.sub('', text)
def get_dataC50All(filetype):
    fileName = 'train1Data.csv'
    path = "data/C50/"+filetype
    file_list = []
    for f in os.listdir(path):
        for file in os.listdir(path + f):
            txt_path = path + f + "/" + file
            file_list += [txt_path]
    with codecs.open(fileName, 'w', 'utf-8') as csvfile:
        # 指定 csv 文件的头部显示项
        filednames = ['author', 'content', 'lxical richness', 'segement length', 'sentence length','stopwords',
                          'Nounword', 'Verbword', 'Adjectiveword', 'determinerword', 'numeral', 'adverb',
                          'To_word', 'preposition', 'comma', 'fullStop', 'pronun', 'WH_pronun', 'questionMark',
                    'semicolon', 'exclamationMark']
        writer = csv.DictWriter(csvfile, fieldnames=filednames)
        writer.writeheader()
        for fi in file_list:
            with open(fi, encoding='utf8') as file_input:
                sentencepath=fi.split("/")
                author=sentencepath[3]
                content=sentencepath[4]
                line = rm_tags(" ".join(file_input.readlines()))
                sentence = "".join(line)
                sentence.strip()
                list_word = nlp.pos_tag(sentence)
                Num = len(list_word)
                sumStopWord=0
                for i in list_word:
                  if i[0] in stopwordslist:
                    sumStopWord=sumStopWord+1
                listword = []
                for i in list_word:
                    if not i in listword:
                        listword.append(i)
                Nword = len(listword)
                abundantRate=Nword/Num
                num =sentence.split("\n")
                pargrapy=len(sentence)/len(num)
                numSentence=sentence.split(".")
                sentenceSize=len(sentence)/len(numSentence)
                count_dict = dict()
                for i in list_word:
                    if i[1] in count_dict:
                        count_dict[i[1]] += 1
                    else:
                        count_dict[i[1]] = 1
                DTsum=0
                Nounsum=0
                VBsum=0
                JJsum=0
                CDsum=0
                Psum=0
                RBsum=0
                Tosum=0
                INsum=0
                Psum=0
                WHsum=0
                questionMark=0
                exclamationMark=0
                semicolon=0
                comma=0
                fullStop=0
                for (k, v) in count_dict.items():
                    if k=="NN":
                        Nounsum=Nounsum+v
                    elif k=="NNS":
                        Nounsum =Nounsum + v
                    elif k=="NNP":
                        Nounsum = Nounsum + v
                    elif k=="NNPS":
                        Nounsum =Nounsum + v
                    elif k=="DT":
                        DTsum=DTsum+v
                    elif k == "WDT":
                        DTsum = DTsum + v
                    elif k=="VB":
                        VBsum=VBsum+v
                    elif k=="VBD":
                        VBsum = VBsum + v
                    elif k=="VBG":
                        VBsum = VBsum + v
                    elif k == "VBN":
                        VBsum = VBsum + v
                    elif k == "VBP":
                        VBsum = VBsum + v
                    elif k == "VBZ":
                        VBsum = VBsum + v
                    elif k == "JJ":
                        JJsum = JJsum + v
                    elif k == "JJR":
                        JJsum = JJsum + v
                    elif k == "JJS":
                        JJsum = JJsum + v
                    elif k == "CD":
                        CDsum = CDsum + v
                    elif k == "P":
                        Psum = Psum + v
                    elif k == "RB":
                        RBsum = RBsum + v
                    elif k == "RBR":
                        RBsum = RBsum + v
                    elif k == "RBS":
                        RBsum = RBsum + v
                    elif k == "WRB":
                        RBsum = RBsum + v
                    elif k == "TO":
                        Tosum = Tosum + v
                    elif k == "IN":
                        INsum = INsum + v
                    elif k == "PN":
                        Psum = Psum + v
                    elif k=="WH":
                        WHsum=WHsum+v
                    elif k==",":
                        comma=comma+v
                    elif k=="?":
                        questionMark=questionMark+v
                    elif k==";":
                        semicolon=semicolon+v
                    elif k=="!":
                        exclamationMark=exclamationMark+v
                    elif k==".":
                        fullStop=fullStop+v
                    book = {
                    'author': author,
                    'content':content,
                     'lxical richness':abundantRate,
                    'segement length': pargrapy,
                    'sentence length':sentenceSize,
                    'stopwords':sumStopWord/Num,
                    'Nounword': Nounsum/Num,
                    'Verbword': VBsum/Num,
                    'Adjectiveword': JJsum/Num,
                    'determinerword':DTsum/Num,
                    'numeral':CDsum/Num,
                    'adverb':RBsum/Num,
                    'To_word':Tosum/Num,
                    'preposition':INsum/Num,
                    'comma':comma/Num,
                    'fullStop':fullStop/Num,
                    'pronun':Psum/Num,
                    'WH_pronun':WHsum/Num,
                    'questionMark':questionMark,
                    'semicolon':semicolon,
                    'exclamationMark':exclamationMark

                }
                try:
                    writer.writerow({'author': book['author'], 'content': book['content'],'lxical richness':book['lxical richness'],
                                     'segement length':book['segement length'],'sentence length':book['sentence length'],'stopwords':book['stopwords'],'Nounword':book['Nounword'],
                                     'Verbword':book['Verbword'], 'Adjectiveword':book[ 'Adjectiveword'], 'determinerword':book[ 'determinerword'],
                                     'numeral':book['numeral'], 'adverb':book[ 'adverb'], 'To_word':book[ 'To_word'], 'preposition':book ['preposition'],
                                     'comma': book[ 'comma'],  'fullStop': book['fullStop'],
                                     'pronun':book['pronun'],'WH_pronun':book['WH_pronun'],
                                     'questionMark': book['questionMark'],
                                     'semicolon': book['semicolon'],
                                     'exclamationMark': book['exclamationMark']
                                     })
                except UnicodeEncodeError:
                    print("编码错误, 该数据无法写到文件中, 直接忽略该数据")




get_dataC50All("C50train/")
