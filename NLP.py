import  os
import re

from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'D:\NLPEnglish\stanford-corenlp-full-2018-10-05',lang='en')
re_tag = re.compile(r'()')
def rm_tags(text):
    return re_tag.sub('', text)
def get_dataC50All(filetype):
    fileName = 'test.csv'
    path = "data/C50/"+filetype
    file_list = []
    for f in os.listdir(path):
        for file in os.listdir(path + f):
            txt_path = path + f + "/" + file
            file_list += [txt_path]
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            sentencepath = fi.split("/")
            author = sentencepath[3]
            content = sentencepath[4]
            print(content)
            line = " ".join(file_input.readlines())
            sentence = "".join(line)
            sentence.strip()
            list_word = nlp.pos_tag(sentence)
            print(list_word)
            break
            Num = len(list_word)
            listword = []
            for i in list_word:
                if not i in listword:
                    listword.append(i)
            Nword = len(listword)
            abundantRate = Nword / Num
            num = sentence.split("\n")
            pargrapy = len(sentence) / len(num)
            numSentence = sentence.split(".")
            sentenceSize = len(sentence) / len(numSentence)
            count_dict = dict()
            for i in list_word:
                if i[1] in count_dict:
                    count_dict[i[1]] += 1
                else:
                    count_dict[i[1]] = 1
            DTsum = 0
            Nounsum = 0
            VBsum = 0
            JJsum = 0
            CDsum = 0
            Psum = 0
            RBsum = 0
            Tosum = 0
            INsum = 0
            Psum = 0
            WHsum = 0
            for (k, v) in count_dict.items():
                if k == "NN":
                    Nounsum = Nounsum + v
                elif k == "NNS":
                    Nounsum = Nounsum + v
                elif k == "NNP":
                    Nounsum = Nounsum + v
                elif k == "NNPS":
                    Nounsum = Nounsum + v
                elif k == "DT":
                    DTsum = DTsum + v
                elif k == "WDT":
                    DTsum = DTsum + v
                elif k == "VB":
                    VBsum = VBsum + v
                elif k == "VBD":
                    VBsum = VBsum + v
                elif k == "VBG":
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
                elif k == "WH":
                    WHsum = WHsum + v

get_dataC50All("C50test/")