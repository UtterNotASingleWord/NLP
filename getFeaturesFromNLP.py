from stanfordcorenlp import StanfordCoreNLP
import re
re_tag = re.compile(r'(\")')
def rm_tags(text):
    return re_tag.sub('', text)
nlp = StanfordCoreNLP(r'D:\NLP\stanford-corenlp-full-2018-10-05',lang='en')
f=open('D:/identifyAuthor/dataset/C50/C50train/AaronPressman/2537newsML.txt')
# line=f.readlines()
line=rm_tags(f.read())
# print(line)
sentence = "".join(line)
sentence.strip()
N=len(sentence)
print("全文长度为"+str(N))
list_word=nlp.pos_tag(sentence)
Num=len(list_word)
print("词汇数目"+str(Num))
listword=[]
for i in list_word:
     if not i in listword:
             listword.append(i)
Nword=len(listword)
print("词汇丰富度"+str(Nword))
count_dict = dict()
for i in listword:
    if i[1] in count_dict:
        count_dict[i[1]] += 1
    else:
        count_dict[i[1]] = 1
print(count_dict)
key_value = list(count_dict.keys())
print(key_value)
value=list(count_dict.values())
print(value)

# print(list_sign)
# print(nlp.word_tokenize(sentence))
#print(nlp.pos_tag(sentence))
# print(nlp.ner(sentence))
# print(nlp.parse(sentence))
# print(nlp.dependency_parse(sentence))
