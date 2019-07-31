import json
import os
from keras.engine.saving import model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import re

import numpy as np
from keras.utils import np_utils

re_tag = re.compile(r'<[^>]+>')
def rm_tags(text):
    return re_tag.sub('', text)
def get_dataC50AllTest(filetype):
    path = "dataset/C50/"+filetype
    file_list = []
    for f in os.listdir(path):
        for file in os.listdir(path + f):
            txt_path = path + f + "/" + file
            file_list += [txt_path]
    count = 1
    all_textsOne = []
    all_textsTwo = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            if count % 2 == 0:
                all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
                count = count + 1
            else:
                all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
                count = count + 1
    for i in range(int(len(file_list) / 2)):
        with open(file_list[i], encoding='utf8') as file_input:
            all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
        with open(file_list[i + 30], encoding='utf8') as file_input:
            all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
    all_labels = ([1] * int(len(file_list)/2) + [0] * int(len(file_list)/2))
    return all_labels, all_textsOne, all_textsTwo
y_testOrign, test_textOne,test_textTwo =get_dataC50AllTest("C50test/")
y_test=np_utils.to_categorical(y_testOrign)
token = Tokenizer(num_words=40000)
x_test_seqOne = token.texts_to_sequences(test_textOne)
x_test_seqTwo = token.texts_to_sequences(test_textTwo)
x_testOne= sequence.pad_sequences(x_test_seqOne, maxlen=1000)
x_testTwo= sequence.pad_sequences(x_test_seqTwo, maxlen=1000)

json_file = open('model/author_Test_model.json', 'r')
loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights('model/author_Test_model.h5')
probility = model.predict([x_testOne,x_testTwo])
predict = np.argmax(probility, axis=1)
for i in probility:
    print(i)
    print(np.argmax(i, axis=0))








