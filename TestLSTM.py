import keras
from keras import Input, Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)
import re
re_tag = re.compile(r'<[^>]+>')
def rm_tags(text):
    return re_tag.sub('', text)
import os
def read_files(filetype):
    path = "data/"
    file_list = []
    same_path = path + filetype + "/same/"
    for f in os.listdir(same_path):
        file_list += [same_path + f]
    different_path = path + filetype + "/different/"
    for f in os.listdir(different_path):
        file_list += [different_path + f]
    print('read', filetype, 'files:', len(file_list))
    all_labels = ([1] * 4+ [0] * 4)
    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:

            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels, all_texts
def get_dataRe(filetype):
    path = "data/"
    same_path = path + filetype + "/same/"
    different_path = path + filetype + "/different/"
    all_textsOne = []
    all_textsTwo = []
    file_list = []
    for f in os.listdir(same_path):
        file_list += [same_path + f]
    for f in os.listdir(different_path):
        file_list += [different_path + f]
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            if int(fi[-5])%2==0:
                all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
            else:
                all_textsTwo+= [rm_tags(" ".join(file_input.readlines()))]
    all_labels = ([1] * int(len(all_textsOne)/2) + [0] * int(len(all_textsTwo)/2))
    return all_labels,  all_textsOne, all_textsTwo

# def get_dataC50All(filetype):
#     path = "data/C50/"+filetype
#     file_list = []
#     for f in os.listdir(path):
#         for file in os.listdir(path + f):
#             txt_path = path + f + "/" + file
#             file_list += [txt_path]
#     count = 1
#     all_textsOne = []
#     all_textsTwo = []
#     x_valOne = []
#     x_valTwo = []
#     for fi in file_list:
#         with open(fi, encoding='utf8') as file_input:
#             if count % 2 == 0:
#                 all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
#                 count = count + 1
#             else:
#                 all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
#                 count = count + 1
#     same_label=len(all_textsOne)
#     for i in range(int(len(file_list) / 2)):
#         with open(file_list[i], encoding='utf8') as file_input:
#             all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
#         with open(file_list[i + 50], encoding='utf8') as file_input:
#             all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
#     for i in range(100):
#         x_valOne += [all_textsOne[-1-i]]
#         all_textsOne.pop(-1-i)
#         x_valTwo += [all_textsTwo[-1-i]]
#         all_textsTwo.pop(-1-i)
#     differnt_label=len(all_textsOne)-same_label
#     all_labels = ([1] * same_label + [0] * differnt_label)
#     y_val =([1] * 100 + [0] * 100)
#     return all_labels, all_textsOne, all_textsTwo,x_valOne,x_valTwo,y_val

def get_dataC50All(filetype):
    path = "data/C50/"+filetype
    file_list = []
    index = 0
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
        with open(file_list[i + 50], encoding='utf8') as file_input:
            all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
    all_labels = ([1] * int(len(file_list)/2) + [0] * int(len(file_list)/2))
    return all_labels, all_textsOne, all_textsTwo

def get_dataReC50(filetype):
    path = "data/"
    same_path = path + filetype + "/same/"
    different_path = path + filetype + "/different/"
    all_textsOne = []
    all_textsTwo = []
    file_list = []
    AaronPressman_path = path + filetype + "/same/" + "AaronPressman/"
    AlanCrosby_path = path +  filetype + "/same/" + "AlanCrosby/"
    AlexanderSmith_path= path +  filetype + "/same/" + "AlexanderSmith/"
    BenjaminKangLim_path=path +  filetype + "/same/" + "BenjaminKangLim/"
    AaronPressman = []
    AlanCrosby = []
    AlexanderSmith=[]
    BenjaminKangLim=[]
    all_textsOne = []
    all_textsTwo = []
    file_list = []
    for f in os.listdir(AaronPressman_path):
        AaronPressman += [AaronPressman_path + f]
    for f in os.listdir(AlanCrosby_path):
        AlanCrosby += [AlanCrosby_path + f]
    for f in os.listdir(AlexanderSmith_path):
        AlexanderSmith += [AlexanderSmith_path + f]
    for f in os.listdir(BenjaminKangLim_path):
        BenjaminKangLim += [BenjaminKangLim_path + f]
    for file in AaronPressman:
        file_list += [file]
    for file in AlanCrosby:
        file_list += [file]
    for file in AlexanderSmith:
        file_list += [file]
    for file in BenjaminKangLim:
        file_list += [file]
    count = 1
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            if count % 2 == 0:
                all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
                count = count + 1
            else:
                all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
                count = count + 1
    for i in range(50):
        with open(AaronPressman[i], encoding='utf8') as file_input:
            all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
        with open(AlanCrosby[i], encoding='utf8') as file_input:
            all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
        with open( AlexanderSmith[i], encoding='utf8') as file_input:
            all_textsOne += [rm_tags(" ".join(file_input.readlines()))]
        with open(BenjaminKangLim[i], encoding='utf8') as file_input:
            all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]

    all_labels = ([1] * 100 + [0] * 100)
    return all_labels,  all_textsOne, all_textsTwo


def get_data(filetype):
    path = "data/"
    same_path = path + filetype + "/same/"
    different_path = path + filetype + "/different/"
    all_texts = [[[]] * 2 for i in range(4)]
    all_labels = ([1] * 2 + [0] * 2)
    row = 0
    for i in range(1, 4, 2):
        file = open(same_path + "test" + str(i) + ".txt", encoding='utf8')
        temp =[rm_tags(" ".join(file.readlines()))]
        all_texts[row][0] = temp
        count = i + 1
        file = open(same_path + "test" + str(count) + ".txt", encoding='utf8')
        temp2 = [rm_tags(" ".join(file.readlines()))]
        all_texts[row][1] = temp2
        row = row + 1

    for i in range(1, 4, 2):
        file = open(different_path + "test" + str(i) + ".txt", encoding='utf8')
        temp = [rm_tags(" ".join(file.readlines()))]
        all_texts[row][0] = temp
        count = i + 1
        file = open(different_path + "test" + str(count) + ".txt", encoding='utf8')
        temp2 = [rm_tags(" ".join(file.readlines()))]
        all_texts[row][1] = temp2
        row = row + 1

    return all_labels, all_texts


# y_train, train_textOne,train_textTwo = get_dataReC50("train")
# y_test, test_textOne,test_textTwo = get_dataReC50("test")
y_train, train_textOne,train_textTwo = get_dataC50All("C50train/")
y_test, test_textOne,test_textTwo = get_dataC50All("C50test/")
# 先读取所有文章建立字典，限制字典的数量为nb_words=2000
token = Tokenizer(num_words=10000)
token.fit_on_texts(train_textOne)
token.fit_on_texts(train_textTwo)
x_train_seqOne=token.texts_to_sequences(train_textOne)
x_train_seqTwo=token.texts_to_sequences(train_textTwo)
x_test_seqOne = token.texts_to_sequences(test_textOne)
x_test_seqTwo = token.texts_to_sequences(test_textTwo)
x_trainOne = sequence.pad_sequences(x_train_seqOne, maxlen=1000)
x_trainTwo = sequence.pad_sequences(x_train_seqTwo, maxlen=1000)
x_testOne= sequence.pad_sequences(x_test_seqOne, maxlen=1000)
x_testTwo= sequence.pad_sequences(x_test_seqTwo, maxlen=1000)
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
input_1 = Input(shape=(1000,),name="input_1")
input_2 = Input(shape=(1000,),name="input_2")
inputs = keras.layers.concatenate([input_1, input_2])
Embedding_layer=(Embedding(output_dim=256,
                    input_dim=10000,
                    input_length=2000))(inputs)
dropout_layer=Dropout(0.2)(Embedding_layer)
LSTM_layer= LSTM(256)(dropout_layer)
# model.add(Dropout(0.2))
dropout_layer1=Dropout(0.2)(LSTM_layer)
Dense_layer= Dense(256, activation='relu')(dropout_layer1)
Dense_layer1= Dense(256, activation='relu')(Dense_layer)
# In[16]:
# model.add(Dense(units=256,
#                 activation='relu'))
# model.add(Dropout(0.2))
# In[17]:
# model.add(Dense(units=1,
#                 activation='sigmoid'))
dropout_layer2=Dropout(0.2)(Dense_layer1)
output_layer=Dense(1, activation='sigmoid', name='outputs')(dropout_layer2)
model=Model(inputs=[input_1, input_2], outputs=[output_layer])
model.summary()
model.compile(loss='binary_crossentropy',
              # optimizer='rmsprop',
              optimizer='adam',
              metrics=['accuracy'])
train_history = model.fit([x_trainOne,x_trainTwo],y_train, batch_size=4, #batchsize 修改
                          epochs=5, verbose=2)
                          # validation_split=0.2))
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()



show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')
scores = model.evaluate([x_testOne,x_testTwo], y_test, verbose=1)
print(scores)
print(scores[0])
probility = model.predict([x_testOne,x_testTwo])
predict=np.argmax(probility,axis=1)
predict=model.predict([x_testOne,x_testTwo])
SentimentDict = {1: '是', 0: '否'}
#
#
def display_test_Sentiment(i):
    print(test_textOne[i])
    print(test_textTwo[i])
    probility = model.predict([[x_testOne[i]], [x_testTwo[i]]])
    predict = np.argmax(probility, axis=0)
    print('标签label:', SentimentDict[y_test[i]], '预测结果:', SentimentDict[predict[0]])

display_test_Sentiment(2)


def predict_review(input_textone,input_texttwo):
    input_seqOne = token.texts_to_sequences([input_textone])
    input_seqTwo = token.texts_to_sequences([input_texttwo])
    pad_input_seqOne = sequence.pad_sequences(input_seqOne, maxlen=1000)
    pad_input_seqTwo = sequence.pad_sequences(input_seqTwo, maxlen=1000)
    predict_result = model.predict([pad_input_seqOne,pad_input_seqTwo])
    predicttest = np.argmax(predict_result, axis=1)
    print("--------------------------------")
    print(SentimentDict[predicttest[0]])


predict_review(
["He turned away to descend; then, irresolute, faced round to her door again. In the act he caught sight of one of the d'Urberville dames, whose portrait was immediately over the entrance to Tess's bedchamber. In the candlelight the painting was more than unpleasant. Sinister design lurked in the woman's features, a concentrated purpose of revenge on the other sex - so it seemed to him then. The Caroline bodice of the portrait was low - precisely as Tess's had been when he tucked it in to show the necklace; and again he experienced the distressing sensation of a resemblance between them."],
["His air remained calm and cold, his small compressed mouth indexing his powers of self-control; his face wearing still that terribly sterile expression which had spread thereon since her disclosure. It was the face of a man who was no longer passion's slave, yet who found no advantage in his enfranchisement. He was simply regarding the harrowing contingencies of human experience, the unexpectedness of things. Nothing so pure, so sweet, so virginal as Tess had seemed possible all the long while that he had adored her, up to an hour ago; but"])

model_json = model.to_json()
with open("SaveModel/LSTM_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("SaveModel/LSTM_model.h5")
print("Saved model to disk")

