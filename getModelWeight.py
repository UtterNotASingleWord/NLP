import random


import keras
from keras import Sequential, Input, Model
from keras.layers import Bidirectional, Embedding, LSTM, Dense, Dropout
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from keras.regularizers import l2
np.random.seed(10)
import re
re_tag = re.compile(r'<[^>]+>')
def rm_tags(text):
    return re_tag.sub('', text)
import os
def read_files(filetype):
    path = "data/C50/" + filetype
    file_list = []
    for f in os.listdir(path):
        for file in os.listdir(path + f):
            txt_path = path + f + "/" + file
            file_list += [txt_path]
    all_texts=[]
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
    all_labels=[]
    for i in range(50):
        all_labels+=([i]*50)
    return all_labels, all_texts
def get_dataC50All(filetype):
    path = "data/C50/"+filetype
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
        with open(file_list[i + 50], encoding='utf8') as file_input:
            all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
    all_labels = ([1] * int(len(file_list)/2) + [0] * int(len(file_list)/2))
    randomNum=random.randint(0,2499)
    random.seed(randomNum)
    random.shuffle(all_textsOne)
    random.seed(randomNum)
    random.shuffle(all_textsTwo)
    random.seed(randomNum)
    random.shuffle(all_labels)
    return all_labels, all_textsOne, all_textsTwo


def get_dataC50AllTest(filetype):
    path = "data/C50/"+filetype
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
        with open(file_list[i + 50], encoding='utf8') as file_input:
            all_textsTwo += [rm_tags(" ".join(file_input.readlines()))]
    all_labels = ([1] * int(len(file_list)/2) + [0] * int(len(file_list)/2))
    randomNum=random.randint(0,2499)
    random.seed(randomNum)
    random.shuffle(all_textsOne)
    random.seed(randomNum)
    random.shuffle(all_textsTwo)
    random.seed(randomNum)
    random.shuffle(all_labels)
    return all_labels, all_textsOne, all_textsTwo
y_train, train_textOne,train_textTwo =get_dataC50All("C50train/")
# y_train=np_utils.to_categorical(y_trainOrign)
y_test, test_textOne,test_textTwo =get_dataC50AllTest("C50test/")
# y_test=np_utils.to_categorical(y_testOrign)
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

author_a = Input(shape=(1000, ))

author_b = Input(shape=(1000, ))
Embedding_layer=(Embedding(output_dim=32,
                    input_dim=10000,
                    name="Embedding",
                    input_length=1000))
Author_A=Embedding_layer(author_a)
Author_B=Embedding_layer(author_b)
Dropout_layer1=Dropout(0.2, name="Dropout1")
Dropout_layer1A=Dropout_layer1(Author_A)
Dropout_layer1B=Dropout_layer1(Author_B)
Bidirectional_layer=Bidirectional(LSTM(32),name="Bidirectional")
Bidirectional_layerA=Bidirectional_layer(Dropout_layer1A)
Bidirectional_layerB=Bidirectional_layer(Dropout_layer1B)
# 然后再连接两个向量：



Dense_layer=Dense(units=256,activation='relu',name="Dense1")
Dense_layerA=Dense_layer(Bidirectional_layerA)
Dense_layerB=Dense_layer(Bidirectional_layerB)
Dropout_layer2=Dropout(0.2,name="Dropout2")
Dropout_layer2A=Dropout_layer2(Dense_layerA)
Dropout_layer2B=Dropout_layer2(Dense_layerB)
merged_vector = keras.layers.concatenate([Dropout_layer2A, Dropout_layer2B], axis=-1)
predictionsDense = Dense(units=512, activation='tanh',kernel_regularizer=l2(0.01))(merged_vector)
predictionsDrop=Dropout(0.2)(predictionsDense)
predictions = Dense(units=1, activation='sigmoid',kernel_regularizer=l2(0.01))(predictionsDrop)
# 定义一个连接输入和预测的可训练的模型
model = Model(inputs=[author_a,author_b], outputs=predictions)
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.load_weights('model/author_model.h5',by_name=True)
early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=1,
                              restore_best_weights=True,
                              verbose=0, mode='auto')
train_history=model.fit([x_trainOne,x_trainTwo],y_train, shuffle=True,batch_size=100,epochs=20,verbose=2,validation_split=0.01,callbacks =[early_stopping])
# validation_split=0.01,callbacks = [early_stopping])

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[22]:
show_train_history(train_history, 'acc', 'val_acc')
# In[23]:
show_train_history(train_history, 'loss', 'val_loss')
# # 评估模型的准确率
scores = model.evaluate([x_testOne,x_testTwo], y_test, verbose=1)
print(scores[1])
SentimentDict = {1: '是', 0: '否'}
probility = model.predict([x_testOne,x_testTwo])
def display_test_Sentiment(i):
    print(test_textOne[i])
    print("***********************************************")
    print(test_textTwo[i])
    probility = model.predict([[x_testOne[i]], [x_testTwo[i]]])
    print(probility)
    predict = np.argmax(probility, axis=0)
    print('标签label:', SentimentDict[y_test[i]], '预测结果:', SentimentDict[predict[0]])

display_test_Sentiment(2)


display_test_Sentiment(10)
# count=1
# for i in probility:
#     print(i)
#     # if(i>0.5):
#     #     print(str(i)+"----------------   1")
#     # else:
#     #     print(str(i)+"----------------   0")

model_json = model.to_json()
with open("model/author_Test_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model/author_Test_model.h5")
print("Saved model to disk")
