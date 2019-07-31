import csv
import os
import random

from keras.engine import Layer
from keras.optimizers import RMSprop
from nltk.cluster import euclidean_distance
from stanfordcorenlp import StanfordCoreNLP
import keras
from keras import Sequential, Input, Model
from keras.layers import Bidirectional, Embedding, LSTM, Dense, Dropout, GRU, Average, GlobalAveragePooling2D, \
    AveragePooling2D, Lambda, K, Flatten
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.regularizers import l2
np.random.seed(10)
randomNum = random.randint(0, 2499)
import re
re_tag = re.compile(r'(\")')
def rm_tags(text):
    return re_tag.sub('', text)
def get_dataC50WritingStyle(filetype):
    path = filetype
    all_inputOne = []
    all_inputTwo = []
    all_input=[]
    reader = csv.reader(open(filetype))
    count=1
    index=1
    for i in reader:
        if count==1:
            count=count+1
            continue
        else:
            all_input+=[i[2:15]]
            if index %2==0:
                all_inputOne+=[i[2:15]]
                index=index+1
            else:
                all_inputTwo+=[i[2:15]]
                index=index+1
    for i in range(int(len(all_input)/2)):
        all_inputOne+=[all_input[i]]
        all_inputTwo+=[all_input[i+50]]
    random.seed(randomNum)
    random.shuffle(all_inputOne)
    random.seed(randomNum)
    random.shuffle(all_inputTwo)

    return all_inputOne,all_inputTwo

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
    all_labels = ([1] * int(len(file_list) / 2) + [0] * int(len(file_list) / 2))
    random.seed(randomNum)
    random.shuffle(all_textsOne)
    random.seed(randomNum)
    random.shuffle(all_textsTwo)
    random.seed(randomNum)
    random.shuffle(all_labels)
    return all_labels,all_textsOne,all_textsTwo
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def create_base_networkText():
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_one = Input(shape=(1000,))
    input_two=Input(shape=(13,))
    Embedding_layer = Embedding(len(word_index) + 1,
                                100,
                                weights=[embedding_matrix],
                                input_length=1000,
                                trainable=False)(input_one)
    GRU_layer = Bidirectional(GRU(64))(Embedding_layer)
    Dense_layer=Dense(units=64,activation='relu')(GRU_layer)
    merged_vector = keras.layers.concatenate([Dense_layer, input_two], axis=-1)
    DenseA = Dense(units=64, activation='relu')(merged_vector)
    resultOne = Dropout(0.1)(DenseA)
    result = Dense(128, activation='relu')(resultOne)
    return Model(inputs=[input_one,input_two],output=result)
    # input_two = Input(shape=(1000,))
    # Embedding_layer = Embedding(len(word_index) + 1,
    #                             100,
    #                             weights=[embedding_matrix],
    #                             input_length=1000,
    #                             trainable=False)(input_one)
    # Embedding_layer = (Embedding(output_dim=64,
    #                              input_dim=10000,
    #                              name="Embedding",
    #                              input_length=1000))(input_one)



y_train, train_textOne,train_textTwo =get_dataC50AllTest("C50train/")
y_test, test_textOne,test_textTwo =get_dataC50AllTest("C50test/")

train_inputOne,train_inputTwo =get_dataC50WritingStyle("trainData.csv")
train_inputOne=np.array(train_inputOne)
train_inputTwo=np.array(train_inputTwo)
test_inputOne,test_inputTwo =get_dataC50WritingStyle("testData.csv")
test_inputOne=np.array(test_inputOne)
test_inputTwo=np.array(test_inputTwo)
token = Tokenizer(num_words=10000)
token.fit_on_texts(train_textOne)
token.fit_on_texts(train_textTwo)
word_index = token.word_index
x_train_seqOne=token.texts_to_sequences(train_textOne)
x_train_seqTwo=token.texts_to_sequences(train_textTwo)
x_test_seqOne = token.texts_to_sequences(test_textOne)
x_test_seqTwo = token.texts_to_sequences(test_textTwo)
x_trainOne = sequence.pad_sequences(x_train_seqOne, maxlen=1000)
x_trainTwo = sequence.pad_sequences(x_train_seqTwo, maxlen=1000)
x_testOne= sequence.pad_sequences(x_test_seqOne, maxlen=1000)
x_testTwo= sequence.pad_sequences(x_test_seqTwo, maxlen=1000)
embeddings_index = {}
GLOVE_DIR='D:/identifyAuthor/glove.6B'
f = open(os.path.join(GLOVE_DIR, 'glove1.6B.100d.txt'), encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Found %s word vectors.' % len(embeddings_index))

author_a = Input(shape=(1000, ))
author_b = Input(shape=(1000, ))
input_a=Input(shape=(13,))
input_b=Input(shape=(13,))
base_network=create_base_networkText()
base_networkA=base_network([author_a,input_a])
base_networkB=base_network([author_b,input_b])
distance = Lambda(function=euclidean_distance,
                  output_shape=(1,))([base_networkA, base_networkB])
# merged_vector = keras.layers.concatenate([base_networkA, base_networkB], axis=-1)
# predictionsDense = Dense(units=512, activation='tanh',kernel_regularizer=l2(0.01))(merged_vector)
# predictionsDrop=Dropout(0.25)(predictionsDense)
# predictions = Dense(units=1, activation='sigmoid',kernel_regularizer=l2(0.01))(predictionsDrop)
model = Model(inputs=[author_a,input_a,author_b,input_b], outputs=distance)
rms = RMSprop()
model.summary()
model.compile(optimizer=rms,
              # optimizer='adam',
              loss=contrastive_loss,
              # loss='binary_crossentropy',
              metrics=['accuracy'])
early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=1,
                              restore_best_weights=True,
                              verbose=0, mode='auto')
train_history=model.fit([x_trainOne,train_inputOne,x_trainTwo,train_inputTwo],y_train, shuffle=True,batch_size=64,epochs=20,verbose=2,validation_split=0.1)
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
scores = model.evaluate([x_testOne,test_inputOne,x_testTwo,test_inputTwo], y_test, verbose=1)
print(scores[1])


# model_json = model.to_json()
# with open("model/author_Test_model.json", "w") as json_file:
#     json_file.write(model_json)
#
# model.save_weights("model/author_Test_model.h5")
# print("Saved model to disk")
