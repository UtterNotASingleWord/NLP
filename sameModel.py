import math
import random

import keras
from keras import Input, Model
from keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, K, Lambda, GRU, \
    AveragePooling1D, merge, Dot, Permute
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


from AttentionLayer import AttentionLayer

np.random.seed(10)
import re
re_tag = re.compile(r'<[^>]+>')
def rm_tags(text):
    return re_tag.sub('', text)
import os
randomNum = random.randint(0, 2499)
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

y_train, train_textOne,train_textTwo =get_dataC50AllTest("C50train/")
y_test, test_textOne,test_textTwo =get_dataC50AllTest("C50test/")


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
# # 建立模型
# In[12]:
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

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
def accuracy(y_true, y_pred): # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# def consin_similarity(vects):
#     x, y = vects
#     distance=euclidean_distance(vects)
#     dist =Dot(x, y) / distance
#     return dist

# def cosine_similarity(vects):
#     x, y = vects
#     dot_product = 0.0
#     normA = 0.0
#     normB = 0.0
#     for a, b in zip(x, y):
#         dot_product += a * b
#         normA += a ** 2
#         normB += b ** 2
#     if normA == 0.0 or normB == 0.0:
#         return 0
#     else:
#         return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)

def sigmoid(x):

    return 1. / (1 + np.exp(-x))

def consin_loss(y_true, y_pred,feat1, feat2):

    return  -K.sum(y_true* math.log(y_pred)+y_pred*math.log(y_true))+0.001*sigmoid(cosine_similarity(feat1, feat2))



def compute_accuracy(y_true, y_pred): # numpy上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(28, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = np.multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul



def create_base_network():
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=(1000,))
    Embedding_layer=Embedding(output_dim=128,
              input_dim=10000,
              input_length=1000,
              name="Embedding" )
    # 对句子中的每个词
    x = Embedding_layer(input)
    Conv1D_layer=Conv1D(64, 2, activation='relu')(x)
    MaxPooling_layer=MaxPooling1D(5)(Conv1D_layer)
    x = Dropout(0.2)(MaxPooling_layer)
    Conv1D_layerSec=Conv1D(64, 2, activation='relu')(x)
    x = Flatten()(Conv1D_layerSec)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def create_base_network1():
    '''Base network to be shared (eq. to feature extraction).
    '''
    # input = Input(shape=(1000,))
    # # Embedding_layer=Embedding(output_dim=128,
    # #           input_dim=10000,
    # #           input_length=1000,
    # #           name="Embedding" )
    # Embedding_layer = Embedding(len(word_index) + 1,
    #                             100,
    #                             weights=[embedding_matrix],
    #                             input_length=1000,
    #                             trainable=False)
    # # 对句子中的每个词
    # x = Embedding_layer(input)
    # # Conv1D_layer=Conv1D(64, 5, activation='relu')(x)
    # #     # MaxPooling_layer=MaxPooling1D(5)(Conv1D_layer)
    # #     # x = Dropout(0.2)(MaxPooling_layer)
    # #     # Conv1D_layerSec=Conv1D(64, 5, activation='relu')(x)
    # #     # x = Flatten()(Conv1D_layerSec)
    # article=Bidirectional(GRU(64))(x)
    # result=AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(article)
    # x = Dense(50, activation='soft')(result)
    # return Model(input, x)
    model= Sequential()
    # bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    model.add(Embedding(len(word_index) + 1,
                        100,
                        weights=[embedding_matrix],
                        input_length=1000,
                        trainable=True))
    # model.add(Bidirectional(GRU(32)))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Bidirectional(LSTM(units=64,return_sequences=True)))
    # model.add(AveragePooling1D(pool_size=2, strides=1))
    # model.add(Conv1D(128, 5, activation='relu'))
    # model.add(Flatten())
    model.add(AttentionLayer())
    model.add(Dropout(0.1))
    model.add(Dense(units=50,activation='softmax'))
    return model


# # In[13]:
# model = Sequential()
# # In[14]:
# model.add(Embedding(output_dim=128,
#                     input_dim=10000,
#                     input_length=1000,
#                     name="Embedding"
#                     ))
#   # 对句子中的每个词
# model.add(Dropout(0.2, name="Dropout1"))
# model.add(Conv1D(32,5,activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Conv1D(32,5,activation='relu'))
# model.add(Flatten())
# model.add(Bidirectional(LSTM(64),name="Bidirectional"))
input_a=Input(shape=(1000, ))
input_b =Input(shape=(1000, ))
base_network = create_base_network1()
author_a = base_network(input_a)
author_b = base_network(input_b)
# simliarity=Dot(axes=0,normalize=True)([author_a, author_b])
distance = Lambda(euclidean_distance, output_shape=(1,))([author_a, author_b])
# cosine_sim = merge([a_rotated, b], mode=cosine, output_shape=lambda x: x[:-1])
# Dot_layer=keras.layers.Dot(0,normalize=True)([author_a,author_b])
# output=Flatten()(simliarity)
# output=Dense(units=1,activation="sigmoid")(output)
model = Model([input_a, input_b], distance)
model.summary()
# # 训练模型
# In[19]:
# model.compile(loss=contrastive_loss,
#               # optimizer='rmsprop',
#               optimizer='adam',
#               metrics=['accuracy'])
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
# model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
# In[20]:
train_history = model.fit([x_trainOne,x_trainTwo],y_train, batch_size=512,
                          epochs=50, verbose=2,
                          validation_data=([x_testOne,x_testTwo],y_test))
# In[21]:
# def show_train_history(train_history, train, validation):
#     plt.plot(train_history.history[train])
#     plt.plot(train_history.history[validation])
#     plt.title('Train History')
#     plt.ylabel(train)
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()


# In[22]:
# show_train_history(train_history, 'acc', 'val_acc')
# # In[23]:
# show_train_history(train_history, 'loss', 'val_loss')
# # 评估模型的准确率
# In[24]:
scores = model.evaluate([x_testOne,x_testTwo],y_test, verbose=1)
print('test loss:', scores[0])
print('test accuracy:', scores[1])

y_pred = model.predict([x_trainOne,x_trainTwo])
tr_acc = compute_accuracy(y_train, y_pred)
y_pred = model.predict([x_testOne,x_testTwo])
te_acc = compute_accuracy(y_test, y_pred)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))



model_json = model.to_json()
with open("model/author_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model/author_model.h5")
print("Saved model to disk")

