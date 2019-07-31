import csv
import random
import keras
from keras import Sequential, Input, Model
from keras.layers import Bidirectional, Embedding, LSTM, Dense, Dropout, Lambda, K, Flatten, TimeDistributed, \
    BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from keras.regularizers import l2
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.python.keras.layers import Attention


np.random.seed(10)
randomNum=random.randint(0,2499)
import re
re_tag = re.compile(r'<[^>]+>')
def rm_tags(text):
    return re_tag.sub('', text)
import os
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
    random.seed(randomNum)
    random.shuffle(all_textsOne)
    random.seed(randomNum)
    random.shuffle(all_textsTwo)
    random.seed(randomNum)
    random.shuffle(all_labels)
    return all_labels, all_textsOne, all_textsTwo

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
            all_input+=[i[2:16]]
            if index %2==0:
                all_inputOne+=[i[2:16]]
                index=index+1
            else:
                all_inputTwo+=[i[2:16]]
                index=index+1
    for i in range(int(len(all_input)/2)):
        all_inputOne+=[all_input[i]]
        all_inputTwo+=[all_input[i+50]]
    random.seed(randomNum)
    random.shuffle(all_inputOne)
    random.seed(randomNum)
    random.shuffle(all_inputTwo)
    return all_inputOne,all_inputTwo

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


y_train, train_textOne,train_textTwo =get_dataC50AllTest("C50train/")
y_test, test_textOne,test_textTwo =get_dataC50AllTest("C50test/")
train_inputOne,train_inputTwo =get_dataC50WritingStyle("train1Data.csv")
train_inputOne=np.array(train_inputOne)
train_inputTwo=np.array(train_inputTwo)
test_inputOne,test_inputTwo =get_dataC50WritingStyle("test1Data.csv")
test_inputOne=np.array(test_inputOne)
test_inputTwo=np.array(test_inputTwo)
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
Dense_layer=Dense(units=256,activation='relu',name="Dense1")
Dense_layerA=Dense_layer(Bidirectional_layerA)
Dense_layerB=Dense_layer(Bidirectional_layerB)
Dropout_layer2=Dropout(0.2,name="Dropout2")
Dropout_layer2A=Dropout_layer2(Dense_layerA)
Dropout_layer2B=Dropout_layer2(Dense_layerB)
outputText = Dense(units=50, activation='softmax',name="OutPut")
Atext=outputText(Dropout_layer2A)
Btext=outputText(Dropout_layer2B)

input_one=Input(shape=(14,))
input_two=Input(shape=(14,))
DenseStyle=Dense(units=14,activation='relu',name="DenseStyle1")
DenseStyleOne=DenseStyle(input_one)
DenseStyleTwo=DenseStyle(input_two)
DropStyle=Dropout(0.2, name="DropoutStyle1")
DropStyleOne=DropStyle(input_one)
DropStyleTwo=DropStyle(input_two)
DenseStyleOther=Dense(units=128,activation='relu',name="DenseStyle2")
DenseStyle2One=DenseStyleOther(input_one)
DenseStyle2Two=DenseStyleOther(input_two)
DenseStyle3=Dense(units=256,activation='relu',name="DenseStyle3")
DenseStyle3One=DenseStyle3(DenseStyle2One)
DenseStyle3Two=DenseStyle3(DenseStyle2Two)
DropStyle2=Dropout(0.2,name="DropoutStyle2")
DropStyle2One=DropStyle2(DenseStyle3One)
DropStyle2Two=DropStyle2(DenseStyle3Two)
outputStyle=Dense(units=50, activation='softmax',name="OutPutStyle")
styleA=outputStyle(DropStyle2One)
styleB=outputStyle(DropStyle2Two)
merged_vectorOne=keras.layers.Multiply()([Atext, styleA])
merged_vectorTwo=keras.layers.Multiply()([Btext, styleB])
merged_vector= keras.layers.concatenate([merged_vectorOne, merged_vectorTwo], axis=-1)
merged = BatchNormalization()(merged_vector)
# result=np.dot(merged_vectorOne,merged_vectorTwo)/(np.linalg.norm(merged_vectorOne)*np.linalg.norm(merged_vectorTwo))
# merged_vector=keras.layers.dot([merged_vectorOne,merged_vectorTwo],0, normalize=True)
# user_tag_matric = np.matrix(np.array([merged_vectorOne, merged_vectorTwo]))
# user_tag_matric=np.array(user_tag_matric)
# user_similarity = cosine_similarity(user_tag_matric)
# distance = Lambda(euclidean_distance,
#                   output_shape=(1,))([merged_vectorOne,merged_vectorTwo])
# merged_vectorOne = keras.layers.concatenate([Atext, styleA], axis=-1)
# merged_vectorTwo = keras.layers.concatenate([Btext, styleB], axis=-1)
# merged_vector = keras.layers.concatenate([merged_vectorOne, merged_vectorTwo], axis=-1)
predictionsDense = Dense(units=256, activation='relu')(merged)
predictionsDrop=Dropout(0.2)(predictionsDense)
merged = BatchNormalization()(predictionsDrop)
predictions = Dense(units=1, activation='sigmoid',kernel_regularizer=l2(0.01))(merged)
# rms=RMSprop()
model = Model(inputs=[author_a,author_b,input_one,input_two], outputs=predictions)
model.summary()
model.load_weights('model/author_model.h5',by_name=True)
model.load_weights('model/author_Style_model.h5',by_name=True)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
train_history = model.fit([x_trainOne,x_trainTwo,train_inputOne,train_inputTwo],y_train,batch_size=32, epochs=20, verbose=2,shuffle=True, validation_split=0.01)

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
scores = model.evaluate([x_testOne,x_testTwo,test_inputOne,test_inputTwo], y_test, verbose=1)
print(scores[1])
