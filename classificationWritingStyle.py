import csv

import random

from keras.optimizers import SGD
from keras.utils import np_utils
from keras import Sequential, Input, Model
from keras.layers import Bidirectional, Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Activation, Reshape, \
    Flatten
import numpy as np
np.random.seed(10)
randomNum = random.randint(0, 2499)
import re
re_tag = re.compile(r'(\")')
def rm_tags(text):
    return re_tag.sub('', text)
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
            all_input+=[i[2:16]]
    for i in range(50):
        all_labels += ([i] * 50)
    random.seed(randomNum)
    random.shuffle(all_input)
    random.seed(randomNum)
    random.shuffle(all_labels)
    return all_labels,all_input


train_label,train_input =get_dataC50WritingStyle("train1Data.csv")
train_label=np_utils.to_categorical(train_label,num_classes=50)
train_input=np.array(train_input)
test_label,test_input =get_dataC50WritingStyle("test1Data.csv")
test_input=np.array(test_input)
test_label=np_utils.to_categorical(test_label,num_classes=50)

model = Sequential()
# In[14]:
# model.add(Dense(units=64,input_dim=14,activation='relu',name="DenseStyle1"))

model.add(Conv1D(filters=32,kernel_size=5,activation='relu',input_shape=(256,14)))
model.add(MaxPooling1D(3))
model.add(Flatten())
# model.add(Conv1D(5,5,activation='relu'))
model.add(Dropout(0.2, name="DropoutStyle1"))
model.add(Dense(units=64,activation='tanh',name="DenseStyle2"))
# In[16]:
# model.add(Dense(units=512,
#                 activation='tanh',name="DenseStyle3"))
model.add(Dropout(0.2,name="DropoutStyle2"))
# In[17]:
model.add(Dense(units=50, activation='softmax',name="OutPutStyle"))
# In[18]:
model.summary()
# # 训练模型
# In[19]:
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# In[20]:
train_history = model.fit(train_input, train_label,
                          epochs=100, verbose=2,
                          validation_data=(test_input, test_label))




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
scores = model.evaluate(test_input, test_label, verbose=1,batch_size=256)
print('test loss:', scores[0])
print('test accuracy:', scores[1])


# predict = model.predict_classes(test_input)
# predict_classes = predict.reshape(2500)




model_json = model.to_json()
with open("model/author_Style_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model/author_Style_model.h5")
print("Saved model to disk")
