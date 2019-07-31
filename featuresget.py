from keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils


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

y_trainOrign, train_text = read_files("C50train/")
y_train=np_utils.to_categorical(y_trainOrign)
y_testOrign, test_text = read_files("C50test/")
y_test=np_utils.to_categorical(y_testOrign)

# In[6]:
# 先读取所有文章建立字典，限制字典的数量为nb_words=2000
token = Tokenizer(num_words=10000)
token.fit_on_texts(train_text)
# In[8]:
# 将文字转为数字序列
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
# In[10]:
# 截长补短，让所有影评所产生的数字序列长度一样
# In[11]:
x_train = sequence.pad_sequences(x_train_seq, maxlen=1000)
x_test = sequence.pad_sequences(x_test_seq, maxlen=1000)
# # 建立模型
# In[12]:
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

# In[13]:
model = Sequential()
# In[14]:
model.add(Embedding(output_dim=128,
                    input_dim=10000,
                    input_length=1000,
                    name="Embedding"
                    ))
  # 对句子中的每个词
model.add(Dropout(0.2, name="Dropout1"))
model.add(Conv1D(32,5,activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(32,5,activation='relu'))
model.add(Bidirectional(LSTM(64),name="Bidirectional"))

# In[16]:
model.add(Dense(units=256,
                activation='relu',name="Dense1"))
model.add(Dropout(0.2,name="Dropout2"))
# In[17]:
model.add(Dense(units=50,
                activation='softmax',name="OutPut"))
# In[18]:
model.summary()
# # 训练模型
# In[19]:
model.compile(loss='categorical_crossentropy',
              # optimizer='rmsprop',
              optimizer='adam',
              metrics=['accuracy'])
# In[20]:
train_history = model.fit(x_train, y_train, batch_size=256,
                          epochs=20, verbose=2,
                          # validation_split=0.01)
                        validation_data=(x_test, y_test))
# In[21]:
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
# In[24]:
scores = model.evaluate(x_test, y_test, verbose=1)
print('test loss:', scores[0])
print('test accuracy:', scores[1])

predict = model.predict_classes(x_test)
predict_classes = predict.reshape(2500)

# def display_test_Sentiment(i):
#     data = np.argmax(y_test[i])
#     if data==predict_classes[i]:
#         print("1")
#     # print('标签label:', data, '预测结果:',predict_classes[i])

# for i in range(2500):
#     display_test_Sentiment(i)
# # 预测概率
model_json = model.to_json()
with open("model/author_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model/author_model.h5")
print("Saved model to disk")

