import keras
import pandas as pd

# 그냥 외부 프로그램으로 합쳐버림...

sum_news = pd.read_csv('sum_news_naver.csv')
#sum_news
# 아무튼 성공임 그런거임

from sklearn.model_selection import train_test_split
X_train, X_test= train_test_split(sum_news,random_state=66,shuffle=True)

import numpy as np

from keras_preprocessing.text import Tokenizer

def one_hot(x_in):
    x_data = x_in['data']
    x_class = x_in['class']
    tokenizer = Tokenizer(num_words=1000)
    # 단어 인덱스 구축
    tokenizer.fit_on_texts(sum_news['data'])
#     return tokenizer.index_word, tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(x_data)
    print("sequences : ", sequences)
    one_hot_results = tokenizer.texts_to_matrix(x_data, mode = 'binary')
    print("one_hot_results : ", one_hot_results)
    return one_hot_results, x_class

x_train, y_train = one_hot(X_train)
x_text, y_text = one_hot(X_test)

y_train = np.asarray(y_train).astype('float32')
y_text =np.asarray(y_text).astype('float32')
#근대 이 과정이 꼭 필요한가? 일단 주석처리해 해보고 나중에 해볼까?

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])

x_val = x_train[:100000]
partial_x_train = x_train[100000:]

y_val = y_train[:100000]2
partial_y_train = y_train[100000:]

history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=20,
                   batch_size=512, #너무 클수도 있다 나중에 조정해 볼껏 (처음 512)
                   validation_data=(x_val,y_val))