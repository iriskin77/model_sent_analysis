# -*- coding: utf-8 -*-
# For DataFrame object
import pandas as pd
import numpy as np
import json

# History visualization
import matplotlib.pyplot as plt

# Neural Network
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM 
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.data_utils import pad_sequences

# Text Vectorizing
from pymystem3 import Mystem
from keras.preprocessing.text import Tokenizer
from keras import utils

# Train-test-split
from sklearn.model_selection import train_test_split


def lemmatization(data):
      lemmatizer = Mystem()
      lemmatized_words = []
      for comment in data:
        lemmas = lemmatizer.lemmatize(comment)
        res = [word.strip('!?"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n').lower() for word in lemmas if word.isalpha()]
        lemmatized_words.append(' '.join(res))
      return lemmatized_words


def get_model_LSTM():
    model = Sequential()
    model.add(Embedding(10000, 128, input_length=64))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model


"""Извлекаем размеченные данные из файла с помощью Pandas"""

data_comments = pd.read_excel(r'Pos_Neg1.xlsx')['comment']
data_target = pd.read_excel(r'Pos_Neg1.xlsx')['class']
prep = lemmatization(data_comments)


"""С помощью функции lemmatization и объекта класса Tokenizer токенизируем и векторизируем комментарии"""

tokenizer = Tokenizer(num_words=10000,
                              lower=True, split=' ',
                              char_level=False)

tokenizer.fit_on_texts(prep)
sequence = tokenizer.texts_to_sequences(prep)
token_json = tokenizer.to_json()
tokenizer_json = tokenizer.to_json()
with open('/content/gdrive/My Drive/tokenizer_json.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(token_json, ensure_ascii=False))


"""Приводим все комментарии к единому размеру: 64 элемента, формируем переменные x, y для обучения и тестирования"""

x = pad_sequences(sequence, 64)
y = data_target


"""Делим данные на обучающую выборку и проверочную, обучаем модель"""


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = get_model_LSTM()
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=1)


"""Выбираем лучший вариант и сохраняем на goolge-drive"""


model_save_path = model.save('/content/gdrive/My Drive/best_model_LSTM10000.h5')
callback = ModelCheckpoint(model_save_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)


"""Визуализируем результаты обучения с помощью библиотеки matplotlib"""

plt.plot(history.history['accuracy'],
         label='Train accuracy')
plt.plot(history.history['val_accuracy'],
         label='Validation accuracy')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

model.summary()
