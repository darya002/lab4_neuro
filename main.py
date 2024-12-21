
import tensorflow.keras as keras

# Параметры
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Параметры
maxWordsCount = 5000
inp_words = 4

# Текст для обработки
with open('уве.txt', 'r', encoding='utf-8') as f:
    texts = f.read()
    texts = texts.replace('\ufeff', '').replace('\n', ' ')

# Токенизация текста
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True)
tokenizer.fit_on_texts([texts])
data = tokenizer.texts_to_sequences([texts])[0]

# Формирование X и Y
X = np.array([data[i:i+inp_words] for i in range(len(data) - inp_words)])
Y = np.array([data[i+inp_words] for i in range(len(data) - inp_words)])

# Построение модели
model = keras.Sequential()
model.add(layers.Embedding(input_dim=maxWordsCount, output_dim=128, input_length=inp_words))
model.add(layers.LSTM(256, return_sequences=False))
model.add(layers.Dense(maxWordsCount, activation='softmax'))
model.build(input_shape=(None, inp_words))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Обучение
history = model.fit(X, Y, batch_size=32, epochs=50)

def buildPhrase(text1, str_len=20, temperature=1.2):
    res = text1
    data = tokenizer.texts_to_sequences([text1])[0]  # Преобразуем начальный текст в последовательность индексов

    # Дополняем последовательность до минимальной длины inp_words
    if len(data) < inp_words:
        data = [0] * (inp_words - len(data)) + data  # Добавляем нули (аналог <pad>)

    for _ in range(str_len):
        # Берем последние `inp_words` индексов
        inp = np.array(data[-inp_words:]).reshape(1, inp_words)  # Преобразуем в форму (1, inp_words)

        # Предсказываем распределение вероятностей следующего слова
        pred = model.predict(inp, verbose=0)[0]
        pred = np.log(pred + 1e-9) / temperature  # Регулируем температуру
        pred = np.exp(pred) / np.sum(np.exp(pred))  # Преобразуем обратно в вероятности

        # Выбираем следующее слово на основе распределения вероятностей
        indx = np.random.choice(range(maxWordsCount), p=pred)
        data.append(indx)

        # Добавляем предсказанное слово в результат
        if indx in tokenizer.index_word:
            res += " " + tokenizer.index_word[indx]
        else:
            res += " ?"  # Если индекс не найден, добавляем "?"

    return res


res = buildPhrase("он услышал шум", str_len=20, temperature=1.2)
print(res)