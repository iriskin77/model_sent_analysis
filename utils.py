import pandas as pd
from pymystem3 import Mystem
from keras.utils import pad_sequences
from pathlib import Path

abs_path = Path(__file__).resolve().parent.parent

model = load_model(rf"{abs_path}/best_model_LSTM10000_2.h5")

with open(rf'{abs_path}/tokenizer_json.json') as file:
        data = json.load(file)
        tokenizer = tokenizer_from_json(data)


def process_file_data(path, name_column):
    """"Функция получает на вход набор комментариев, определяет их тональность, возвращает обработанные комментарии в виде DataFrame"""""

    try:
        data_comments = pd.read_excel(path)[name_column]
        model = model
        tokenizer = tokenizer
        m = Mystem()
        added_commentaries = []
        added_score = []
        added_mark = []

        for comment_index in range(len(data_comments)):
            if type(data_comments[comment_index]) == str:
                lemms = m.lemmatize(data_comments[comment_index])
                prep = [word.strip(r'!?"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n').lower() for word in lemms if word.isalpha()]
                sequence = tokenizer.texts_to_sequences([' '.join(prep)])
                sequence = pad_sequences(sequence, 64)
                score = model.predict(sequence)

                if score[[0]] > 0.9:
                    added_commentaries.append(data_comments[comment_index])
                    added_mark.append('Отрицательный')
                    added_score.append(score)

                elif 0.9 > score[[0]] > 0.4:
                    added_commentaries.append(data_comments[comment_index])
                    added_mark.append('Нейтральный')
                    added_score.append(score)

                else:
                    added_commentaries.append(data_comments[comment_index])
                    added_mark.append('Положительный')
                    added_score.append(score)

        data['Сообщение'] = added_commentaries
        data['Класс'] = added_mark
        data['Коэффициент'] = added_score

        return data

    except Exception as ex:
        print("Error:", ex)
