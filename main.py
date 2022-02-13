import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
sns.set(style="ticks")
import nltk
import re
import string
from sklearn.metrics import accuracy_score, confusion_matrix
from io import StringIO
import pickle
import streamlit as st
from collections.abc import Iterable   # import directly from collections for Python < 3.3



def convertStringToDataFrame(str):
    newStr = """review
     {}""".format(str)
    preData = StringIO(newStr)
    newData = pd.read_csv(preData, sep = "////////")
    return newData


# Функция для очистки пунктуации
def remove_punc(text):
    if isinstance(text, Iterable):
        clean = "".join([x.lower() for x in text if x not in punc_set])
        return clean
    else:
        text = str(text).replace('\n', ' ')
        clean = "".join([x.lower() for x in text if x not in punc_set])
        return clean


# Фунцкция для токенизации
def tokenize(text):
    tokens = re.split("\W+", text)
    return tokens


# Функция для очистки датасета от необязательных слов
def remove_stopwords(tokenized_words):
    Ligit_text = [word for word in tokenized_words if word not in stopwords]
    return Ligit_text


# Функция для лемматизации датасета
def lemmatizing(tokenized_text):
    lemma = [wnl.lemmatize(word) for word in tokenized_text]
    return lemma

@st.cache(suppress_st_warning=True)
def prepareData(data):
    data_out = data.copy()
    data_out['no_punc'] = data_out['review'].apply(lambda z: remove_punc(z))

    # Применяем функцию токенизации
    data_out['tokenized_Data'] = data_out['no_punc'].apply(lambda z: tokenize(z))

    # Очищаем датасет от необязательных слов
    data_out["no_stop"] = data_out["tokenized_Data"].apply(lambda z: remove_stopwords(z))

    # Применяем функцию лемматизации
    data_out['lemmatized'] = data_out['no_stop'].apply(lambda z: lemmatizing(z))

    # Этот шаг выполняется здесь, потому что столбец «lemmatized» представляет собой список токенизированных слов, и когда мы применяем векторизацию
    # методы, такие как count vectorizer или TFIDF, требуют ввода строки. Следовательно, преобразуем все токенизированные слова в строку
    data_out['lemmatized'] = [" ".join(review) for review in data_out['lemmatized'].values]

    data_out.head()
    return data_out

# Отрисовка ROC-кривой
def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score,
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    #plt.figure()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")


@st.cache(suppress_st_warning=True)
def print_models(models_select, x_test, y_test):
    current_models_list = []
    scores = {}
    for model_name in models_select:
        model = clas_models[model_name]
        # Предсказание значений
        Y_pred = model.predict(x_test)
        # Предсказание вероятности класса "1" для roc auc
        Y_pred_proba_temp = model.predict_proba(x_test)
        Y_pred_proba = Y_pred_proba_temp[:, 1]

        current_models_list.append(model_name)
        scores["accuracy"] = accuracy_score(y_test, Y_pred)
        scores["precision"] = precision_score(y_test, Y_pred)
        scores["recall"] = recall_score(y_test, Y_pred)
        scores["f1 score"] = f1_score(y_test, Y_pred)

        scores = pd.DataFrame(data = scores, index=[0])
        # Отрисовка ROC-кривых
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        draw_roc_curve(y_test.values, Y_pred_proba, ax[0])

        plot_confusion_matrix(model, x_test, y_test.values, ax=ax[1],
                              display_labels=['0', '1'],
                              cmap=plt.cm.Blues, normalize='true')
        fig.suptitle(model_name)
        st.pyplot(fig)
        st.write(scores)

@st.cache(suppress_st_warning=True)
def load_data():
    data = pd.read_csv('movie_data.csv', nrows = 2000)
    return data
def clear_text():
    st.session_state["text"] = ""

if __name__ == '__main__':
    # Убираем пунктуацию со всего датасета
    punc_set = string.punctuation

    # Получаем список необязательных слов (местоимений, артиклей)
    stopwords = nltk.corpus.stopwords.words('english')

    # Импорт «WordNetLemmatizer» в качестве функции лемматизации для поиска леммы слов
    wnl = nltk.wordnet.WordNetLemmatizer()

    st.header('Классификация отзывов на фильмы')

    # загружаем данные
    data_load_state = st.text('Загрузка данных...')
    tfidf_vect = pickle.load(open("tfidf_vect.pickle", 'rb'))
    best_model = pickle.load(open("best_model.sav", 'rb'))
    ber_nb_model = pickle.load(open("Ber_NB_tf_best.sav", 'rb'))
    lg_model = pickle.load(open("Lg_reg_tf.sav", 'rb'))
   # rf_model = pickle.load(open("RF_tf.sav", 'rb'))
    gb_model = pickle.load(open("GB_tf.sav", 'rb'))
    data = load_data()
    data = prepareData(data)
    # Разделение данных на более мелкие кадры данных для обучения и тестирования
    x_test = data.iloc[:, 5]
    y_test = data.iloc[:, 1]
    x_test = tfidf_vect.transform(x_test.values)
    data_load_state.text('Данные загружены!')


    # Модели
    models_list = ['Bernoulli Naive Bayes', 'Multinomial Naive Bayes', 'Logistic Regression', 'Gradient Boosting']

    clas_models = {'Bernoulli Naive Bayes': ber_nb_model,
                      'Multinomial Naive Bayes': best_model,
                      'Logistic Regression': lg_model,
       #               'Random Forest': rf_model,
                      'Gradient Boosting': gb_model}

    st.sidebar.header('Модели машинного обучения')
    models_select = st.sidebar.multiselect('Выберите модели', models_list)

    st.subheader('Оценка качества моделей')
    print_models(models_select, x_test, y_test)

    txt = st.sidebar.text_area(label = "Отзыв для классификации", key="text", height=500)
    if st.sidebar.button("Классифицировать"):
        review = convertStringToDataFrame(txt)
        data_text = prepareData(review)
        data_text = tfidf_vect.transform(data_text['lemmatized'].values)
        result = best_model.predict(data_text)
        if result[0] == 1:
            st.sidebar.success("Положительный отзыв")
        else:
            st.sidebar.error("Отрицательный отзыв")
    st.sidebar.button("Очистить", on_click=clear_text)
    #st.write(txt)









