import pandas as pd
import numpy as np
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC

from matplotlib import pyplot as plt

#Instanciamos el modulo de spacy
nlp = spacy.load('es_core_news_sm')

stopwords = list(nlp.Defaults.stop_words)
stop_words = ' '.join(stopwords)

#Cargamos y abrimos los corpus
corpus_train = pd.read_csv('df-train_corpus_normalized.csv')
corpus_test = pd.read_csv('df-test_corpus_normalized.csv')

#Agrupamos con base en la bi-ideologia
grupos = corpus_train.groupby('Bi-Ideologia')
grupo_clases_left = grupos.get_group(0)
grupo_clases_right = grupos.get_group(1)

#Combinamos los corpus para hacer la vectorizacion
corpus_completo_clases = pd.concat([grupo_clases_left['Tweets'], grupo_clases_right['Tweets']], ignore_index=True)

#Realizamos la vectorizacion por TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=5, ngram_range=(1,3))
vectorizer_clases = TfidfVectorizer(stop_words=stopwords, min_df=5, ngram_range=(1,3))

#Ajustamos los vectorizadores
vectorizer.fit(corpus_train['Tweets'])
vectorizer_clases.fit(corpus_completo_clases)

#Obtenemos los vocabularios
vocabulario = vectorizer.get_feature_names_out()
vocabulario_clases = vectorizer_clases.get_feature_names_out()

#Hacemos los filtrados para el test
test_filtrado = [documento for documento in corpus_test['Tweets'] if any(palabra in documento for palabra in vocabulario)]
test_filtrado_clases = [documento for documento in corpus_test['Tweets'] if any(palabra in documento for palabra in vocabulario_clases)]

#Obtenemos las representaciones vectoriales
vector_test = vectorizer.transform(test_filtrado)
vector_test_clases = vectorizer_clases.transform(test_filtrado_clases)
tweets_vectorizados = vectorizer.transform(corpus_train['Tweets'])
tweets_vectorizados_clases = vectorizer_clases.transform(corpus_completo_clases)

#Ahora separamos los datos en train y test de nuevo
longitud_corpus_train = corpus_train.shape[0]
longitud_corpus_train_clase_left = grupo_clases_left.shape[0]

#Obtebnemos ya los valores de entrada de nuestros modelos
tfidf_matriz_train_left = tweets_vectorizados_clases[:longitud_corpus_train_clase_left]
tfidf_matriz_train_right = tweets_vectorizados_clases[longitud_corpus_train_clase_left:]
tfidf_matriz_train = tweets_vectorizados

#Separamos nuestros dataset
X_train_bi, X_test_bi, y_train_bi, y_test_bi = train_test_split(tfidf_matriz_train, corpus_train['Bi-Ideologia'], random_state=0, test_size=0.2)
X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(tfidf_matriz_train, corpus_train['Genero'], random_state=0, test_size=0.2)
X_train_pro, X_test_pro, y_train_pro, y_test_pro = train_test_split(tfidf_matriz_train, corpus_train['Profesion'], random_state=0, test_size=0.2)
X_train_left, X_test_left, y_train_left, y_test_left = train_test_split(tfidf_matriz_train_left, grupo_clases_left['Multi-Ideologia'], random_state=0, test_size=0.2)
X_train_right, X_test_right, y_train_right, y_test_right = train_test_split(tfidf_matriz_train_right, grupo_clases_right['Multi-Ideologia'], random_state=0, test_size=0.2)

#Instanciamos los modelos
clf_bi = LogisticRegression()
clf_gender = SVC(kernel='linear', C = 1.0) #LogisticRegression()
clf_pro = SVC(kernel='linear', C = 1.0)
clf_left = LogisticRegression()
clf_right = SVC(kernel='linear', C = 1.0) #LogisticRegression()

#Entrenamos los modelos
clf_bi.fit(X_train_bi, y_train_bi)
clf_gender.fit(X_train_gender, y_train_gender)
clf_pro.fit(X_train_pro, y_train_pro)
clf_left.fit(X_train_left, y_train_left)
clf_right.fit(X_train_right, y_train_right)

y_pred_bi = clf_bi.predict(X_test_bi)
y_pred_gender = clf_gender.predict(X_test_gender)
y_pred_pro = clf_pro.predict(X_test_pro)
y_pred_left = clf_left.predict(X_test_left)
y_pred_right = clf_right.predict(X_test_right)

matriz_confusion_bi = confusion_matrix(y_test_bi, y_pred_bi)
matriz_confusion_gender = confusion_matrix(y_test_gender, y_pred_gender)
matriz_confusion_pro = confusion_matrix(y_test_pro, y_pred_pro)
matriz_confusion_left = confusion_matrix(y_test_left, y_pred_left)
matriz_confusion_right = confusion_matrix(y_test_right, y_pred_right)

disp_bi = ConfusionMatrixDisplay(confusion_matrix = matriz_confusion_bi)
disp_gender = ConfusionMatrixDisplay(confusion_matrix = matriz_confusion_gender)
disp_pro = ConfusionMatrixDisplay(confusion_matrix = matriz_confusion_pro)
disp_left = ConfusionMatrixDisplay(confusion_matrix = matriz_confusion_left)
disp_right = ConfusionMatrixDisplay(confusion_matrix = matriz_confusion_right)

disp_bi.plot()
disp_gender.plot()
disp_pro.plot()
disp_left.plot()
disp_right.plot()

cr_bi = classification_report(y_test_bi, y_pred_bi)
cr_gender = classification_report(y_test_gender, y_pred_gender)
cr_pro = classification_report(y_test_pro, y_pred_pro)
cr_left = classification_report(y_test_left, y_pred_left)
cr_right = classification_report(y_test_right, y_pred_right)

print('Clasificacion bi')
print(f"Reporte de clasificacion: {cr_bi}")

print('Clasificacion gender')
print(f"Reporte de clasificacion: {cr_gender}")

print('Clasificacion pro')
print(f"Reporte de clasificacion: {cr_pro}")

print('Clasificacion left')
print(f"Reporte de clasificacion: {cr_left}")

print('Clasificacion right')
print(f"Reporte de clasificacion: {cr_right}")


plt.show()

"""
predicciones_bi = []
predicciones_gender = []
predicciones_pro = []
predicciones_multi = []


i = 0
for item in vector_test:
    i+=1
    print(i)
    bi_predicted = clf_bi.predict(item)
    predicciones_bi.append(bi_predicted)
    predicciones_gender.append(clf_gender.predict(item))
    predicciones_pro.append(clf_pro.predict(item))
    if bi_predicted == 0:
        predicciones_multi.append(clf_left.predict(item))
    elif bi_predicted == 1:
        predicciones_multi.append(clf_right.predict(item))


#Creamos el corpus de salida
df_output_labeled = pd.DataFrame({'user': corpus_test['label'], 'genero': predicciones_gender, 'profesion': predicciones_pro, 'ideology_binary': predicciones_bi, 'ideology_multiclass': predicciones_multi})
df_output_labeled.to_csv('df_corpus_salida_multiclass_ioioioi.csv', index=False)
"""