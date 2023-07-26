import pandas as pd
import numpy as np
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from IPython.display import display

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
corpus_completo = pd.concat([grupo_clases_left['Tweets'], grupo_clases_right['Tweets'], corpus_test['Tweets']], ignore_index=True)

#Realizamos la vectorizacion por TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=5, ngram_range=(1,2))
tweets_vectorizados = vectorizer.fit_transform(corpus_completo)

#Ahora separamos los datos en train y test de nuevo
longitud_corpus_train_clase_left = grupo_clases_left.shape[0]
longitud_corpus_train_clase_right = grupo_clases_right.shape[0]

tfidf_matriz_train_left = tweets_vectorizados[:longitud_corpus_train_clase_left]
tfidf_matriz_train_right = tweets_vectorizados[longitud_corpus_train_clase_left : longitud_corpus_train_clase_left + longitud_corpus_train_clase_right]
tfidf_matriz_test = tweets_vectorizados[longitud_corpus_train_clase_left + longitud_corpus_train_clase_right:]

#Separamos nuestros dataset
X_train_left, X_test_left, y_train_left, y_test_left = train_test_split(tfidf_matriz_train_left, grupo_clases_left['Multi-Ideologia'], random_state=0, test_size=0.2)
X_train_right, X_test_right, y_train_right, y_test_right = train_test_split(tfidf_matriz_train_right, grupo_clases_right['Multi-Ideologia'], random_state=0, test_size=0.2)

#Instanciamos los modelos
clf_left = LogisticRegression()
clf_right = LogisticRegression()

#Entrenamos los modelos
clf_left.fit(X_train_left, y_train_left)
clf_right.fit(X_train_right, y_train_right)

#Evaluamos los modelos con distintas métricas
cvs_left = cross_val_score(clf_left, X_train_left, y_train_left, cv=5)
cvs_right = cross_val_score(clf_right, X_train_right, y_train_right, cv=5)

score_left = clf_left.score(X_test_left, y_test_left)
score_right = clf_right.score(X_test_right, y_test_right)

mean_score_left = cvs_left.mean()
mean_score_right = cvs_right.mean()

y_pred_left = clf_left.predict(X_test_left)
y_pred_right = clf_right.predict(X_test_right)

acc_left = accuracy_score(y_test_left, y_pred_left)
acc_right = accuracy_score(y_test_right, y_pred_right)

print('Clasificacion left')
print(f"Score: {score_left}")
print(f"CV Score: {mean_score_left}")
print(f"Accuracy: {acc_left}")

print('Clasificacion right')
print(f"Score: {score_right}")
print(f"CV Score: {mean_score_right}")
print(f"Accuracy: {acc_right}")

"""
#Procedemos a hacer las predicciones
predicciones_bi = clf_bi.predict(tfidf_matriz_test)
predicciones_gender = clf_gender.predict(tfidf_matriz_test)
predicciones_pro = clf_pro.predict(tfidf_matriz_test)

#Creamos el corpus de salida
df_output_labeled = pd.DataFrame({'user': corpus_test['label'], 'genero': predicciones_gender, 'profesion': predicciones_pro, 'ideology_binary': predicciones_bi})
df_output_labeled.to_csv('df_corpus_salida.csv', index=False)"""