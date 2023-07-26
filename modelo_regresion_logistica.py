import pandas as pd
import numpy as np
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

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
corpus_completo_clases = pd.concat([grupo_clases_left['Tweets'], grupo_clases_right['Tweets'], corpus_test['Tweets']], ignore_index=True)
corpus_completo = pd.concat([corpus_train['Tweets'], corpus_test['Tweets']], ignore_index=True)

#Realizamos la vectorizacion por TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=5, ngram_range=(1,2))
tweets_vectorizados = vectorizer.fit_transform(corpus_completo)
tweets_vectorizados_clases = vectorizer.fit_transform(corpus_completo_clases)

#Ahora separamos los datos en train y test de nuevo
longitud_corpus_train = corpus_train.shape[0]
longitud_corpus_train_clase_left = grupo_clases_left.shape[0]
longitud_corpus_train_clase_right = grupo_clases_right.shape[0]

tfidf_matriz_train_left = tweets_vectorizados_clases[:longitud_corpus_train_clase_left]
tfidf_matriz_train_right = tweets_vectorizados_clases[longitud_corpus_train_clase_left : longitud_corpus_train_clase_left + longitud_corpus_train_clase_right]
tfidf_matriz_test_clases = tweets_vectorizados_clases[longitud_corpus_train_clase_left + longitud_corpus_train_clase_right:]

tfidf_matriz_train = tweets_vectorizados[:longitud_corpus_train]
tfidf_matriz_test = tweets_vectorizados[longitud_corpus_train:]

#Separamos nuestros dataset
X_train_bi, X_test_bi, y_train_bi, y_test_bi = train_test_split(tfidf_matriz_train, corpus_train['Bi-Ideologia'], random_state=0, test_size=0.2)
X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(tfidf_matriz_train, corpus_train['Genero'], random_state=0, test_size=0.2)
X_train_pro, X_test_pro, y_train_pro, y_test_pro = train_test_split(tfidf_matriz_train, corpus_train['Profesion'], random_state=0, test_size=0.2)
X_train_left, X_test_left, y_train_left, y_test_left = train_test_split(tfidf_matriz_train_left, grupo_clases_left['Multi-Ideologia'], random_state=0, test_size=0.2)
X_train_right, X_test_right, y_train_right, y_test_right = train_test_split(tfidf_matriz_train_right, grupo_clases_right['Multi-Ideologia'], random_state=0, test_size=0.2)

#Instanciamos los modelos
clf_bi = LogisticRegression()
clf_gender = LogisticRegression()
clf_pro = LogisticRegression()
clf_left = LogisticRegression()
clf_right = LogisticRegression()

#Entrenamos los modelos
clf_bi.fit(X_train_bi, y_train_bi)
clf_gender.fit(X_train_gender, y_train_gender)
clf_pro.fit(X_train_pro, y_train_pro)
clf_left.fit(X_train_left, y_train_left)
clf_right.fit(X_train_right, y_train_right)

#Evaluamos los modelos con distintas métricas
cvs_bi = cross_val_score(clf_bi, X_train_bi, y_train_bi, cv=5)
cvs_gender = cross_val_score(clf_gender, X_train_gender, y_train_gender, cv=5)
cvs_pro = cross_val_score(clf_pro, X_train_pro, y_train_pro, cv=5)
cvs_left = cross_val_score(clf_left, X_train_left, y_train_left, cv=5)
cvs_right = cross_val_score(clf_right, X_train_right, y_train_right, cv=5)

score_bi = clf_bi.score(X_test_bi, y_test_bi)
score_gender = clf_gender.score(X_test_gender, y_test_gender)
score_pro = clf_pro.score(X_test_pro, y_test_pro)
score_left = clf_left.score(X_test_left, y_test_left)
score_right = clf_right.score(X_test_right, y_test_right)

mean_score_bi = cvs_bi.mean()
mean_score_gender = cvs_gender.mean()
mean_score_pro = cvs_pro.mean()
mean_score_left = cvs_left.mean()
mean_score_right = cvs_right.mean()

y_pred_bi = clf_bi.predict(X_test_bi)
y_pred_gender = clf_gender.predict(X_test_gender)
y_pred_pro = clf_pro.predict(X_test_pro)
y_pred_left = clf_left.predict(X_test_left)
y_pred_right = clf_right.predict(X_test_right)

acc_bi = accuracy_score(y_test_bi, y_pred_bi)
acc_gender = accuracy_score(y_test_gender, y_pred_gender)
acc_pro = accuracy_score(y_test_pro, y_pred_pro)
acc_left = accuracy_score(y_test_left, y_pred_left)
acc_right = accuracy_score(y_test_right, y_pred_right)

f1_bi = f1_score(y_test_bi, y_pred_bi)
f1_gender = f1_score(y_test_gender, y_pred_gender)
f1_pro = f1_score(y_test_pro, y_pred_pro, average='macro')
f1_left = f1_score(y_test_left, y_pred_left)
f1_right = f1_score(y_test_right, y_pred_right, average='macro')

print('Clasificacion bi')
print(f"Score: {score_bi}")
print(f"CV Score: {mean_score_bi}")
print(f"Accuracy: {acc_bi}")
print(f"F1: {f1_bi}")

print('Clasificacion gender')
print(f"Score: {score_gender}")
print(f"CV Score: {mean_score_gender}")
print(f"Accuracy: {acc_gender}")
print(f"F1: {f1_gender}")

print('Clasificacion pro')
print(f"Score: {score_pro}")
print(f"CV Score: {mean_score_pro}")
print(f"Accuracy: {acc_pro}")
print(f"F1: {f1_pro}")

print('Clasificacion left')
print(f"Score: {score_left}")
print(f"CV Score: {mean_score_left}")
print(f"Accuracy: {acc_left}")
print(f"F1: {f1_left}")

print('Clasificacion right')
print(f"Score: {score_right}")
print(f"CV Score: {mean_score_right}")
print(f"Accuracy: {acc_right}")
print(f"F1: {f1_right}")

predicciones_bi = []
predicciones_gender = []
predicciones_pro = []
predicciones_multi = []

i = 0
for item in tfidf_matriz_test:
    i+=1
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
df_output_labeled.to_csv('df_corpus_salida_multiclass.csv', index=False)