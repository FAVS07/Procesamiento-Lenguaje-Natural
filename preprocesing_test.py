#IMPORTACIONES
import pandas as pd

import normalizacion
import lemma_token

from IPython.display import display
from sklearn import preprocessing

#Definimos la ruta del corpus
path_corpus = "politicES_phase_2_test_public.csv"

#Abrimos el corpus como un archivo de pandas
df_train = pd.read_csv(path_corpus)

lista_labels = df_train['label'].tolist()
print(len(set(lista_labels)))
input('Presione')

#Ahora agrupamos con base en el id
df_train_grouped = df_train.groupby('label')['tweet'].agg(lambda x: ' '.join(x)).reset_index()
df_train_grouped = df_train_grouped.rename(columns={'tweet': 'Tweets'})

df = df_train_grouped

tweets_normalizados = df_train_grouped['Tweets'].apply(normalizacion.normalizar_tweet)
df['Tweets'] = tweets_normalizados

corpus = []
for i in range(len(df)):
    print(f"Lematizando {i}")
    corpus.append(
        lemma_token.clean_text(df['Tweets'].loc[i])
    )       
df['Tweets'] = corpus

df.to_csv('df-test_corpus_normalized.csv', index=False)