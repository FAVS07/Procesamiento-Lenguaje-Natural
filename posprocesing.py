import pandas as pd

corpus = pd.read_csv('df_corpus_salida_multiclass_ioioioi.csv')

mapa_genero = {
    '[0]': 'male', 
    '[1]': 'female'
}
mapa_profesiones = {
    '[0]': 'journalist',
    '[1]': 'celebrity',
    '[2]': 'politician'
}
mapa_bi_ideologia = {
    '[0]': 'left',
    '[1]': 'right'
}
mapa_multi_ideologia = {
    '[0]': 'left',
    '[1]': 'moderated_left',
    '[2]': 'moderated_right',
    '[3]': 'right'
}

corpus['genero'] = corpus['genero'].map(mapa_genero)
corpus['profesion'] = corpus['profesion'].map(mapa_profesiones)
corpus['ideology_binary'] = corpus['ideology_binary'].map(mapa_bi_ideologia)
corpus['ideology_multiclass'] = corpus['ideology_multiclass'].map(mapa_multi_ideologia)

corpus.to_csv('corpus_salida_decoded_ioioioi.csv', index=False)