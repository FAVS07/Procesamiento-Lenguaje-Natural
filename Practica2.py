import csv
import spacy
import pandas as pd
from spacy.lang.es import Spanish

#####leemos y convertimos el corpus_noticias a csv
df = pd.read_csv('corpus_noticias.txt', sep='&&&&&&&&')
df.to_csv('archivo.csv', index=False)

######## leemos el archivo.csv 
with open('archivo.csv', 'r', encoding='utf-8') as archivo:
	nlp = spacy.load("es_core_news_sm")
	lector_csv = csv.reader(archivo, delimiter=',')
	resultados = []
	for fila in lector_csv:
		stop =" "
		texto = fila[2]
		doc = nlp(texto)
		tokens = [token.text for token in doc]
		lemas = [token.lemma_ for token in doc ]
		#tokens_sin_stopwords = [token.lemma_ for token in doc ] 

		for token in doc:
			stopwords = token.pos_
			if(stopwords != "PRON" and stopwords != "CONJ" and stopwords !="ADP" and stopwords != "P" and stopwords !="DET"):
				stop= stop +token.lemma_+ " "

		resultados.append([stop])
####### lo escribimos en el archivo resultado.csv
with open('resultado.csv', 'w', newline='', encoding='utf-8') as archivo_salida:
	escritor_csv = csv.writer(archivo_salida, delimiter=',')
	for fila in resultados:
		escritor_csv.writerow(fila)

with open('resultado.csv', 'r',encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    with open('resultadofinal.txt', 'w',encoding='utf-8') as txt_file:
        for row in csv_reader:
            txt_file.write('\t'.join(row) + '\n\n')