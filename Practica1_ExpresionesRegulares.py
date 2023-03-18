import pandas as pd
import re

data = pd.read_csv('tweetsdata2.csv')
x = data['Tweet'].values
print(data)

print(x)

numusuarios=0
numhashtag=0
numemoji=0
numfechas=0;
numhoras =0

"""## Buscar hashtags"""

patron = re.compile("\#+\w+")
for i in range(len(x)):
  resultado = patron.findall(x[i])
  numhashtag = numhashtag +len(resultado)
  print(resultado)

print(numhashtag)

"""## Buscar usuarios"""

for i in range(len(x)):
  patron = re.compile("\@+\w+")
  resultado = patron.findall(x[i])
  numusuarios = numusuarios + len(resultado)
  print(resultado)

print (numusuarios)

"""## Buscar Horas"""

Patron= re.compile("\s(\d{1,2}\:\d{1,2}\s?(?:AM|PM|am|pm)?)")
for i in range(len(x)):
  res = Patron.findall(x[i])
  numhoras = numhoras + len(res)
  print(res)

print(numhoras)

"""## Buscar fechas"""

Patron= re.compile("((\d+\/\d+)|(\d+\sde\s(?:.nero|.ebrero|.arzo|.bril|.unio|.ulio|.gosto|.eptiembre|.ctubre|.oviembre|.iciembre)))")
for i in range(len(x)):
  res = Patron.findall(x[i])
  numfechas = numfechas + len(res)
  print(res)

print(numfechas)

"""## Buscar emojis"""

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+"
)

for i in range(len(x)):
  resultado = EMOJI_PATTERN.findall(x[i])
  numemoji = numemoji + len(resultado)
  print(resultado)

print(numemoji)

"""## Tabla"""

print ("-------------------------------")
print ("Numero de Hashtag  = ",numhashtag)
print ("Numero de usuarios = ",numusuarios)
print ("Numero de Horas    = ",numhoras)
print ("Numero de fechas   = ",numfechas)
print ("Numero de emojies  = ",numemoji)
print ("-------------------------------")