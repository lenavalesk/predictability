#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import nltk
import numpy as np
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')

def cantPalsMayorA(frase, n):
    return sum([1 for x in frase if len(x) > n])

def hayNumero(frase):
    return any([char.isdigit() for char in frase])

cuentos = pd.read_csv('cuentos.csv', index_col=0)

spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

numbers           = list(range(0,2000))
str_numbers       = [str(w) for w in numbers]
not_stopwords     = ['estoy', 'estás', 'está', 'estamos', 'estáis', 'están', 'esté', 'estés', 'estemos', 'estéis', 'estén', 'estaré', 'estarás', 'estará', 'estaremos', 'estaréis', 'estarán', 'estaría', 'estarías', 'estaríamos', 'estaríais', 'estarían', 'estaba', 'estabas', 'estábamos', 'estabais', 'estaban', 'estuve', 'estuviste', 'estuvo', 'estuvimos', 'estuvisteis', 'estuvieron', 'estuviera', 'estuvieras', 'estuviéramos', 'estuvierais', 'estuvieran', 'estuviese', 'estuvieses', 'estuviésemos', 'estuvieseis', 'estuviesen', 'estando', 'estado', 'estada', 'estados', 'estadas', 'estad', 'he', 'has', 'ha', 'hemos', 'habéis', 'han', 'haya', 'hayas', 'hayamos', 'hayáis', 'hayan', 'habré', 'habrás', 'habrá', 'habremos', 'habréis', 'habrán', 'habría', 'habrías', 'habríamos', 'habríais', 'habrían', 'había', 'habías', 'habíamos', 'habíais', 'habían', 'hube', 'hubiste', 'hubo', 'hubimos', 'hubisteis', 'hubieron', 'hubiera', 'hubieras', 'hubiéramos', 'hubierais', 'hubieran', 'hubiese', 'hubieses', 'hubiésemos', 'hubieseis', 'hubiesen', 'habiendo', 'habido', 'habida', 'habidos', 'habidas', 'soy', 'eres', 'es', 'somos', 'sois', 'son', 'sea', 'seas', 'seamos', 'seáis', 'sean', 'seré', 'serás', 'será', 'seremos', 'seréis', 'serán', 'sería', 'serías', 'seríamos', 'seríais', 'serían', 'era', 'eras', 'éramos', 'erais', 'eran', 'fui', 'fuiste', 'fue', 'fuimos', 'fuisteis', 'fueron', 'fuera', 'fueras', 'fuéramos', 'fuerais', 'fueran', 'fuese', 'fueses', 'fuésemos', 'fueseis', 'fuesen', 'sintiendo', 'sentido', 'sentida', 'sentidos', 'sentidas', 'siente', 'sentid', 'tengo', 'tienes', 'tiene', 'tenemos', 'tenéis', 'tienen', 'tenga', 'tengas', 'tengamos', 'tengáis', 'tengan', 'tendré', 'tendrás', 'tendrá', 'tendremos', 'tendréis', 'tendrán', 'tendría', 'tendrías', 'tendríamos', 'tendríais', 'tendrían', 'tenía', 'tenías', 'teníamos', 'teníais', 'tenían', 'tuve', 'tuviste', 'tuvo', 'tuvimos', 'tuvisteis', 'tuvieron', 'tuviera', 'tuvieras', 'tuviéramos', 'tuvierais', 'tuvieran', 'tuviese', 'tuvieses', 'tuviésemos', 'tuvieseis', 'tuviesen', 'teniendo', 'tenido', 'tenida', 'tenidos', 'tenidas', 'tened']
stopwords_numbers = stopwords.words('spanish') + ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas'] + str_numbers +  ['']

nombres = ['rebeca', 'biblia', 'eclesiastes', 'nina', 'qaddish', 'axolotl', 'negro', 'paris', 'buenos aires', 'sociedad protectora de animales', 'francia', 'pulqui', 'casimiro', 'andrée', 'sara', 'suipacha', 'conejitos', 'conejito', 'paris', 'molina', 'lópez', 'cansino', 'loco', 'sparta', 'suárez', 'wernicke', 'oliveros', 'federación', 'bob', 'roberto', 'inés', 'junta militar', 'isabel', 'córdoba', 'martínez de hoz', 'buenos aires', 'massera', 'lyell', 'wallace', 'hooker', 'especie', 'especies']

for index, row in cuentos.iterrows():
    esteCuento_tokenizado             = spanish_tokenizer.tokenize(row['texto'])
    df_esteCuento                     = pd.DataFrame(esteCuento_tokenizado,columns = ['oracion'])
    df_esteCuento['nombre']           = row['nombre']
    df_esteCuento['numero_oracion']   = df_esteCuento.index
    df_esteCuento['cant_palabra']     = df_esteCuento['oracion'].str.count(' ') + 1
    df_esteCuento['cant_caracteres']  = df_esteCuento['oracion'].str.len()

    propMayor2  = []
    propMayor3  = []
    contOracion = []
    hay_Numero  = []
    nombre_Repe = []

    for index2, oracion in df_esteCuento.iterrows():
        strEstaOracion  = oracion.oracion.lower()   #oracion.oracion es una oracion del cuento como str
        strEstaOracion  = [w for w in strEstaOracion if (w.isalpha() or w == ' ')]
        strEstaOracion  = ''.join(strEstaOracion).split(' ')  #strEstaOracion es una lista de las palabras de una oracion particular
        contEstaOracion = [w for w in strEstaOracion if (w not in stopwords_numbers or w in not_stopwords)]
        repeEstaOracion = [w for w in strEstaOracion if (w in nombres)]
        propMayor2      = propMayor2 + [cantPalsMayorA(strEstaOracion, 2)/len(strEstaOracion)]
        propMayor3      = propMayor3 + [cantPalsMayorA(strEstaOracion, 3)/len(strEstaOracion)]
        contOracion     = contOracion + [contEstaOracion]
        hay_Numero      = hay_Numero + [hayNumero(oracion['oracion'])]
        nombre_Repe     = nombre_Repe + [repeEstaOracion]

    df_esteCuento['propMayor2']     = propMayor2
    df_esteCuento['propMayor3']     = propMayor3
    df_esteCuento['palContenido']   = contOracion
    df_esteCuento['NpalContenido']  = df_esteCuento['palContenido'].str.len()
    df_esteCuento['Numeros']        = hay_Numero
    df_esteCuento['nombreRepetido'] =  nombre_Repe

    if index == 0:
        df_todosLosCuentos = df_esteCuento
    else:
        df_todosLosCuentos = pd.concat([df_todosLosCuentos, df_esteCuento])


df_todosLosCuentos.reset_index(inplace=True, drop=True)

df_cuentosFiltrados = df_todosLosCuentos.query('cant_palabra >= 6 & NpalContenido >= 4 & cant_caracteres <= 1000 ')

#print(df_todosLosCuentos)

df_todosLosCuentos.to_csv('todosLosCuentos.csv')
df_cuentosFiltrados.to_csv('cuentosFiltrados.csv')

