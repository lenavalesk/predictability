#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import unidecode
import matplotlib
import scipy  
from scipy import special
import matplotlib.pyplot as plt
import seaborn as sns


def logit(p,n):
	n = 2*n
	if isinstance(p, int) or isinstance(p, float) :
		if p == 0:
			p = 1/n
		elif p == 1:
			p = 1-1/n
		logitpred = scipy.log10(p/(1-p))
	else:
		p[p == 0] = 1/n
		p[p == 1] = 1 - (1/n)
		logitpred = scipy.log10(p/(1-p))

	return logitpred

# Load
logFileName = 'log.csv'
columnas = ['dbID', 'sujID', 'mail', 'seqID', 'oracID', 'null', 'type', 'tStart', 'completada', 'original', 'tEnd', 'palNum', 'age']
log = pd.read_csv(logFileName, encoding='iso-8859-1',header=None, names=columnas, quotechar="'")
log.palNum = log.palNum+1
log['PalOrac'] = log.oracID*1000 + log.palNum
log.original = [re.sub('</p>', '', x) for x in log['original']]
simbolosElim = '[?!¿¡ç,."]'
log.original   = [re.sub(simbolosElim, '', x) for x in log['original']]
# para la completada tengo que salvar los casos (raros) donde no completaron nada
# Por su pusieron numeros, convierto a str todo
log.completada = [re.sub(simbolosElim, '', str(x) + ' ') for x in log['completada']]


completados = [unidecode.unidecode(x.lower().strip()) for x in log.completada]
originales  = [unidecode.unidecode(x.lower().strip()) for x in log.original]
log['iguales'] = [completados[i] == originales[i] for i,n in enumerate(completados)] 

# Elimino los mails de testeo
mailsToFilter = ['brunobian@gmail.com', 'dfslezak+aa@gmail.com', 'juan@test', 'test1@test.com', 'test2@test.com', 'test3@test.com', 'test4@test.com', 'test5@test.com', 'test6@test.com', 'test7@test.com']
for thisMail in mailsToFilter:
	log = log[log.mail != thisMail]

# Elimino los duplicados de
# -> Cuando envío la primer palabra sola
# -> Cuando por error de la DB alguno leyó dos veces la misma
log = log[-log.duplicated(subset=['mail', 'oracID', 'palNum'], keep='first')]

log = log[log.oracID<105]

grouppedPalOrac = log.groupby(['PalOrac'])

nCompletadas = grouppedPalOrac.count().iguales
preds    = grouppedPalOrac.mean().iguales

logit_preds    = grouppedPalOrac.mean().iguales
logit_preds[:] = [logit(i, nCompletadas.loc[nCompletadas.index[n]]) for n,i in enumerate(logit_preds)]

grouppedSuj = log.groupby('mail')
# print(grouppedSuj.count().iguales)


oracFileName = 'textos.csv'
columnas = ['oracID', 'type', 'null', 'oracStr']
oraciones = pd.read_csv(oracFileName, encoding='iso-8859-1',header=None, names=columnas,quotechar="<")
oraciones['oracStr'] = [re.sub('/p>', '', x) for x in oraciones['oracStr']]
oraciones['oracStr'] = [re.sub('p>',  '', x) for x in oraciones['oracStr']]
oraciones['lenght']  = [len(x.split(' ')) for x in oraciones['oracStr']]
#Tomo solo las oraciones de los cuentos
oraciones = oraciones[oraciones.oracID<105]


#Modifico textos para que tenga el numero de cuento igual que en contexto y el id de posicion de palabra en cuento
modiFileName = 'modificaciones.csv'
columns = ['palNumSt','st']
modificaciones = pd.read_csv(modiFileName,encoding='iso-8859-1',header=None, names=columns)

oraciones['id_text'] = modificaciones['st']
oraciones['palNumSt'] = modificaciones['palNumSt']


# Analsis por palabra
splitted = [x.split() for x in oraciones.oracStr]
oracID   = [[x]*oraciones.lenght[n] for n,x in enumerate(oraciones.oracID)]
id_text  = [[x]*oraciones.lenght[n] for n,x in enumerate(oraciones.id_text)]

palNumGobal  = [[x]*oraciones.lenght[n] for n,x in enumerate(oraciones.palNumSt)]
palNumGobal  = [i+n for x in palNumGobal for n,i in enumerate(x)]

palOrac  = [y*1000+n for x in oracID for n,y in enumerate(x)]
palNum   = [x % 1000 for x in palOrac]

oracID   = [y for x in oracID for y in x]
id_text  = [y for x in id_text for y in x]
splitted = [y for x in splitted for y in x]
splitted = list(map(lambda x: re.sub(simbolosElim, '' ,x), splitted))

#Load los analisis gramaticales
logGrammatical = '/home/lena/Documents/predictability/predictability/Texts_Data/grammatical_analysis.csv'
grammatical_analysis = pd.read_csv(logGrammatical, encoding='iso-8859-1')
print(grammatical_analysis)
gramm_tag = grammatical_analysis['tag']

diccionario = {'id_text':id_text,'palabra':splitted,'palNum':palNum,'palNumGobal':palNumGobal,'palOrac':palOrac, 'gramm_tag':gramm_tag}
df1 = pd.DataFrame(diccionario).set_index('palOrac')


result = pd.concat([df1, preds, logit_preds, nCompletadas], axis=1)
result.columns = ['id_text','palabra','palNum','palNumGobal','gramm_tag' ,'pred', 'logit_pred', 'nCompletadas']

result.to_csv('result.csv',index=False)



#Grafico algunas cosas para ver q onda

groupped = (result.groupby('gramm_tag').agg({'pred': ['mean', 'std']}))
print(groupped)
ax = sns.boxplot(x="gramm_tag", y="pred", data=result)
plt.show()