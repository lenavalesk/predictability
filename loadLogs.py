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
 
%pylab

toPlot = 0
export = 1
sorteo = 1

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
mailsToFilter = ['brunobian@gmail.com', 'brunobian2@gmail.com', 'asd@asd.com', 'lena.peroni@gmail.com']
for thisMail in mailsToFilter:
	log = log[log.mail != thisMail]

# Elimino los duplicados de
# -> Cuando envío la primer palabra sola
# -> Cuando por error de la DB alguno leyó dos veces la misma
log = log[-log.duplicated(subset=['mail', 'oracID', 'palNum'], keep='first')]

grouppedPalOrac = log.groupby(['PalOrac'])
preds    = grouppedPalOrac.mean().iguales
nCompletadas = grouppedPalOrac.count().iguales

preds[:] = [logit(i, nCompletadas.loc[nCompletadas.index[n]]) for n,i in enumerate(preds)]

grouppedOracID = log.groupby(['oracID', 'mail'])
cantPalsCompletadas = grouppedOracID.count().iguales

grouppedSuj = log.groupby('mail')
# print(grouppedSuj.count().iguales)

oracFileName = 'textos.csv'
columnas = ['oracID', 'type', 'null', 'oracStr']
oraciones = pd.read_csv(oracFileName, encoding='iso-8859-1',header=None, names=columnas,quotechar="<")
oraciones['oracStr'] = [re.sub('/p>', '', x) for x in oraciones['oracStr']]
oraciones['oracStr'] = [re.sub('p>',  '', x) for x in oraciones['oracStr']]
oraciones['lenght']  = [len(x.split(' ')) for x in oraciones['oracStr']]

# Analsis por palabra
splitted = [x.split() for x in oraciones.oracStr]
oracID   = [len(x)*[n] for n,x in enumerate(splitted)]
palOrac  = [ y*1000+n for x in oracID for n,y in enumerate(x)]
palNum   = [x % 1000 for x in palOrac]
oracIDJoin   = [y for x in oracID for y in x]
splittedJoin = [y for x in splitted for y in x]
splittedJoin = list(map(lambda x: re.sub(simbolosElim, '' ,x), splittedJoin))

diccionario = {'original': splittedJoin, 'palOrac': palOrac, 'oracID':oracIDJoin, 'palNum': palNum}
df1 = pd.DataFrame(diccionario)
df1 = df1.set_index('palOrac')

result = pd.concat([df1, preds, nCompletadas], axis=1)
result.columns = ['oracID', 'original', 'palNum', 'pred', 'nCompletadas']
result.pred    = result.pred.fillna(logit(0, 50))


pred = pd.concat([result.pred, result.nCompletadas], axis=1) 
if export:
	cantPalsCompletadas.to_csv('completadas')
	pred.to_csv('preds.csv', header = False)
	result.to_csv('result.csv',na_rep='0')

if toPlot:	
	# Hacer plots de distribucion de completadas
	# Hacer plots de distribucion de predictibilidad (histograma)
	# Hacer plots de correlacion posicion de la pal vs pred
	for i in set(result.oracID):
		fname = 'figs/'+str(i)
		thisSntc = result[result.oracID==i]
		plot(thisSntc.palNum, thisSntc.pred)

		thisSntc['nCompletadas'] = thisSntc.nCompletadas.apply(str)
		thisSntc['palNum'] = thisSntc.palNum.apply(str)
		thisSntc['titles'] = thisSntc[['palNum', 'original', 'nCompletadas']].apply(lambda x: ' - '.join(x), axis=1)
		xticks( arange(len(thisSntc.original)), thisSntc.titles, rotation=90 )
		ylim( -2, 2 )  
		plt.gcf().subplots_adjust(bottom=0.35)
		savefig(fname)
		close()


# Analisis por oracion (para sorteos)
from collections import Counter

puntaje = dict((el,0) for el in set(log.mail))
nOrac   = dict((el,0) for el in set(log.mail))
lista = []
if sorteo:
	set(log.mail)
	for index, row in oraciones.iterrows():
		sujsThisSntc = log.mail[log.oracID == row.oracID]
		lenThisSntc  = row.lenght-1

		count = Counter(sujsThisSntc)
		for i in count:
			if count[i] == lenThisSntc:
				nOrac[i] = nOrac[i] + 1

	for i in puntaje:
		puntaje[i] = (nOrac[i] + 25*(nOrac[i]//10)) * ((nOrac[i]//315) + 1) 
		lista = lista + ([i] * puntaje[i])

	s = 0
	for i in puntaje:
		s = s + puntaje[i]

	if len(lista) != s:
		print('Ojo que hay algo mal en el calculo')
	else:			
		import random
		winner = random.choice(lista) 
		print('El ganador o la ganadora del sorteo es:')
		print()
		print(winner)
		print('Con un total de ' + str(nOrac[winner]) + ' oraciones completadas')
