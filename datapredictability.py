# -*- coding: utf-8 -*-
import pyreadr

# Importo las librerias necesarias
import pandas as pd
import numpy as np
import re
import unidecode
import matplotlib
import scipy  
from scipy import special
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import scipy.stats as stats
import nltk
from nltk.corpus import webtext
from nltk.probability import FreqDist
import wordfreq


"""#Funciones Utiles"""

# Funcion para calcular logit pred
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

def shannon_nuestra(pred):
  H = -(pred * np.log(pred))
  return H

"""#Los Cuentos

Carga un archivo que se llama *textos.csv* que tiene las oraciones que testeamos aisladas, 105 son de los cuentos. 
Terminamos con un dataframe **df1** que tiene: `'original', 'oracID', 'palNum', 'palNumGlobal', 'text_id'.`

El type corresponde a si viene de los cuentos o de los proverbios. Null es de que cuento viene.
"""

#Cargamos el archivo con los textos que usamos y lo limpiamos
oracFileName = "Texts_Data/textos.csv"
columnas = ['oracID', 'type', 'null', 'oracStr']
oraciones = pd.read_csv(oracFileName, encoding='iso-8859-1',header=None, names=columnas,quotechar="<")
oraciones['oracStr'] = [re.sub('/p>', '', x) for x in oraciones['oracStr']]
oraciones['oracStr'] = [re.sub('p>',  '', x) for x in oraciones['oracStr']]
oraciones['lenght']  = [len(x.split(' ')) for x in oraciones['oracStr']]
oraciones = oraciones.truncate(after=104) #Corto para que solo incluya cuentos

#Modifico textos para que tenga el numero de cuento igual que cloze en contexto
modiFileName = "ClozeTask_Data/4_Stories_sentences/raw/modificaciones.csv"
columns               = ['palNumSt','st']
modificaciones        = pd.read_csv(modiFileName,encoding='iso-8859-1',header=None, names=columns)
oraciones['type']     = modificaciones['st']
oraciones['posFirst'] = modificaciones['palNumSt'] #Esto no me acuerdo bien que era


simbolosElim = '[?!¿¡ç,."]'
# Analsis por palabra
splitted      = [x.split() for x in oraciones.oracStr]
oracID        = [len(x)*[n] for n,x in enumerate(splitted)]
palOrac       = [ y*1000+n for x in oracID for n,y in enumerate(x)]
palNum        = [x % 1000 for x in palOrac]
oracIDJoin    = [y for x in oracID for y in x]
splittedJoin  = [y for x in splitted for y in x]
splittedJoin  = list(map(lambda x: re.sub(simbolosElim, '' ,x), splittedJoin))
palNumGlobal  = [[x]*oraciones.lenght[n] for n,x in enumerate(oraciones.posFirst)]
palNumGlobal  = [[i+n for n,i in enumerate(x)] for x in palNumGlobal ]
palNumGlobal  = [num for sublist in palNumGlobal for num in sublist]
text_id       = [[x]*oraciones.lenght[n] for n,x in enumerate(oraciones.type)]
text_id       = [num for sublist in text_id for num in sublist]



diccionario = {'original': splittedJoin, 'palOrac': palOrac, 'oracID':oracIDJoin, 'palNum': palNum, 'palNumGlobal':palNumGlobal, 'text_id':text_id}

df1 = pd.DataFrame(diccionario)
df1 = df1.set_index('palOrac')
df1['original']  = [unidecode.unidecode(x.lower().strip()) for x in df1.original]



grouppeddf1 = df1.groupby(['text_id'])
N = [] #N es una lista que va a tener la cantidad de palabras en cada particion (cuentos)
for cuento in grouppeddf1:
  N.append(len(cuento[1]))


n_counts = pd.DataFrame(grouppeddf1['original'].value_counts()) #cuento cuantas veces aparece una pal en un cuento
conteo = pd.DataFrame(n_counts.unstack())
conteo = conteo['original'].reset_index()
conteo.drop(columns='text_id')
conteo.fillna(0, inplace=True)
data_dict = conteo.to_dict('list')


"""# Data Oraciones

Abre la data como un df **log** que tiene: `'dbID', 'sujID', 'mail', 'seqID', 'oracID', 'null', 'type', 'tStart', 'completada', 'original', 'tEnd', 'palNum', 'age'`.
"""

path_log = "ClozeTask_Data/4_Stories_sentences/raw/log.csv"

# Cargamos los datos y arma un df que tiene mucha data que no nos interesa ahora
columnas = ['dbID', 'sujID', 'mail', 'seqID', 'oracID', 'null', 'type', 'tStart', 'completada', 'original', 'tEnd', 'palNum', 'age']
log = pd.read_csv(path_log, encoding='iso-8859-1',header=None, names=columnas, quotechar="'")



# Elimino los mails de testeo
mailsToFilter = ['brunobian@gmail.com', 'brunobian2@gmail.com', 'asd@asd.com', 'lena.peroni@gmail.com', 'test@test.com' ] #Deberia volver a recorrer para ver si hay nuevos datos basura
for thisMail in mailsToFilter:
	log = log[log.mail != thisMail]

log.palNum = log.palNum+1       #palNum es el numero de la palabra en la oracion
log['PalOrac'] = log.oracID*1000 + log.palNum

# En log original esta la palabra que era
log.original = [re.sub('</p>', '', x) for x in log['original']]
simbolosElim = '[?!¿¡ç,."]'
log.original   = [re.sub(simbolosElim, '', x) for x in log['original']]

# para la completada tengo que salvar los casos (raros) donde no completaron nada
# Por su pusieron numeros, convierto a str todo
log.completada = [re.sub(simbolosElim, '', str(x) + ' ') for x in log['completada']]

# Iguales va a tener un booleano de si la palabra completada es igual a la original
completados = [unidecode.unidecode(x.lower().strip()) for x in log.completada]
originales  = [unidecode.unidecode(x.lower().strip()) for x in log.original]
log['iguales'] = [completados[i] == originales[i] for i,n in enumerate(completados)] 


# Elimino los duplicados de
# -> Cuando envío la primer palabra sola
# -> Cuando por error de la DB alguno leyó dos veces la misma
log = log[-log.duplicated(subset=['mail', 'oracID', 'palNum'], keep='first')]


#Calculo de pred y logit pred
grouppedPalOrac = log.groupby(['PalOrac'])
preds    = grouppedPalOrac.mean().iguales
nCompletadas = grouppedPalOrac.count().iguales
preds[:] = [logit(i, nCompletadas.loc[nCompletadas.index[n]]) for n,i in enumerate(preds)]

#El df preds tiene dos columnas, logitpred y pred. El indice es PalOrac
preds = pd.DataFrame(preds)
preds['pred'] = grouppedPalOrac.mean().iguales 
#Corto para que solo incluya los cuentos
preds        = preds.truncate(after=104013)


grouppedOracID = log.groupby(['oracID', 'mail'])
cantPalsCompletadas = grouppedOracID.count().iguales
grouppedSuj = log.groupby('mail') #Cuantas palabras completo un sujeto

nuevo = log[['PalOrac', 'completada']].copy()
nuevo.set_index('PalOrac')

"""Arma el DF con los resultados"""

nCompletadas = nCompletadas.truncate(after=104013) #Solo cuentos
result = pd.concat([df1, preds, nCompletadas], axis=1)
result.columns = ['palabra','orac', 'palNum','palNumGlobal','text_id','logit_pred','pred', 'nCompletadas']
result.logit_pred    = result.logit_pred.fillna(logit(0, 50))
result['repetition']=result.groupby(['text_id', 'orac','palabra'])['palabra'].transform('count') 


pred = pd.concat([result.text_id, result.palNumGlobal, result.palabra,result.pred,result.logit_pred, result.nCompletadas, result.palNum, result.repetition], axis=1)
pred['pred'] = pred['pred'].replace(to_replace=0, value=0.005952380952380952)
pred['surprisal'] = -np.log(pred['pred']) #surprisal measures the relative unexpectedness of a word in context
# pred['repetition'] = pred.groupby(['text_id','original']).transform('count')


# m = min(i for i in pred['pred'] if i > 0)
# m/2

#Limpieza de las respuestas, falta chequear con un diccionario
#Los proverbios empiezan en 104013
simbolosElim = '[?!¿¡ç,.= + _ -"]'
data = log[['PalOrac','completada','original']].copy()


data['correccion'] = [unidecode.unidecode(x.lower().strip()) for x in data['completada']] #Por ahora solo saque tildes y elimine mayusculas
data['correccion']  = [re.sub(simbolosElim, '', x) for x in data['correccion']]

respuestas = pd.DataFrame(data.groupby(['PalOrac', 'original'])['correccion'].value_counts()) #Tengo la palabra original, la completada y cuantas veces la completaron
respuestas.columns = ['count']
respuestas = respuestas.reset_index()
respuestas = respuestas.set_index('PalOrac')
respuestas = respuestas.truncate(after=104013)


respuestas['pred'] = respuestas['count']/respuestas.groupby('PalOrac')['count'].transform('sum')
respuestas['entropy'] = shannon_nuestra(respuestas['pred'])  
respuestas['surprisal'] = -np.log(respuestas['pred']) #surprisal measures the relative unexpectedness of a word in context



pred['entropy'] = respuestas.groupby(['PalOrac'])['entropy'].sum()*-1 
print(pred)

# """#Data Stories"""

# stories_full = pd.read_csv("ClozeTask_Data/3_Stories_full/data_all/predictability.csv", index_col=0)
# stories_full

# simbolosElim = '[?!¿¡ç %.= + _ - 1 2 3 4 5 6 7 8 9 0 "]'

# stories_full['correccion'] = [unidecode.unidecode(x.lower().strip()) for x in stories_full['words']] #Por ahora solo saque tildes y elimine mayusculas
# stories_full['correccion'] = stories_full['correccion'].str.split(',') #Saco las comas, falta el whitespace


# for count1, i in enumerate(stories_full['correccion']):
#   stories_full['correccion'][count1+1] = [x for x in i if x]    #Saco los str vacios

# for count1, i in enumerate(stories_full['correccion']):
#    stories_full['correccion'][count1+1] = [algo.split()[0] for algo in i] #Saco los espacios en blanco y cuando hay mas de una palabra en la rta me quedo con la primera

# for count1, i in enumerate(stories_full['correccion']):
#   stories_full['correccion'][count1+1] = [re.sub(simbolosElim, '', algo) for algo in i]

# for count1, i in enumerate(stories_full['correccion']):
#   stories_full['correccion'][count1+1] = [x for x in i if x]

# stories_full['nPred'] = [len(i) for i in stories_full['correccion']]
# stories_full

# stories_full['pred'] = [lista.count(original)/len(lista) for (lista, original) in zip (stories_full['correccion'], stories_full['originales'])]
# stories_full['pred'] = stories_full['pred'].replace(to_replace=0, value=0.015151515151515152)

# # m = min(i for i in stories_full['pred'] if i > 0)
# # m/2 #0.015151515151515152

# from collections import Counter
# from operator import add

# stories_desarmado = pd.concat([pd.DataFrame(list(map(add, list(Counter(lista).items()), [(original,tId,index)]*len(set(lista)))), columns=['respuesta', 'count', 'original', 'tId', 'index']) for (lista, original, tId, index) in zip(stories_full['correccion'], stories_full['originales'], stories_full['tId'], stories_full.index)])

# stories_desarmado[0:20]

# stories_desarmado['pred'] = stories_desarmado['count']/stories_desarmado.groupby('index')['count'].transform('sum')
# stories_desarmado['entropy'] = shannon_nuestra(stories_desarmado['pred'])  
# stories_desarmado['pred'] = stories_desarmado['pred'].replace(to_replace=0, value=0.013513513513513514)
# stories_desarmado['surprisal'] = -np.log(stories_desarmado['pred']) #surprisal measures the relative unexpectedness of a word in context

# stories_desarmado['nRTa'] = stories_desarmado.groupby(['index'])['count'].transform('sum')


# stories_full['entropy'] = stories_desarmado.groupby(['index'])['entropy'].sum()*-1 
# stories_full['surprisal'] = -np.log(stories_full['pred']) #surprisal measures the relative unexpectedness of a word in context
# stories_full['nRTa'] = stories_desarmado.groupby(['index'])['nRTa'].mean()

# stories_full

# log_full = stories_full[['tId', 'originales','misWords', 'pred', 'entropy', 'surprisal', 'nRTa']].copy()

# log_full.columns = ['text_id','palabra','palNum','pred','entropy','surprisal', 'nCompletadas']
# # log_full

# log_full = log_full[log_full.nCompletadas > 10]

# log_full

# log_full['repetition'] = log_full.groupby(['text_id','palabra'])['palabra'].transform('count') #Cuantas veces se repite la palabra en el cuento
# log_full['wordlen'] = [len(x) for x in log_full.palabra]
# log_full['repetition'].max()

# orac_stories = pd.merge(log_full, pred, left_on=['text_id', 'palNum'], right_on=['text_id', 'palNumGlobal'])
# orac_stories

# ########Agrupo por numero de palabra en oracion############

# groupped = (
#     orac_stories.groupby('palNum_y').agg({'pred_x': ['mean', 'count', 'std'], 'pred_y': ['mean', 'count', 'std'], 'entropy_x': ['mean', 'count', 'std'], 'entropy_y': ['mean', 'count', 'std'], 'surprisal_x': ['mean', 'count', 'std'], 'surprisal_y': ['mean', 'count', 'std']})).dropna()

# groupped['error_pred_x'] = (groupped['pred_x']['std']) / np.sqrt(groupped['pred_x']['count'])
# groupped['error_pred_y'] = (groupped['pred_y']['std']) / np.sqrt(groupped['pred_y']['count'])

# groupped['error_entropy_x'] = (groupped['entropy_x']['std']) / np.sqrt(groupped['entropy_x']['count'])
# groupped['error_entropy_y'] = (groupped['entropy_y']['std']) / np.sqrt(groupped['entropy_y']['count'])

# groupped['error_surprisal_x'] = (groupped['surprisal_x']['std']) / np.sqrt(groupped['surprisal_x']['count'])
# groupped['error_surprisal_y'] = (groupped['surprisal_y']['std']) / np.sqrt(groupped['surprisal_y']['count'])


# ########Agrupo por largo de palabra###############

# groupped2 = (
#     orac_stories.groupby('wordlen').agg({'pred_x': ['mean', 'count', 'std'], 'pred_y': ['mean', 'count', 'std'], 'entropy_x': ['mean', 'count', 'std'], 'entropy_y': ['mean', 'count', 'std'], 'surprisal_x': ['mean', 'count', 'std'], 'surprisal_y': ['mean', 'count', 'std']})).dropna()

# groupped2['error_pred_x'] = (groupped2['pred_x']['std']) / np.sqrt(groupped2['pred_x']['count'])
# groupped2['error_pred_y'] = (groupped2['pred_y']['std']) / np.sqrt(groupped2['pred_y']['count'])

# groupped2['error_entropy_x'] = (groupped2['entropy_x']['std']) / np.sqrt(groupped2['entropy_x']['count'])
# groupped2['error_entropy_y'] = (groupped2['entropy_y']['std']) / np.sqrt(groupped2['entropy_y']['count'])

# groupped2['error_surprisal_x'] = (groupped2['surprisal_x']['std']) / np.sqrt(groupped2['surprisal_x']['count'])
# groupped2['error_surprisal_y'] = (groupped2['surprisal_y']['std']) / np.sqrt(groupped2['surprisal_y']['count'])

# ########Agrupo por numero de repeticion######### 

# groupped3 = (
#     orac_stories.groupby('repetition_x').agg({'pred_x': ['mean', 'count', 'std'], 'pred_y': ['mean', 'count', 'std'], 'entropy_x': ['mean', 'count', 'std'], 'entropy_y': ['mean', 'count', 'std'], 'surprisal_x': ['mean', 'count', 'std'], 'surprisal_y': ['mean', 'count', 'std']})).dropna()

# groupped3['error_pred_x'] = (groupped3['pred_x']['std']) / np.sqrt(groupped3['pred_x']['count'])
# groupped3['error_pred_y'] = (groupped3['pred_y']['std']) / np.sqrt(groupped3['pred_y']['count'])

# groupped3['error_entropy_x'] = (groupped3['entropy_x']['std']) / np.sqrt(groupped3['entropy_x']['count'])
# groupped3['error_entropy_y'] = (groupped3['entropy_y']['std']) / np.sqrt(groupped3['entropy_y']['count'])

# groupped3['error_surprisal_x'] = (groupped3['surprisal_x']['std']) / np.sqrt(groupped3['surprisal_x']['count'])
# groupped3['error_surprisal_y'] = (groupped3['surprisal_y']['std']) / np.sqrt(groupped3['surprisal_y']['count'])

# # pred.to_csv('data_orac.csv')
# # data_orac.csv "drive/My Drive/Cuarentesis"
# # log_full.to_csv('data_full.csv')
# # data_full.csv "drive/My Drive/Cuarentesis"
# # orac_stories.to_csv('data_oraciones_full.csv')
# # data_oraciones_full.csv "drive/My Drive/Cuarentesis"

# data_analysis = nltk.FreqDist(pred.palabra)
# # data_analysis.plot(40, cumulative=False)
# orac_stories


# """#Graficamos algunas cosas"""

# # ax = sns.scatterplot(x=pred['pred'], y=pred['entropy'])
# # ax.set_title("Pred vs. Entropy")
# # ax.set_xlabel("pred")
# # sns.lmplot(x='pred', y='entropy', data=pred)
# # plt.set_title("Pred vs. Entropy")
# size_letters = 'large'

# sns.lmplot(x="pred", y="entropy", data=pred)
# plt.xlabel(xlabel='Predictability', fontsize=size_letters)
# plt.ylabel(ylabel='Entropy', fontsize=size_letters)

# #No deberia ser que la entropia va tendiendo a 0 a medida que aumenta la pred?

# sns.lmplot(x="surprisal", y="pred", data=log_full);
# plt.xlabel(xlabel='Surprisal', fontsize=size_letters)
# plt.ylabel(ylabel='Pred', fontsize=size_letters)

# # ax = sns.scatterplot(x=respuestas['surprisal'], y=respuestas['entropy'])
# # ax.set_title("Surpsisal vs. Entropy")
# # ax.set_xlabel("Surprisal")
# # #sns.lmplot(x='surprisal', y='entropy', data=respuestas)
# sns.lmplot(x="surprisal", y="entropy", data=pred)
# plt.xlabel(xlabel='Surprisal', fontsize=size_letters)
# plt.ylabel(ylabel='Entropy', fontsize=size_letters)

# images_dir = '/content/drive/My Drive/Cuarentesis'


# plt.rcParams['figure.figsize'] = [25, 20]
# size_letters = 50



# plt.errorbar(groupped.index + 0.1, groupped['pred_x']['mean'], yerr=groupped['error_pred_x'],linewidth=3, label='Contextualizada')
# plt.errorbar(groupped.index, groupped['pred_y']['mean'], yerr=groupped['error_pred_y'],linewidth=3, label='Oraciones Aisladas')


# #plt.title(label="Stories")
# plt.xlabel(xlabel='Word Position', fontsize=size_letters)
# plt.ylabel(ylabel='Predictability', fontsize=size_letters)
# plt.xticks(fontsize=size_letters)
# plt.yticks(fontsize=size_letters)
# plt.ylim(0.0, 0.7)

# plt.savefig(f"{images_dir}/pred-pos.svg", transparent = True)

# plt.errorbar(groupped2.index + 0.1, groupped2['pred_x']['mean'], yerr=groupped2['error_pred_x'],linewidth=3, label='Contextualizada')
# plt.errorbar(groupped2.index, groupped2['pred_y']['mean'], yerr=groupped2['error_pred_y'],linewidth=3, label='Oraciones Aisladas')


# #plt.title(label="Stories")
# plt.xlabel(xlabel='Word Length', fontsize=size_letters)
# plt.ylabel(ylabel='Predictability', fontsize=size_letters)
# plt.xticks(fontsize=size_letters)
# plt.yticks(fontsize=size_letters)
# plt.ylim(0.0, 0.7)
# plt.savefig(f"{images_dir}/pred-len.svg", transparent = True)

# plt.errorbar(groupped.index + 0.1, groupped['surprisal_x']['mean'], yerr=groupped['error_surprisal_x'],linewidth=3, label='Contextualizada')
# plt.errorbar(groupped.index, groupped['surprisal_y']['mean'], yerr=groupped['error_surprisal_y'],linewidth=3, label='Oraciones Aisladas')


# #plt.title(label="Stories")
# plt.xlabel(xlabel='Word Position', fontsize=size_letters)
# plt.ylabel(ylabel='Surprisal', fontsize=size_letters)
# plt.xticks(fontsize=size_letters)
# plt.yticks(fontsize=size_letters)
# plt.ylim(0.5, 5)


# plt.savefig(f"{images_dir}/surp-pos.svg", transparent = True)

# plt.errorbar(groupped2.index + 0.1, groupped2['surprisal_x']['mean'], yerr=groupped2['error_surprisal_x'],linewidth=3, label='Contextualizada')
# plt.errorbar(groupped2.index, groupped2['surprisal_y']['mean'], yerr=groupped2['error_surprisal_y'],linewidth=3, label='Oraciones Aisladas')


# #plt.title(label="Stories")
# plt.xlabel(xlabel='Word Length', fontsize=size_letters)
# plt.ylabel(ylabel='Surprisal', fontsize=size_letters)
# plt.xticks(fontsize=size_letters)
# plt.yticks(fontsize=size_letters)
# plt.ylim(0.5, 5)

# plt.savefig(f"{images_dir}/surp-len.svg", transparent = True)

# plt.errorbar(groupped.index + 0.1, groupped['entropy_x']['mean'], yerr=groupped['error_entropy_x'],linewidth=3, label='Contextualizada')
# plt.errorbar(groupped.index, groupped['entropy_y']['mean'], yerr=groupped['error_entropy_y'],linewidth=3, label='Oraciones Aisladas')


# #plt.title(label="Stories")
# plt.xlabel(xlabel='Word Position', fontsize=size_letters)
# plt.ylabel(ylabel='Entropy', fontsize=size_letters)
# plt.xticks(fontsize=size_letters)
# plt.yticks(fontsize=size_letters)
# plt.ylim(0.5, 3.5)


# plt.savefig(f"{images_dir}/entropy-pos.svg", transparent = True)

# plt.errorbar(groupped2.index + 0.1, groupped2['entropy_x']['mean'], yerr=groupped2['error_entropy_x'],linewidth=3, label='Contextualizada')
# plt.errorbar(groupped2.index, groupped2['entropy_y']['mean'], yerr=groupped2['error_entropy_y'],linewidth=3, label='Oraciones Aisladas')


# #plt.title(label="Stories")
# plt.xlabel(xlabel='Word Length', fontsize=size_letters)
# plt.ylabel(ylabel='Entropy', fontsize=size_letters)
# plt.xticks(fontsize=size_letters)
# plt.yticks(fontsize=size_letters)
# plt.ylim(0.5, 3.5)

# plt.savefig(f"{images_dir}/entropy-len.svg", transparent = True)

# plt.errorbar(groupped3.index + 0.1, groupped3['entropy_x']['mean'], yerr=groupped3['error_entropy_x'],linewidth=1, label='Contextualizada')
# # plt.errorbar(groupped3.index, groupped3['surprisal_y']['mean'], yerr=groupped3['error_surprisal_y'],linewidth=3, label='Oraciones Aisladas')


# #plt.title(label="Stories")
# plt.xlabel(xlabel='Repetition Number', fontsize=size_letters)
# plt.ylabel(ylabel='Entropy', fontsize=size_letters)
# plt.xticks(fontsize=size_letters)
# plt.yticks(fontsize=size_letters)





# x=groupped['pred_x']['mean']
# y=groupped['pred_y']['mean']
# from scipy.stats import ks_2samp
# ks_2samp(x, y)

# #Dos distribuciones y la posicion de la palabra
# #Despues hago comparaciones

# print('correlacion entropia/pred oraciones', pred['entropy'].corr(pred['pred'], method='spearman'))  # Spearman's rho 
# print('correlacion entropia/surp oraciones', pred['entropy'].corr(pred['surprisal'], method='spearman'))

# print('correlacion entropia/pred stories', log_full['entropy'].corr(log_full['pred'], method='spearman'))  # Spearman's rho 
# print('correlacion entropia/surp stories', log_full['entropy'].corr(log_full['surprisal'], method='spearman'))




