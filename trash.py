"""Guardo las cosas que voy probando y no funcionan, pero que quizas sirvan en otro mommento"""
"""#Trash"""

# def prob_word_in_part(n, N): #La probabilidad de que una palabra aparezca en un cuento dado que aparece en el corpus
#   n = np.array(n)
#   N = np.array(N)
#   f = n/N #Vector con todas las f (frec de aparicion de la palabra en i)
#   den = np.sum(f)
#   return np.array([fi/den for fi in f])

# for key, value in data_dict.items():
#   print(f'key = {key}') # La palabra
#   print(f'value = {value}') # Cuantas veces aparece en cada cuento
#   print(f'N = {N}') #Cantidad de pals en cada cuento
#   p = prob_word_in_part(value, N) 
#   print(f'p = {p}')
#   S = shannon_entropy(p, 7)
#   print(f'S = {S}')

# #Imprime la palabra y las posibles correcciones, no es muy bueno
# spell = SpellChecker(language='es')
# # find those words that may be misspelled
# misspelled = spell.unknown(nuevo['completada'])

# for word in misspelled:
#     # Get the one `most likely` answer
#     print(spell.correction(word))

#     # Get a list of `likely` options
#     print(spell.candidates(word))

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
