### Analisis gramatico de las oraciones que se utilizaron para el experimento de Cloze-Task Oraciones online en 2020 ###

### Importo los paquetes necesarios
from numpy import False_, true_divide
import spacy
from spacy_syllables import SpacySyllables
import pandas as pd 
import re
import matplotlib.pyplot as plt


#Cargo el modelo de nlp
nlp = spacy.load("es_core_news_sm")
syllables = SpacySyllables(nlp)
nlp.add_pipe('syllables')

#Cargo los archivos de los textos
textsFileName = 'Texts_Data/textos.csv'
columnas = ['oracID', 'type', 'null', 'oracStr']
oraciones = pd.read_csv(textsFileName, encoding='iso-8859-1',header=None, names=columnas,quotechar="<")
oraciones['oracStr'] = [re.sub('/p>', '', x) for x in oraciones['oracStr']]
oraciones['oracStr'] = [re.sub('p>',  '', x) for x in oraciones['oracStr']]
oraciones['lenght']  = [len(x.split(' ')) for x in oraciones['oracStr']]

#Tomo solo los cuentos
oraciones = oraciones[oraciones.oracID<105]

#Le paso los archivos al modelo
docs = list(nlp.pipe(oraciones.oracStr))

def extract_tokens_plus_meta(doc:spacy.tokens.doc.Doc):  #Extrae tokens y metadata de un doc 
    """Extract tokens and metadata from individual spaCy doc."""
    return [
        (i.text, i.i, i.lemma_, i.ent_type_, i.tag_, 
         i.dep_, i.pos_, i.is_stop, i.is_alpha, 
         i.is_digit, i.is_punct, i._.syllables, i._.syllables_count) for i in doc
    ]

def tidy_tokens(docs): #Extrae tokens y metadata de una lista de docs de spacy
    """Extract tokens and metadata from list of spaCy docs."""
    
    cols = [
        "doc_id", "token", "token_order", "lemma", 
        "ent_type", "tag", "dep", "pos", "is_stop", 
        "is_alpha", "is_digit", "is_punct", "syllables", "syllables_count"
    ]
    
    meta_df = []
    for ix, doc in enumerate(docs):
        meta = extract_tokens_plus_meta(doc)
        meta = pd.DataFrame(meta)
        meta.columns = cols[1:]
        meta = meta.assign(doc_id = ix).loc[:, cols]
        meta_df.append(meta)
        
    return pd.concat(meta_df)    

grammatical_analysis = tidy_tokens(docs)
grammatical_analysis.reset_index(inplace=True)
grammatical_analysis.drop(grammatical_analysis[grammatical_analysis['is_punct'] == True].index, inplace=True) #Saco los signos de puntuacion

print(grammatical_analysis)

grammatical_analysis.to_csv(path_or_buf='Texts_Data/grammatical_analysis.csv')


# Algunos analisis de las palabras
# tidy_docs.query("is_stop == False & is_punct == False").lemma.value_counts().head(10).plot(kind="barh", figsize=(24, 14), alpha=.7)
# plt.yticks(fontsize=20)
# plt.xticks(fontsize=20)

# print(tidy_docs.query("ent_type != ''").ent_type.value_counts())







