# Predictability WIP

Este repositorio contiene el codigo para generar los estimulos para el experimento de Cloze-Task online para oraciones aisladas. Tambien contiene los scripts para analizar los resultados el experimento Cloze-Task.

## Texts_Data

#### _texts.csv_
Contiene los cuentos que se utilizaron en el experimento cloze-task con sus respectivos tags. Cada fila es una oracion de un cuento.
#### _grammatical_analysis.csv_
Contiene el analisis de las categorias gramaticales de las oraciones de los cuentos. 

## ClozeTask_Data

Corpus de historias naturales utilizado por Bianchi et al. (2020).

Con este corpus se realizaron 2 cloze-task online:
Se realizó el experimento sobre el total de las palabras [Bianchi et al. (2020)]
Se seleccionaron 15 oraciones de cada una de 7 historias y se realizó el experimento con las mismas, aisladas de su contexto [Peroni et al. (en preparación)]

Contiene los resultados de ambos experimentos mas los scripts para analizar los datos.

## Code
#### _grammatical_analysis.py_
Codigo para hacer el analisis gramatical de las oraciones con spaCy.
#### _procesamiento textos.py_
Son dos scripts que son redunantes y tienen que ser modificados, analisis previo al experimento de los cuentos. 
#### _loadLogs.py_
Analisis de los resultados del CT. A cambiar. 
