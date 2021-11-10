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
