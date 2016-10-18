import Stemmer
stemmer = Stemmer.Stemmer('spanish')
palabras = ['spark', 'es', 'un', 'framework', 'de', 'analisis', 'distribuido', 'en', 'memoria', 'el', 'cual', 'fue', 'desarrollado', 'en', 'la', 'universidad', 'de', 'california']
palabras = ['hablemos', 'hablar','hablamos', 'habla', 'programando', 'programa', 'programar', 'programado'
'colombia', 'colombiano', 'jugar', 'juega', 'jugamos','jueguemos']
print(stemmer.stemWords(palabras))


stemmer = Stemmer.Stemmer('english')
palabras = ['spark', 'es', 'un', 'framework', 'de', 'analisis', 'distribuido', 'en', 'memoria', 'el', 'cual', 'fue', 'desarrollado', 'en', 'la', 'universidad', 'de', 'california']
palabras = ['newsgroup', 'path','cyclying', 'go', 'go', 'programer', 'program', 'programado'
'colombia', 'colombiano', 'jugar', 'juega', 'jugamos','jueguemos']
print(stemmer.stemWords(palabras))
