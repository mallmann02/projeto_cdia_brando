"""
Coleção de métodos estáticos para pré-processamento de texto.
"""

# externos
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.downloader import download
import re
# locais
import src.const.text as const

# ao chamar este modulo, verifica e baixa dependencias adicionais do nltk (corpora,modelos,etc.)
download('popular',quiet=True)

def remover_caracteres(texto:str,
                       conjunto_chars:set[str]=const.TEXT_REMOVER_CHARS)->str:
    """
    Retorna novo texto sem os caracteres presentes no conjunto
    """
    # word-bounded para não pegar ocorrências no meio de palavras (e.g., fernando vs. nan nan)
    padroes = {
        compile(r"\b("+
                re.escape(char)+
                r")+\b",
                re.IGNORECASE,re.DOTALL) for char in conjunto_chars
    }
    tratado=texto
    for padrao in padroes:
        tratado=re.sub(padrao,'',tratado)
    return tratado

def higienizador(texto:str)->str:
    """
    Realiza multiplos tratamentos do texto para tratar de espaços desnecessários ou quebrar sentença
    """
    # strip e remove caracteres e palavras desnecessárias
    tratado=remover_caracteres(texto.strip())
    # remove espaços antes de quebra de linha
    tratado=re.sub(re.compile(r"[ ]+(?=\n|$)"),'',tratado)
    # sub 2 ou mais espaços por quebra sentenca ('. ')
    return re.sub(re.compile(r"(?<=[a-z0-9])[ ]{2,}(?=\w)"),'. ',tratado)

def tokenizador_topico(texto:str,
                       idioma:str='portuguese',
                       higienizar:bool=True)->list[list[str]]:
    """
    Converte um texto longo, composto por múltiplas sentenças contendo quebras de
    linha ou pontuação de fim, para uma lista de 'topicos' composto por lista de sentenças,
    de acordo com logica de higienizacao e quebra pelo sent_tokenize do nltk.
    """
    tratado=texto
    if higienizar:
        tratado=higienizador(tratado)
    # duas ou mais quebras de linha limita escopo de um 'topico'
    topicos=[
        sent for sent in [sent_tokenize(topico, idioma) for topico in re.split(r"\n{2,}",tratado)]
    ]
    return topicos

def pos_tagging(sentenca:str,idioma:str='portuguese')->list[tuple[str,str]]:
    """
    Retorna uma lista de (palavra,POS) para sentença passada.
    """
    return pos_tag(word_tokenize(sentenca,language=idioma),lang=idioma)
