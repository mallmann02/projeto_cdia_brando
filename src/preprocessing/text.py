"""
Coleção de métodos estáticos para pré-processamento de texto.
"""

# externos
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.downloader import download
import re
import os
import json
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
        re.compile(r"\b("+re.escape(char)+r")+\b",re.IGNORECASE) for char in conjunto_chars
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

    # substitui acronimos - BENTO: desativado por enquanto
    # tratado = substituir_acronimos(tratado, obter_acronimos())

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

def obter_acronimos(path:str='acronimos.json') -> dict:
    if not os.path.isfile(path):
        raise ValueError(f"esse arquivo '{path}' não existe.")
    f = open(path, 'r', encoding='utf-8')
    return json.load(f)

def substituir_acronimos(texto:str,
                         acronimos:dict)->str:
    """
    Método que substitui abreviações encontradas em um dicionário passado
    por parâmetro no texto.
    """
    pattern = re.compile(r'\b(' + '|'.join(acronimos.keys()) + r')\b')

    return pattern.sub(lambda x: acronimos[x.group()], texto)