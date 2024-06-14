#!/usr/bin/env python3
"""Exemplo de uso Dados"""

from src.const.dados import DADOS_ROOT
from src.utils.dados import Dados
import src.preprocessing.text as ppt

# carrega todos os datasets em um unico objeto
dados_todos = Dados('todas_amostras',DADOS_ROOT)
for i,dado in enumerate(dados_todos):
    # elemento é uma tupla de cada exemplar de cada dataset:
    # ('nome dataset','texto a ser processado deste exemplar','dict com todos dados do exemplar neste dataset')
    print(dado)
    if i==3: break

# carrega um conjunto de datasets de um zip
dados_amostra_01 = Dados('amostra-001',DADOS_ROOT+'tratados-amostra-001.zip')
for i,dado in enumerate(dados_amostra_01):
    print(dado)
    if i==3: break

# filtra de um objeto de dados existente
dados_mri_01 = filter(lambda e: e[0]=='t_mri', dados_amostra_01)
for i,dado in enumerate(dados_mri_01):
    print(dado)
    if i==3: break

# limpa txt do mri
for texto_limpo in map(lambda e: ppt.higienizador(e[1]),dados_mri_01):
    print(texto_limpo)
    