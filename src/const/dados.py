"""
Constantes relacionadas aos dados de pacientes
"""

from os.path import join as pjoin

# set str, assume wildcard: 'string*'; se lista, faz match exato
DADOS_ROOT=pjoin('pac_data','')
DADOS_TMP_ZIPS=pjoin(DADOS_ROOT,'zips','')
DADOS_COLUNA_TEXTO="coment_"