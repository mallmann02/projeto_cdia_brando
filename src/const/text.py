"""
Constantes relacionadas ao pré-processamento de texto
"""

from re import compile,IGNORECASE,DOTALL

# conjunto de caracteres para serem removidos (um por linha)
TEXT_REMOVER_CHARS = {
    'nan',
}
