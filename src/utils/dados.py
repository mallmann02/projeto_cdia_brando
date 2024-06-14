# externo
from zipfile import ZipFile
from os import path, listdir, walk, remove, rmdir
from pandas import read_csv, DataFrame
from pandas.errors import ParserError
# local
import src.const as const

class Dados():
    """
    Classe para manipular arquivos com texto a ser processado
    """

    def __init__(self,
                 nome:str,
                 path_arquivo:str|list[str]) -> None:
        self.__nome = nome
        self.dados = path_arquivo

    def __iter__(self):
        """
        Itera por dataset e por paciente do dataset. Ao iterar este objeto, retorna
        a seguinte tupla para cada paciente e dataset:
        (filename, texto livre do paciente, dict dos dados deste paciente no dataset)
        """
        for filename in self.dados.keys():
            coluna_com_texto=None
            # acha coluna com texto neste dataset
            for col in self.dados[filename].columns:
                if str(col).find(const.DADOS_COLUNA_TEXTO)>-1:
                    coluna_com_texto=col
                    break
            if not coluna_com_texto:
                # dataset sem texto, então pula
                continue
            # passa exemplares na forma (filename, texto, dict de todos dados)
            for exemplar in self.dados[filename].itertuples(index=False):
                dict_exemplar=exemplar._asdict()
                yield (filename,dict_exemplar[coluna_com_texto],dict_exemplar)
    
    def __hash__(self)->int:
        return self.nome.__hash__()
    
    def __len__(self):
        return len(self.dados)

    @property
    def nome(self)->str:
        return self.__nome
    @nome.setter
    def nome(self,nome:str):
        self.__nome=nome
    
    @property
    def dados(self)->dict[str,DataFrame]:
        return self.__dados
    @dados.setter
    def dados(self,path_arquivo:str|list[str]):
        # lógica para ler CSVs de zip, lista de zips ou dir com zips
        self.__dados=dict()
        zips=list()
        # parse arquivos por parametro
        if isinstance(path_arquivo,list):
            # assume list[str]
            zips=path_arquivo
        elif path.isdir(path_arquivo):
            # assume dir com zips
            zips=[path.join(path_arquivo,z) for z in listdir(path_arquivo) if z.lower().endswith('.zip')]
        else:
            # assume unico zip
            zips.append(path_arquivo)
        # extrai zips para subdir em constantes
        for zip in zips:
            with ZipFile(zip) as zf:
                zf.extractall(const.DADOS_TMP_ZIPS)
        # processa todos csvs no caminhos dos zips
        for root,dirs,files in walk(const.DADOS_TMP_ZIPS,topdown=False):
            for f in files:
                nome,ext=path.splitext(f)
                if ext.lower()!='.csv': continue
                # tenta criar df e adiciona a coleção
                csv=path.join(root,f)
                default_kwargs=dict(encoding='utf-8')
                try:
                    dado={nome:read_csv(csv,**default_kwargs)}
                    # valida que tem mais do que uma coluna
                    if not len(dado[nome].columns) > 1:
                        # tenta logica para outros parsers deste arquivo
                        raise ParserError
                except ParserError:
                    # se não consegue interpretar com separador ',', tenta com ';'
                    dado={nome:read_csv(csv,sep=';',**default_kwargs)}
                finally:
                    remove(csv)
                self.__dados.update(dado)
            for d in dirs:
                try:
                    rmdir(path.join(root,d))
                except OSError:
                    # dir tem arquivo que nao foi processado, entao nao remove por seguranca
                    continue
        rmdir(const.DADOS_TMP_ZIPS)
