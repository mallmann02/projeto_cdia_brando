import mysql.connector
import pandas as pd

class DatabaseReader():
    def __init__(self,
                folder_to_save:str="",
                host:str="www.medbase.com.br",
                user:str="mwcco814_dani_c01",
                password:str="turma_puc@240101",
                database:str="mwcco814_brando_pucrs_dc01"
                ):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.folder_to_save = folder_to_save
        self.__cnx = None
        self.__list_of_tables = ["cond_add", "cond_familia", "cons_reg",
                                 "diag_status", "gravidez", "lab_aac",
                                 "lab_basic", "lab_hemat", "lab_quimica",
                                 "lab_sorol", "liquor", "mri",
                                 "pacientes_reg", "pot_evoc", "surtos_reg",
                                 "trat_dmd", "trat_odr", "trat_out",
                                 "vacinas_reg"]
        self.create_connection()

    def create_connection(self):
        # Criar uma conexão
        self.__cnx = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )

    def read(self, tables:list[str]=None):
        if tables is None:
            tables = self.__list_of_tables

        # Criar um cursor
        cursor = self.__cnx.cursor()

        # Executar a consulta para obter os dados da tabela cond_add
        for table in tables:
            query = f"SELECT * FROM {table}"
            cursor.execute(query)

            # Obter os resultados
            data = cursor.fetchall()

            # Obter os nomes das colunas
            cursor.execute(f"SHOW COLUMNS FROM {table}")
            columns = [column[0] for column in cursor.fetchall()]
            pd.DataFrame(data, columns=columns).to_csv(f"{self.folder_to_save}/{table}.csv", index=False)
        
        cursor.close()
        self.__cnx.close()
