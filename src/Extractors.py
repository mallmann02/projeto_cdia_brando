from langchain_core.output_parsers import JsonOutputParser
from functools import reduce

dict_values_mapper = {
    "yes": 1.0,
    "no": 0,
    "not found": 99.0
}

class TableDomainExtractor:
    def __init__(self, pacient_report_file, TableAttributes, key_mappings=None, value_mappings=None):
        self.pacient_report_file = pacient_report_file
        self.attributes = TableAttributes
        self.pacient_report = pacient_report_file
        self.key_mappings = key_mappings
        self.value_mappings = value_mappings

    def extract(self):
        total_extracted_data = []
        for attribute in self.attributes:
            messages = attribute.get_chat_model_messages(self.pacient_report)
            try:
                response = self.chat_model.invoke(messages, temperature=0.01)
                extracted_data = self.parse_to_json(response)
                if self.key_mappings:
                    for key, value in self.key_mappings.items():
                        if key in extracted_data:
                            extracted_data[value] = extracted_data.pop(key)
                for key, value in extracted_data.items():
                    if value.lower() in self.value_mappings:
                        extracted_data[key] = self.value_mappings[value]
            except Exception as e:
                print(e)
                continue

            total_extracted_data.append(extracted_data)

        return total_extracted_data

    def parse_to_json(self, extracted_data):
        return JsonOutputParser().parse(extracted_data.content.split("</s>")[-1])
    
    # create a code to compare the extracted data with the expected data
    def compare(self, extracted_data, expected_data):
        tp, fp, fn = 0, 0, 0
        for key in expected_data.keys():
            if key not in extracted_data:
                fn += 1
                continue
            if extracted_data[key] != expected_data[key]:
                fp += 1
            else:
                tp += 1
            del extracted_data[key]
        fp += len(extracted_data)
        return self.calculateMetrics(tp, fp, fn)
    
    def calculateMetrics(self, tp, fp, fn):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
            return precision, recall, f1
        except ZeroDivisionError:
            print("ZeroDivisionError")
        return precision, recall, 0
        
    
class DrugsExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "trat_odr"
        self.chat_model = chat_model
        self.dict_keys_mapper = {
            "other_drugs": "trat_odr",
            "symptom":"sintoma_tratado"}
        self.dict_values_mapper = {
            "fampiridina": 1,
            "espasticidade": 1,
            "canabidiol": 2,
            "fadiga": 2,
            "incotinencia_urinaria": 3,
            "amantadina": 3,
            "modafinil": 4,
            "incontinencia_fecal": 4,
            "baclofeno": 5,
            "disfuncao_eretil": 5,
            "antidepressivo": 6,
            "depressao": 6,
            "aine": 7,
            "ansiedade": 7,
            "vitamina_d": 8,
            "dor_cronica": 8,
            "oxibutinina": 9,
            "alteracao_da_marcha": 9,
            "metilfenidato": 10,
            "sintomas_gripais": 10,
            "hipovitaminose_d": 11,
            "lisdexanfetamina": 11,
            "sindrome_disexecutiva": 12,
            "outro": 98,
            "not found": 99}
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.dict_keys_mapper, value_mappings=self.dict_values_mapper)

class DmdExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "trat_dmd"
        self.chat_model = chat_model
        self.dict_keys_mapper = {
            "dmd":'trat_dmd',
            "natalizumabe":'esq_natalizumabe',
            "interrup_dmd":'mot_interrup_dmd'}
        self.dict_values_mapper = {
            "no":0,
            "avonex":1,
            "rebif_22":2,
            "rebif_44":3,
            "betaferon":4,
            "copaxone_20":5,
            "copaxone_40":6,
            "tysabri":7,
            "gilenya":8,
            "aubagio":9,
            "lemtrada":10,
            "tecfidera":11,
            "ocrevus":12,
            "mavenclad":13,
            "mitoxantrona":14,
            "ciclofosfamida":15,
            "azatioprina":16,
            "mabthera":17,
            "kiendra":18,
            "transplante_medula_ossea":19,
            "plegridy":20,
            "not found":99}
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.dict_keys_mapper, value_mappings=self.dict_values_mapper)
         


        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.dict_keys_mapper)
        
class OtherTreatExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "trat_out"
        self.chat_model = chat_model
        self.dict_keys_mapper = {
            "nao_farmacologico": "trat_out"}
        self.dict_values_mapper = {
            "fisioterapia": 1,
            "terapia_ocupacional": 2,
            "acupuntura": 3,
            "reabilitacao_neurologica": 4,
            "natacao": 5,
            "caminha/correr": 6,
            "pilates": 7,
            "academia": 8,
            "bicileta": 9,
            "yoga": 10,
            "outro":98,
            "not found": 99}
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.dict_keys_mapper, value_mappings=self.dict_values_mapper)

class LabBasicExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "lab_basic"
        self.chat_model = chat_model
        self.dict_keys_mapper = {
            "leucocitos": "leucocitos",
            "neutrofilos": "neutrofilos",
            "neutrofilos": "neutrofilos",
            "linfocitos": "linfocitos",
            "plaquetas": "plaquetas",
            "hemoglobina": "hemoglobina",
            "ast_tgo": "ast_tgo",
            "alt_tgp": "alt_tgp",
            "tsh": "tsh",
            "jc_virus": "anti_jc_virus",
            "jc_virus_index": "jc_virus_index",
            "anti_hiv": "anti_hiv",
            "vdrl": "vdrl",
            "vit_b12": "vit_b12",
            "vit_d": "vit_d",}
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.dict_keys_mapper)

class LabHematExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "lab_hemat"
        self.chat_model = chat_model
        self.dict_keys_mapper = {
            "monocitos": "monocitos",
            "eosinofilos": "eosinofilos",
            "basofilos": "basofilos",
            "celulas_t": "celulas_t",
            "cd4": "cd4",
            "cd8": "cd8",
            "cd19": "cd19",
            "nk_cel": "nk_cel"}

        super().__init__(pacient_report_file, TableAttributes)

class LabChemicalExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "lab_quimica"
        self.chat_model = chat_model
        self.dict_keys_mapper = {
            "proteinas_totais": "proteinas_totais_alt",
            "albumina": "albumina_alt",
            "calcio": "calcio_total_alt",
            "ureia": "ureia_alt",
            "ac_urico": "ac_urico_alt",
            "creatinina": "creatinina_alt",
            "amilise": "amilase_alt",
            "lipase": "lipase_alt",
            "gama_gt": "gama_gt_alt",
            "bilirrubina_direta": "bilirrubina_direta_alt",
            "bilirrubina_indireta": "bilirrubina_indireta_alt",
            "bilirrubina_total": "bilirrubina_total_alt",
            "fosf_alcalina": "fosf_alcalina_alt",
            "t3_total": "t3_total_alt",
            "t4_livre": "t4_livre_alt",
            "col_tot": "col_tot_alt",
            "col_hdl": "col_hdl_alt",
            "col_ldl": "col_ldl_alt",
            "triglicerideos": "triglicerideos_alt",
            "glicemia_jej": "glicemia_jejum_alt"}
        
        self.dict_values_mapper = {
            "normal": 0,
            "alterado": 1,
            "not found": 99
        }

        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.dict_keys_mapper, value_mappings=self.dict_values_mapper)

class LabAacExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "lab_aac"
        self.chat_model = chat_model
        self.dict_keys_mapper = {
            "aquaporina-4":'aqp4',
            "mog":"anti_mog",
            "mitocondria":"anti_mitocondrial",
            "cel_parietal":"anti_cel_parietal",
            "musculo_liso":"anti_musculo_liso",
            "anti_ro":"ssa",
            "anti_la":"ssb",
            "anti_rnp":"rnp",
            "anti_scl_70":"scl_70",
            "anti_jo1":"jo1",
            "dna":"anti_dna",
            "p_anca":"p_anca",
            "c_anca":"c_anca",
            "cardiolipina_igg":"anti_cardiolipina_igg",
            "cardiolipina_igm":"anti_cardiolipina_igm",
            "cardiolipina_iga":"anti_cardiolipina_iga",
            "coagulante_lupico":"anti_coagulante_lupico",
            "transglutaminase":"anti_transglutaminase",
            "fan":"fan",
            "fator_reumatoide":"fat_reumatoide",
            'microssomal':'anticorpo_microsomal',
            'tireoglobulina':'anticorpo_tireoglobulina',
            'anti_beta_2_gpi':'anti_beta2_glicoprot1'}
        self.dict_values_mapper = {
            'no':0,
            'yes':1,
            'Not Found':99}
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.dict_keys_mapper, value_mappings=self.dict_values_mapper)
        
class LabSorolExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "lab_sorol"
        self.chat_model = chat_model
        self.dict_keys_mapper = {
            "urina": "dna_jcv_urina",
            "HBV": "hbsag",
            "hcv": "anti_hcv",
            "htlv": "anti_htlv",
            "varicela": "anti_varicela",
            "IGRA": "quantiferon_test",
            "Mantoux": "mantoux",
        }
        self.dict_values_mapper = {
            "yes": 1,
            "no": 0,
            "not found": 99,
            "weak": 1,
            "strong": 2,
            "positive":2,
            "indeterminate": 1
        }
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.dict_keys_mapper, value_mappings=self.dict_values_mapper)

class SurtosRegExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "surtos_reg"
        self.chat_model = chat_model
        self.dict_keys_mapper = {
            "visual": "sa_func_visuais",
            "motor": "sa_func_tronco",
            "sensory": "sa_func_sensibilidade",
            "cerebellar": "sa_func_cerebelares",
            "pyramidal": "sa_func_piramidais",
            "bladder": "sa_func_intestino_bexiga",
            "bowel": "sa_func_intestino_bexiga",
            "cognitive": "sa_func_cerebrais_mentais",
        }
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.dict_keys_mapper)

class CondAddExtractor(TableDomainExtractor):
 #CONDIÇÕES MÉDICAS ADICIONAIS
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "cond_add"
        self.chat_model = chat_model
        self.key_mappings = {
            "condicao_adicional": "cond_add"
        }
        self.value_mappings = {
            "diabetes": 1,
            "hipertensao": 2,
            "dislipidemia": 3,
            "disfuncao": 4,
            "lupus": 5,
            "artrite_reumatoide": 6,
            "asma": 7,
            "neoplasia_mama": 8,
            "neoplasia_pulmao": 9,
            "neoplasia_hematologica": 10,
            "neoplasia_pele": 11,
            "neoplasia_outra": 12,
            "pupura": 13,
            "glomerulopatia": 14,
            "bloqueio_atrioventricular": 15,
            "aids": 16,
            "infeccao_vias_aereas": 17,
            "pneumonia": 18,
            "herpes": 19,
            "varicela_zoster": 20,
            "outro": 98,
            "not found": 99
        }
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.key_mappings, value_mappings=self.value_mappings)

class CondFamiliaExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "cond_familia"
        self.chat_model = chat_model
        self.key_mappings = {
            "kinship": "ql_familiar_cond",
            "condicao": "cond_familia"
        }
        self.value_mappings = {
            "brother": 1,
            "mom": 2,
            "dad": 3,
            "uncle": 4,
            "cousin": 5,
            "grandparents": 6,
            "esclerose_multipla": 1,
            "nmsod": 2,
            "diabetes": 3,
            "hipertensao": 4,
            "dislipidemia": 5,
            "disfuncao": 6,
            "lupus": 7,
            "artrite": 8,
            "asma": 9,
            "neoplasia_mama": 10,
            "neoplasia_pulmao": 11,
            "neoplasia_hematologica": 12,
            "neoplasia_pele": 13,
            "neoplasia_outra": 14,
            "outro": 98,
            "not found": 99}
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.key_mappings, value_mappings=self.value_mappings)

class ConsRegExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "cons_reg"
        self.chat_model = chat_model
        self.key_mappings = {
            "efeito_adverso": "efeito_adverso_med",
            "surto": "relata_surto",
            "progressao": "progressao_clinica",
            "piora_rad": "piora_radiologica",
            "melhora_rad": "melhora_radiologica",
        }
        self.value_mappings = {
            "yes": 1,
            "no": 0,
            "not found": 99
        }
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.key_mappings)

class DiagStatusExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "diag_status"
        self.chat_model = chat_model
        self.key_mappings ={
            'death': 'obito',
            'cause_death': 'causa_princ_obito',
        }
        self.value_mappings = {
            'yes': 1,
            'no': 0,
            'complicacao':1,
            'neoplasia_mama': 2,
            'neoplasia_pulmao': 3,
            'neoplasia_hematologica': 4,
            'neoplasia_pele': 5,
            'neoplasia_outra': 6,
            'infeccao_oportunista': 7,
            'avc': 8,
            'acidente': 9,
            'suicidio': 10,
            'outro': 98,
            'not found': 99
        }
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.key_mappings, value_mappings=self.value_mappings)

class GravidezExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "gravidez"
        self.chat_model = chat_model
        self.key_mappings ={
            "amamentacao": "amamentou",
            "surto": "surto_na_gravidez",
            "tratou":"trat_surto_gravidez",
            "momento_surto": "mom_surto_gravidez",
            "dmd": "usou_dmd_gravidez"}

        self.value_mappings = {
            "yes": 1,
            "no": 0,
            "not found": 99,
            "pri_trimestre": 1,
            "seg_trimestre": 2,
            "ter_trimestre": 3,
            "sem_dmd": 1,
            "c_dmd": 2,
            "other": 98
        }
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.key_mappings, value_mappings=self.value_mappings)
class LiquorExtractor(TableDomainExtractor):
    def __init__ (self,pacient_report_file, chat_model, TableAttributes):
        self.table_name = "liquor"
        self.chat_model = chat_model
        self.key_mappings = {
            "albumina_lcr": "albumina_lcr",
            "albumina_soro": "albumina_soro",
            "igg_lcr": "igg_lcr",
            "igg_soro": "igg_soro",
            "igg_index_lcr": "igg_index_lcr",
            "leucocitos": "leucocitos_lcr",
            "proteinas": "prot_totais_lcr",
            "bandas_oligoclonais": "boc_lcr"
        }
        self.value_mappings = {
            "no": 0,
            "so_lcr":1,
            "lcr_plasma": 2,
            "not found": 99
        }
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.key_mappings, value_mappings=self.value_mappings)
class PotEvocExtractor(TableDomainExtractor):
    def __init__ (self,pacient_report_file, chat_model, TableAttributes):
        self.table_name = "pot_evoc"
        self.chat_model = chat_model
        self.key_mappings = {
            "visual": "pot_evoc_visual",
            "somato": "pot_evoc_somato",
            "auditivo": "pot_evoc_auditivo",}
        self.value_mappings = {
            'no': 0,
            'yes': 1,
            'not found': 99}
        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.key_mappings, value_mappings=self.value_mappings)

class VacinasExtractor(TableDomainExtractor):
    def __init__ (self,pacient_report_file, chat_model, TableAttributes):
        self.table_name = "vacinas"
        self.chat_model = chat_model
        self.key_mappings = {
            "tipo_vacina": "tipo_vacina",
        }
        self.value_mappings = {
            "dupla_adulto": 1,
            "febre_amarela": 2,
            "hepatite_a": 3,
            "hepatite_b": 4,
            "herpes_zoster": 5,
            "hpv_bivalente": 6,
            "hpv_quadrivalente": 7,
            "influenza": 8,
            "meningococica_b": 9,
            "meningococica_c": 10,
            "meningococica_quadrivalente": 11,
            "pneumococica_13_valente": 12,
            "pneumococica_23_valente": 13,
            "triplice_viral": 14,
            "varicela": 15,
            "covid_19": 16,
            "covid_19_coronavac": 17,
            "covid_19_oxford_astrazeneca": 18,
            "covid_19_sputnik_v": 19,
            "covid_19_pfizer": 20,
            "covid_19_covaxin": 21,
            "covid_19_janssen": 22,
            "covid_19_butanvac": 23,
            "outro": 98,
            "not found": 99
        }

        super().__init__(pacient_report_file, TableAttributes, key_mappings=self.key_mappings, value_mappings=self.value_mappings)