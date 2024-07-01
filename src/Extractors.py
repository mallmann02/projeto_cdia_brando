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
        super().__init__(pacient_report_file, TableAttributes)

class DmdExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "trat_dmd"
        self.chat_model = chat_model
        super().__init__(pacient_report_file, TableAttributes)
        
class OtherTreatExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "trat_out"
        self.chat_model = chat_model
        super().__init__(pacient_report_file, TableAttributes)

class LabBasicExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "lab_basic"
        self.chat_model = chat_model
        super().__init__(pacient_report_file, TableAttributes)

class LabHematExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "lab_hemat"
        self.chat_model = chat_model
        super().__init__(pacient_report_file, TableAttributes)

class LabChemicalExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "lab_quimica"
        self.chat_model = chat_model
        super().__init__(pacient_report_file, TableAttributes)

class LabAacExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "lab_aac"
        self.chat_model = chat_model
        super().__init__(pacient_report_file, TableAttributes)
        
class LabSorolExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "lab_sorol"
        self.chat_model = chat_model
        super().__init__(pacient_report_file, TableAttributes)

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