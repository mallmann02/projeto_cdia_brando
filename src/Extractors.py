from langchain_core.output_parsers import JsonOutputParser

class TableDomainExtractor:
    def __init__(self, pacient_report_file, TableAttributes):
        self.pacient_report_file = pacient_report_file
        self.attributes = TableAttributes
        self.pacient_report = pacient_report_file

    def extract(self):
        total_extracted_data = []
        for attribute in self.attributes:
            messages = attribute.get_chat_model_messages(self.pacient_report)
            try:
                response = self.chat_model.invoke(messages, temperature=0.01)
                extracted_data = self.parse_to_json(response)
            except Exception as e:
                print(e)
                continue

            total_extracted_data.append(extracted_data)

        return total_extracted_data

    def parse_to_json(self, extracted_data):
        return JsonOutputParser().parse(extracted_data.content.split("</s>")[-1])
    
class DrugsExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "trat_odr"
        self.chat_model = chat_model
        super().__init__(pacient_report_file, TableAttributes)

class LabBasicExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "lab_basic"
        self.chat_model = chat_model
        super().__init__(pacient_report_file, TableAttributes)

class SurtosRegExtractor(TableDomainExtractor):
    def __init__(self, pacient_report_file, chat_model, TableAttributes):
        self.table_name = "surtos_reg"
        self.chat_model = chat_model
        super().__init__(pacient_report_file, TableAttributes)