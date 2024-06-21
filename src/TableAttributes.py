from langchain.prompts import ChatPromptTemplate

class AttributesInterface:
    def __init__(self, context_definition, user_prompt, json_extracion_template, pre_intruct_prompts=None):
        self.context_definition = context_definition
        self.pre_intruct_prompts = pre_intruct_prompts
        self.user_prompt = user_prompt
        self.json_template = json_extracion_template

    def get_chat_model_messages(self, context_data):
        pre_instruct_prompts = ([("ai", pre_instruct_prompt) for pre_instruct_prompt in self.pre_intruct_prompts]) if self.pre_intruct_prompts else ()
        template = ChatPromptTemplate.from_messages([
            ("system", self.context_definition),
            *pre_instruct_prompts,
            ("human", self.user_prompt),
        ])

        messages = template.format_messages(
            context_data=context_data,
            json_template=self.json_template
        )

        return messages
    

odr_drugs_attributes = AttributesInterface(
    context_definition="""
    You are a medical expert, helping to extract drugs names from text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the drugs mentioned in the text. If you don't find any drugs, you should say 'No drugs mentioned'.
    """,
    user_prompt="""
        <<CONTEXT>>
        {context_data}

        <<QUESTION>>
        What are the names of the drugs mentioned in the text?

        <<EXAMPLE>>
        If the text mentions the drugs "aspirin" and "ibuprofen" you should output:

        {json_template}

        If there are no drugs mentioned you should output:
        ```json
        {{
            "drug": "No drugs mentioned"
        }}
        ```

        REMEMBER: The names of the drugs should be in the same format as they appear in the text.
        REMEMBER: If there are multiple drugs mentioned you should list them all.
        REMEMBER: You don't need to include any other information in the output.
    """,
    json_extracion_template="""
        ```json
        {{
            "drugs": [
                {{
                    "drug": "<extracted_drug>",
                    "prescription": "<extracted_interval>"
                }},
                ...
            ]
        }}
        ```
    """
)

lab_basic_attributes = AttributesInterface(
    context_definition="""
            You are a medical expert, helping to extract basic lab tests from text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the lab tests and their results mentioned in the text. If you don't find any lab tests, you should say 'No lab tests mentioned'.
        """,
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the lab tests that you may find, what are the names of the lab tests and their results?

            <<EXAMPLE>>
            If the text mentions the lab tests "blood pressure" with the result "120/80" and "cholesterol" with the result "200 mg/dL" you should output:

            {json_template}

            If there are no lab tests mentioned you should output:
            ```json
            {{
                "lab_basic": "No lab tests mentioned"
            }}
            ```

            REMEMBER: The names of the lab tests and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple lab tests mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "lab_basic": [
                    {{
                        "test": "<extracted_test>",
                        "result": "<extracted_result>"
                    }},
                    ...
                ]
            }}
            ```
        """,
    pre_intruct_prompts=[
        """
        Examples of lab tests (some of them with their acronyms) that you may find in the text are:
            - leukocytes (leuco,WBC)
            - neutrophils
            - lymphocytes (linf)
            - platelets (plaq,plaquetas)
            - hemoglobin (Hb)
            - TGO (AST)
            - TGP (ALT)
            - TSH
            - Vitamin D (vit D)
            - Vitamin B12 (vit B12)
        """
    ]
)

lab_hemat_attributes = AttributesInterface(
    context_definition="""
            You are a medical expert and help extract laboratory hematocrit tests from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the lab tests and their results mentioned in the text. If you don't find any lab tests, you should say 'No lab tests mentioned'.
        """,
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the lab tests that you may find, what are the names of the lab tests and their results?

            <<EXAMPLE>>
            If the text mentions the lab tests "HT" with the result "33,2" and "TGP" with the result "108" you should output:

            {json_template}

            If there are no lab tests mentioned you should output:
            ```json
            {{
                "lab_hemat": "No lab tests mentioned"
            }}
            ```

            REMEMBER: The names of the lab tests and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple lab tests mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "lab_hemat": [
                    {{
                        "test": "<extracted_test>",
                        "result": "<extracted_result>"
                    }},
                    ...
                ]
            }}
            ```
        """,
    pre_intruct_prompts=[
        """
        Examples of lab tests (some of them with their acronyms) that you may find in the text are:
            - liver function tests(TGO, TGP, FA, bilirrubinas,PFH, ALT)
            - gama-gt (GGT,gamagt)
            - hematocrit (HT, Hct)
        """
    ]
)

lab_chemical_attributes = AttributesInterface(
    context_definition="""
            You are a medical expert and help extract chemical laboratory tests from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the lab tests and their results mentioned in the text. If you don't find any lab tests, you should say 'No lab tests mentioned'.
        """,
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the lab tests that you may find, what are the names of the lab tests and their results?

            <<EXAMPLE>>
            If the text mentions the lab tests "LDL" with the result "142,56" and "HDL" with the result "32" you should output:

            {json_template}

            If there are no lab tests mentioned you should output:
            ```json
            {{
                "lab_chemical": "No lab tests mentioned"
            }}
            ```

            REMEMBER: The names of the lab tests and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple lab tests mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "lab_chemical": [
                    {{
                        "test": "<extracted_test>",
                        "result": "<extracted_result>"
                    }},
                    ...
                ]
            }}
            ```
        """,
    pre_intruct_prompts=[
        """
        Examples of lab tests (some of them with their acronyms) that you may find in the text are:
            - low density lipoprotein (LDL)
            - high density lipoprotein (HDL)
            - triglycerides (Trigl, TL)
        """
    ]
)

lab_aac_attributes = AttributesInterface(
    context_definition="""
            You are a medical expert and help extract antibodies laboratory tests from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the lab tests and their results mentioned in the text. If you don't find any lab tests, you should say 'No lab tests mentioned'.
        """,
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the lab tests that you may find, what are the names of the lab tests and their results?

            <<EXAMPLE>>
            If the text mentions the lab tests "ANTI TPO" with the result "Inferior a 5,0" and "AC ANTI-AQP4" with the result "NEG" you should output:

            {json_template}

            If there are no lab tests mentioned you should output:
            ```json
            {{
                "lab_aac": "No lab tests mentioned"
            }}
            ```

            REMEMBER: The names of the lab tests and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple lab tests mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "lab_aac": [
                    {{
                        "test": "<extracted_test>",
                        "result": "<extracted_result>"
                    }},
                    ...
                ]
            }}
            ```
        """,
    pre_intruct_prompts=[
        """
        Examples of lab tests (some of them with their acronyms) that you may find in the text are:
            - anti-MOG antibody-associated disease (MOGAD, MONEM)
            - anti-aquaporin-4 antibody (Anti-AQP4 Ab, anti-NMO-IgG Ab)
            - antibodies (ANTI)
            - Anti-LKM
            - Anti-CCP
            - Anti-ML
            - erythrocyte sedimentation (VSG, VHS, ESR)
        """
    ]
)

lab_sorol_attributes = AttributesInterface(
    context_definition="""
            You are a medical expert and help extract sorol laboratory tests from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the lab tests and their results mentioned in the text. If you don't find any lab tests, you should say 'No lab tests mentioned'.
        """,
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the lab tests that you may find, what are the names of the lab tests and their results?

            <<EXAMPLE>>
            If the text mentions the lab tests "HIV" with the result "NEG" and "VZV" with the result "1.019" you should output:

            {json_template}

            If there are no lab tests mentioned you should output:
            ```json
            {{
                "lab_sorol": "No lab tests mentioned"
            }}
            ```

            REMEMBER: The names of the lab tests and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple lab tests mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "lab_sorol": [
                    {{
                        "test": "<extracted_test>",
                        "result": "<extracted_result>"
                    }},
                    ...
                ]
            }}
            ```
        """,
    pre_intruct_prompts=[
        """
        Examples of lab tests (some of them with their acronyms) that you may find in the text are:
            - HIV
            - HCV
            - MANTOUX (PPD)
            - VZV (vírus varicela-zoster)
            - anti-aquaporin-4 antibody (Anti-AQP4 Ab, anti-NMO-IgG Ab)
            - cytomegalovirus(CMV)
            - syphilis (VDRL+, reag., reagente)
            - Epistein Barr
            - RUBELLA
            - Hepatitis
        """
    ]
)
