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

surtos_reg_alt_vital_attributes = AttributesInterface(
    context_definition="""
            You are a medical expert, identifying from the context if some body function was affected or not. The context you'll be shown is a relatory of a patient's visit to the doctor being tracked of it's multiple sclerosis. Your task is to identify if any body function was affected in the outbreak.
        """,
    user_prompt="""    
            <<CONTEXT>>
            {context_data}

            <<INSTRUCT>>
            You should output with "Yes" for the affected functions and "Not found" for all the non-affected functions following the template below:

            {json_template}

            If there are no outbreaks mentioned you should output:
            ```json
            {{
                "visual": "Not found",
                "motor": "Not found",
                "sensory": "Not found",
                "cerebellar": "Not found",
                "pyramidal": "Not found",
                "bladder": "Not found",
                "bowel": "Not found",
                "cognitive": "Not found"
            }}
            ```

            REMEMBER: The names of the outbreaks should be in the same format as they appear in the text.
            REMEMBER: If there are multiple outbreaks mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "visual": <yes/not found>,
                "motor": <yes/not found>,
                "sensory": <yes/not found>,
                "cerebellar": <yes/not found>,
                "pyramidal": <yes/not found>,
                "bladder": <yes/not found>,
                "bowel": <yes/not found>,
                "cognitive": <yes/not found>
            }}
            ```
        """,
    pre_intruct_prompts=[
        """
        Examples of body functions alterations that you may find in the text are:
            - Visual
            - Motor
            - Sensory
            - Cerebellar
            - Pyramidal
            - Bladder
            - Bowel
            - Cognitive

        The following are the descriptions of the body functions alterations:
        - Hipoestesia (hypoesthesia): reduced sense of touch or sensation (sensory)
        - Parestesia (paresthesia): abnormal sensation, typically tingling or pricking (sensory)
        - Ataxia (ataxia): lack of muscle coordination (cerebellar)
        - Disartria (dysarthria): difficulty speaking (cerebellar)
        - Diplopia (diplopia): double vision (visual)
        - Nistagmo (nystagmus): involuntary eye movement (visual)
        - Ptose palpebral (ptosis): drooping eyelid (visual)
        - Espasticidade (spasticity): muscle stiffness (motor)
        - Fraqueza (weakness): lack of muscle strength (motor)
        - Disfunção vesical (bladder dysfunction): bladder problems (bladder)
        - Disfunção intestinal (intestinal dysfunction): bowel problems (bowel)
        - Incontinência urinária (urinary incontinence): loss of bladder control (bladder)
        - pyramidal: related to the pyramidal tract, which is responsible for the voluntary control of the muscles
        """
    ]
)