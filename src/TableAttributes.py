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
            "other_drugs": "Not found",
            "symptom": "Not found",
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
                    "other_drugs": "<extracted_drug>",
                    "symptom":"<extracted_symptom>"
                }},
                ...
            ]
        }}
        ```
    """,
    pre_intruct_prompts=[
        """
        Some examples of drugs that you may find in the text are:
         - Fampiridina
         - Baclofeno
         - Antidepressivo
         - Canabidiol
         - Modadinil
         
         Some examples of symptoms that you may find in the text are:
         - Espasticidade
         - Fadiga
         - Depressao
         - Ansiedade
         """
    ]
)

dmd_drugs_attributes = AttributesInterface(
        context_definition="""
    You are a medical expert, helping to extract disease modifying drugs (dmd) names from text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the dmd mentioned in the text. If you don't find any drugs, you should say 'No dmd mentioned'.
    """,
    user_prompt="""
        <<CONTEXT>>
        {context_data}

        <<QUESTION>>
        What are the names of the dmd mentioned in the text?

        <<EXAMPLE>>
        If the text mentions the dmd "natalizumabe" and "ciclofosfamida" you should output:

        {json_template}

        If there are no dmd mentioned you should output:
        ```json
        {{
            "dmd": "not found"
        }}
        ```

        REMEMBER: The names of the dmd should be in the same format as they appear in the text.
        REMEMBER: If there are multiple dmd mentioned you should list them all.
        REMEMBER: You don't need to include any other information in the output.
    """,
    json_extracion_template="""
        ```json
        {{
            "dmd": [
                {{
                    "dmd": "<extracted_dmd>",
                    "naturazalibe": "<extracted_interval>",
                    "interrup_dmd": "<extracted_interval>"
                }},
                ...
            ]
        }}
        ```
    """,
    pre_intruct_prompts=[
        """
        Some examples of dmd drugs that you may find in the text are:
        - ciclofosfamida
        - fingolimode      
        - betainterferona (type 1a: rebif,avonex, type 1b: betaferon) 
        - glatirâmer
        - interferon beta
        - fumarato de dimetila
        """ 
    ]
)

other_treat_attributes = AttributesInterface(
        context_definition="""
    You are a medical expert, helping to extract others treatments(out) names from text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the out mentioned in the text. If you don't find any drugs, you should say 'No out mentioned'.
    """,
    user_prompt="""
        <<CONTEXT>>
        {context_data}

        <<QUESTION>>
        What are the names of the others treatments mentioned in the text?

        <<EXAMPLE>>
        If the text mentions the dmd "natalizumabe" and "ciclofosfamida" you should output:

        {json_template}

        If there are no out mentioned you should output:
        ```json
        {{
            "nao_farmacologico": "not found"
        }}
        ```

        REMEMBER: The names of the out should be in the same format as they appear in the text.
        REMEMBER: If there are multiple out mentioned you should list them all.
        REMEMBER: You don't need to include any other information in the output.
    """,
    json_extracion_template="""
        ```json
        {{
            "out": [
                {{
                    "nao_farmacologico": "<extracted_trat_out>",
                }},
                ...
            ]
        }}
        ```
    """,
    pre_intruct_prompts=[
        """
        Some examples of other treatments that you may find in the text are:
        - fisioterapia
        - terapia_ocupacional
        - acupuntura
        - reabilitacao_neurologica
        - natacao
        - caminha/correr
        - pilates
        - academia
        - bicileta
        - yoga
        """ 
    ]
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
                "leucocitos": "Not found",
                "neutrofilos": "Not found",
                "linfocitos": "Not found",
                "plaquetas": "Not found",
                "hemoglobina": "Not found",
                "ast_tgo": "Not found",
                "alt_tgp": "Not found",
                "tsh": "Not found",
                "jc_virus": "Not found",
                "jc_virus_index": "Not found",
                "anti_hiv": "Not found",
                "vdrl": "Not found",
                "vit_b12": "Not found",
                "vit_d": "Not found"

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
                        "leucocitos": <extracted_result>,
                        "neutrofilos": <extracted_result>,
                        "linfocitos": <extracted_result>,
                        "hemoglobina": <extracted_result>,
                        "ast_tgo": <extracted_result>,
                        "alt_tgp": <extracted_result>,
                        "tsh": <extracted_result>,
                        "jc_virus": <extracted_result>,
                        "jc_virus_index": <extracted_result>,
                        "anti_hiv": <extracted_result>,
                        "vdrl": <extracted_result>,
                        "vit_b12": <extracted_result>,
                        "vit_d": <extracted_result>

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
            REMEMBER: Don't nest json outputs. Each output should be a single json object.
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

surtos_reg_info_surto_attributes = AttributesInterface(
    context_definition="""
            You are a medical expert, identifying from the context if there is any outbreak mentioned. The context you'll be shown is a relatory of a patient's visit to the doctor being tracked of it's multiple sclerosis. Your task is to answer the questions about the outbreak, following the json template.
        """,
    user_prompt="""    
            <<QUESTIONS>>
            - Was the daily routine affected by the outbreak? (Yes/No/Not Found)
            - What was the severity of the outbreak, in levels? Level 1 to 3 (softer to harder) or Not Found
            - What was the recovery os the pacient? Level 1 to 3 (no recovery to full recovery) or Not Found
            - What was the duration of the outbreak? (extracted duration in days) or Not Found
            - Was the pacient treated? (Yes/No/Not Found)
            - When did the treatment started? (extracted time in days) or Not Found

            
            <<TEMPLATE>>
            {json_template}

            <<EXAMPLE>>
            Sentence: ""
            
            <<CONTEXT>>
            {context_data}
        """,
    json_extracion_template="""
            ```json
            {{
                "affected_daily_routine": "<Yes/No/Not Found>",
                "severity": "<1/2/3/Not Found>",
                "resolution": "<1/2/3/Not Found>",
                "duration": "<extracted duration in days/Not Found>",
                "was_treated": "<Yes/No/Not Found>",
                "treatment_start": "<extracted time in days/Not found>"
            }}
            ```
        """
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
                "lab_hemat": "Not found"
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
                        "monocitos": <extracted_result>,
                        "eosinofilos": <extracted_result>,
                        "basofilos": <extracted_result>,
                        "celulas_t": <extracted_result>,
                        "cd4":  <extracted_result>,
                        "cd8": <extracted_result>,
                        "cd19": <extracted_result>,
                        "nk_cel": <extracted_result>
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
                "aquaporina-4":"Not Found",
                    "mog":"Not Found",
                    "mitocondria":"Not Found",
                    "cel_parietal":"Not Found",
                    "musculo_liso":"Not Found",
                    "anti_ro":"Not Found",
                    "anti_la":"Not Found",
                    "anti_rnp":"Not Found",
                    "anti_scl_70":"Not Found",
                    "anti_jo1":"Not Found",
                    "dna":"Not Found",
                    "p_anca":"Not Found",
                    "c_anca":"Not Found",
                    "cardiolipina_igg":"Not Found",
                    "cardiolipina_igm":"Not Found",
                    "cardiolipina_iga":"Not Found",
                    "coagulante_lupico":"Not Found",
                    "transglutaminase":"Not Found",
                    "fan":"Not Found",
                    "fator_reumatoide":"Not Found",
                    'microssomal':"Not Found",
                    'tireoglobulina':"Not Found",
                    'anti_beta_2_gpi':"Not Found"
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
                    "aquaporina-4":<yes/no>,
                    "mog":<yes/no>,
                    "mitocondria":<yes/no>,
                    "cel_parietal":<yes/no>,
                    "musculo_liso":<yes/no>,
                    "anti_ro":<yes/no>,
                    "anti_la":<yes/no>,
                    "anti_rnp":<yes/no>,
                    "anti_scl_70":<yes/no>,
                    "anti_jo1":<yes/no>,
                    "dna":<yes/no>,
                    "p_anca":<yes/no>,
                    "c_anca":<yes/no>,
                    "cardiolipina_igg":<yes/no>,
                    "cardiolipina_igm":<yes/no>,
                    "cardiolipina_iga":<yes/no>,
                    "coagulante_lupico":<yes/no>,
                    "transglutaminase":<yes/no>,
                    "fan":<yes/no>,
                    "fator_reumatoide":<yes/no>,
                    'microssomal':<yes/no>,
                    'tireoglobulina':<yes/no>,
                    'anti_beta_2_gpi':<yes/no>
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
                        "urina":<yes/no>,
                        "HBV":<yes/no>,
                        "hcv":<yes/no>,
                        "htlv":<yes/no>,
                        "varicela":<yes/no>,
                        "IGRA":<no/indeterminate/positive>,
                        "Mantoux":<weak/strong/no>,
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

cond_add_attributes = AttributesInterface(
    context_definition="""
            You are a medical expert and help extract additional medical conditions from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the conditions mentioned in the text. If you don't find any condition, you should say 'No additionals conditions mentioned'.
        """,
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the additionals conditions that you may find, what are the names of the conditions?

            {json_template}

            If there are no conditions mentioned you should output:
            ```json
            {{
                "cond_add": "Not Found"
            }}
            ```

            REMEMBER: The names of the conditions and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple lab tests mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "cond_add": [
                    {{
                        "cond_add": <diabetes/hipertensao/disfuncao/dislipidemia/asma/aids/pneumonia/neoplasia_de_mama/neoplasia_de_pulmao/
                                    neoplasia_hematologica/neoplasia_pele>,
                    }},
                    ...
                ]
            }}
            ```
        """,
    pre_intruct_prompts=[
        """
        Examples of some conditions (some of them with their acronyms) that you may find in the text are:
            - diabetes
            - hipertensao
            - disfuncao
            - dislipidemia
            - asma
            - aids
            - pneumonia
            - neoplasia_de_mama
            - neoplasia_de_pulmao
            - neoplasia_hematologica
            - neoplasia_pele
            - neoplasia_outra
            - lupus
            - artrite_reumatoide
            - glomerulopatia
            - bloqueio_atrioventricular
            - infeccao_vias_aereas
            - herpes
            - varicela_zoster (varicela/zoster)
        If you find any of these conditions, you should output them as 'outro'.

        """
    ]
)

cond_fam_attributes = AttributesInterface(
    context_definition="""
            You are a medical expert and help extract family medical conditions from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the conditions mentioned in the text. If you don't find any condition, you should say 'No additionals conditions mentioned'.
        """,
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the additionals conditions that you may find, what are the names of the conditions?

            {json_template}

            If there are no conditions mentioned you should output:
            ```json
            {{
                "kinship": "Not Found",
                "condicao": "Not Found"
            }}
            ```

            REMEMBER: The names of the conditions and the kinship their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple conditions mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "cond_familia": [
                    {{
                        "kinship": <brother/mom/dad/uncle/cousin/grandparents/outro>,
                        "condicao": <escleorose_multipla/nmsod/diabetes/hipertensao/disfuncao/dislipidemia/asma/pneumonia/neoplasia_de_mama/neoplasia_de_pulmao/
                                    neoplasia_hematologica/neoplasia_pele/neoplasia_outra/lupus/outro>
                    }},
                    ...
                ]
            }}
            ```
        """,
    pre_intruct_prompts=[
        """
        Examples of some conditions (some of them with their acronyms) that you may find in the text are:
            - escleorose_multipla
            - nmsod
            - diabetes
            - hipertensao
            - disfuncao
            - dislipidemia
            - asma
            - pneumonia
            - neoplasia_de_mama
            - neoplasia_de_pulmao
            - neoplasia_hematologica
            - neoplasia_pele
            - neoplasia_outra
            - lupus
        If you find any of these conditions, you should output them as 'outro'.
        Some of kinships you may find in the text are:
            - brother (sister)
            - mom (mother)
            - dad (father)
            - uncle (aunt)
            - cousin
            - grandparents (grandmother, grandfather)
        """
    ]
)
cons_reg_attributes = AttributesInterface(
    context_definition="""
            You are a medical expert and help extract information about the patient's consultation from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the some informations in the text. If you don't find any information, you should say 'No informations mentioned'.
            """,
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the consultation that you may find, what are the names of the informations?

            {json_template}

            If there are no informations mentioned you should output:
            ```json
            {{
                "efeito_adverso": "Not found"
                "surto": "Not found"
                "progressao": "Not found"
                "piora_rad": "Not found"
                "melhora_rad": "Not found"
            }}
            ```

            REMEMBER: The names of the informations and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple informations mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "cons_reg": [
                    {{
                        "efeito_adverso": <yes/no>,
                        "surto": <yes/no>,
                        "progressao": <yes/no>,
                        "piora_rad": <yes/no>,
                        "melhora_rad": <yes/no>
                    }},
                    ...
                ]
            }}
            ```
        """)

diag_status_attributes = AttributesInterface(
    context_definition="""
            You are a medical expert and help extract information about the patient's diagnosis status from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the some informations in the text. If you don't find any information, you should say 'No informations mentioned'.
            """,
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the diagnosis status that you may find, what are the names of the informations?

            {json_template}

            If there are no informations mentioned you should output:
            ```json
            {{
                "death": "Not found"
                "cause_death": "Not found"
            }}
            ```

            REMEMBER: The names of the informations and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple informations mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "diag_status": [
                    {{
                        "death": <yes/no>,
                        "cause_death": <complicacao/neoplasia_mama/neoplasia_pulmao/neoplasia_hematologica/neoplasia_pele/neoplasia_outra/infeccao_oportunista/avc/
                                        acidente/suicidio/outro>
                    }},
                    ...
                ]
            }}
            ```
        """,
    pre_intruct_prompts=[
        """
        Examples of some causes of death that you may find in the text are:
        - complicacao
        - neoplasia mama
        - neuplasia pulmao
        - neoplasia hematologica
        - neoplasia pele
        - neoplasia outra
        - Stroke/myocardial infarction
        - accident
        - suicide

        If you find any other causes, you should output them as 'outro'.
    """]
)
gravidez_attributes = AttributesInterface(
    context_definition="""
        You are a medical expert and help extract information about the patient's pregnancy from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the some informations about breast-feeding, outbreak and modification drugs in the text. If you don't find any information, you should say 'No informations mentioned'.
""",
    user_prompt="""
     <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the pregnancy that you may find, what are the names of the informations?

            {json_template}

            If there are no informations mentioned you should output:
            ```json
            {{
                "amamentacao": "Not found"
                "surto": "Not found"
                "tratou": "Not found"
                "momento_surto": "Not found"
                "dmd": "Not found"
            }}
            ```

            REMEMBER: The names of the informations and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple informations mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
""",
    json_extracion_template="""
            ```json
            {{
                "gravidez": [
                    {{
                        "amamentacao": <sem_dmd/c_dmd/no/other>,
                        "surto": <yes/no>,
                        "tratou": <yes/no>,
                        "momento_surto": <pri_trimestre/seg_trimestre/ter_trimestre/no/other>,
                        "dmd": <yes/no>
                    }},
                    ...
                ]
            }}
            ```
        """,
    pre_intruct_prompts=[
        """
        Some examples of drugs that you may find in the text are:
        - ciclofosfamida
        - fingolimode      
        - betainterferona (type 1a: rebif,avonex, type 1b: betaferon) 
        - glatirâmer
        - interferon beta
        - fumarato de dimetila
        The periods of pregnancy that you can find about the outbreak are:
        - primeiro trimestre (pri_trimestre, first trimester)
        - segundo trimestre (seg_trimestre, second trimester)
        - terceiro trimestre (ter_trimestre, third trimester)
        If you find any other periods, you should output them as 'other'.
        NOTICE: If you dont find any outbreak mentioned, "tratou" should be "no" and "momento_surto" should be "no". 
        """]
)
liquor_attributes = AttributesInterface(
    context_definition="""
    You are a medical expert and help extract information about the patient's liquor from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the some informations about the liquor in the text. If you don't find any information, you should say 'No informations mentioned'.""",
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the liquor that you may find, what are the names of the informations?

            {json_template}

            If there are no informations mentioned you should output:
            ```json
            {{
                "albumina_lcr": "Not found",
                "albumina_soro": "Not found",
                "igg_lcr": "Not found",
                "igg_soro": "Not found",
                "igg_index_lcr": "Not found",
                "leucocitos": "Not found",
                "proteinas": "Not found",
                "bandas_oligoclonais": "Not found"
            }}
            ```

            REMEMBER: The names of the informations and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple informations mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
            NOTICE:  "bandas_oligoclonais" only accepts "no", "so_lcr" (This means that the oligoclonal bands are only in the liquor) and "lcr_plasma" (This means that the oligoclonal bands are in the liquor and plasma).
        """,
    json_extracion_template="""
            ```json
            {{
                "liquor": [
                    {{
                        "albumina_lcr": <extracted_result>,
                        "albumina_soro": <extracted_result>,
                        "igg_lcr": <extracted_result>,
                        "igg_soro": <extracted_result>,
                        "igg_index_lcr": <extracted_result>,
                        "leucocitos": <extracted_result>,
                        "proteinas": <extracted_result>,
                        "bandas_oligoclonais": <no/so_lcr/lcr_plasma>
                    }},
                    ...
                ]
            }}
            ```
        """,
    pre_intruct_prompts=[
        """
        Examples of some informations, that you may find in the text are:
        - albumina in lcr (albumin in CSF)
        - albumina in soro (albumin in serum)
        - igg in lcr (igg in CSF)
        - igg in soro (igg in serum)
        - igg index in lcr (igg index in CSF)
        - leucocitos
        - proteins
        - oligoclonal bands
        """])
pot_evoc_attributes = AttributesInterface(
    context_definition="""
    You are a medical expert and help extract information about the patient's evoked potentials from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the some informations about the evoked potentials in the text. If you don't find any information, you should say 'No informations mentioned'.""",
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the evoked potentials that you may find, what are the names of the informations?

            {json_template}

            If there are no informations mentioned you should output:
            ```json
            {{
                "visual": "Not found",
                "somato": "Not found",
                "auditivo": "Not found",
            }}
            ```

            REMEMBER: The names of the informations and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple informations mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "pot_evoc": [
                    {{
                        "visual": <yes/no>,
                        "somato": <yes/no>,
                        "auditivo": <yes/no>
                    }},
                    ...
                ]
            }}
            """,
    pre_intruct_prompts=["""
    Somatosensory evoked potential evaluates the sensory pathway made up of thick sensory fibers (touch and proprioception) of the tibial nerve, with peripheral capture, in the lumbar spine and brain.
    Auditory evoked potential evaluates the path of sound, from entering the ear to the brain stem, aiming to analyze the integrity of the auditory pathways
    Visual Evoked Potential is used to diagnose visual deficit due to some type of damage to the optic nerve
"""]
)

vacinas_attributes = AttributesInterface(
    context_definition="""
    You are a medical expert and help extract information about the type of patient's vaccines from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the some informations about the vaccines in the text. If you don't find any information, you should say 'No informations mentioned'.""",
    user_prompt="""
        <<CONTEXT>>
            {context_data}

            <<QUESTION>>
            Using the information about the vaccines that you may find, what are the names of the informations?

            {json_template}

            If there are no informations mentioned you should output:
            ```json
            {{
                "tipo_vacina": "Not found"
            }}
            ```

            REMEMBER: The names of the informations and their results should be in the same format as they appear in the text.
            REMEMBER: If there are multiple informations mentioned you should list them all.
            REMEMBER: You don't need to include any other information in the output.
        """,
    json_extracion_template="""
            ```json
            {{
                "vacinas": [
                    {{
                        "tipo_vacina": <dupla_adulto/febre_amarela/hepatite_a/hepatite_b/herpes_zoster/hpv_bivalente/hpv_quadrivalente/influenza/meningococica_b/meningococica_c/
                        meningococica_quadrivalente/pneumococica_13_valente/pneumococica_23_valente/triplice_viral/varicela/covid_19/covid_19_coronavac/covid_19_oxford_astrazeneca/
                        covid_19_sputnik_v/covid_19_pfizer/covid_19_covaxin/covid_19_janssen/covid_19_butanvac/outro>
                    }},
                    ...
                ]
            }}
            """,
    pre_intruct_prompts=["""
    Examples of some vaccines that you may find in the text are:
    - dupla adulto
    - febre amarela
    - hepatite a
    - hepatite b
    - herpes zoster
    - hpv bivalente
    - hpv quadrivalente
    - influenza
    - meningococica b
    - meningococica c
    - meningococica quadrivalente
    - pneumococica 13 valente
    - pneumococica 23 valente
    - triplice viral
    - varicela
    - covid_19
    - covid_19 coronavac
    - covid_19 oxford astrazeneca
    - covid_19 sputnik v
    - covid_19 pfizer
    - covid_19 covaxin
    - covid_19 janssen
    - covid_19 butanvac
                         
    If you find any other vaccines, you should output them as 'outro'.
    """])