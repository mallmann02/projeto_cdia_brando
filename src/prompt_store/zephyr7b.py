prompts_dict = {
        "surtos_reg_info_surto": {
            "context_definition":"""
                    You are a medical expert, identifying from the context if there is any outbreak mentioned. The context you'll be shown is a relatory of a patient's visit to the doctor being tracked of it's multiple sclerosis. Your task is to answer the questions about the outbreak, following the json template.
                """,
            "user_prompt":"""    
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
            "json_extracion_template":"""
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
        },
        "lab_hemat": {
            "context_definition":"""
                You are a medical expert and help extract laboratory hematocrit tests from the text. The context you'll be shown is a relatory of a patient's visit to the doctor. You'll be asked to extract the names of the lab tests and their results mentioned in the text. If you don't find any lab tests, you should say 'No lab tests mentioned'.
            """,
            "user_prompt":"""
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
            "json_extracion_template":"""
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
            "pre_intruct_prompts":[
                """
                Examples of lab tests (some of them with their acronyms) that you may find in the text are:
                    - liver function tests(TGO, TGP, FA, bilirrubinas,PFH, ALT)
                    - gama-gt (GGT,gamagt)
                    - hematocrit (HT, Hct)
                """
            ]
        },
        
}