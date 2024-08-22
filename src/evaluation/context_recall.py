# This is an attempt to create a functional and useful LLM-as-a-judge context recall metrics to evaluate 
# the retriever capacity based on a given context (documents) and a reference answer (from the test dataset question-answer). 
import json
import re

import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

TEMPLATE_CONTEXT_RECALL = """ 
<|assistant|>
Pour CHAQUE phrase de la réponse attendue ci-dessous, déterminez si la phrase peut être attribuée aux documents fournis comme contexte. Veuillez générer une liste de JSON avec deux clés : « verdict » et « raison ».
La clé « verdict » doit STRICTEMENT être un « oui » ou un « non ». Répondez « oui » si la phrase peut être attribuée à n'importe quelle partie du contexte de récupération, sinon répondez « non ».
La clé « raison » doit fournir une raison pour le verdict. Dans la raison, vous devez viser à inclure le nombre de documents dans le contexte de récupération (par exemple, le 1er document et le 2ème document dans le contexte de récupération) qui sont attribués à ladite phrase. Vous devez également viser à citer la partie spécifique du contexte de récupération pour justifier votre verdict, mais restez extrêmement concis et raccourcissez la citation avec des points de suspension si possible.

IMPORTANT : Respecte OBLIGATOIREMENT le JSON schema suivant : 
{{
    "type": "object",
    "properties": {{
        "verdicts": {{
            "type": "array","items": {{
                "type": "object",
                "properties": {{
                    "raison": {{
                        "type": "string"
                    }},
                    "verdict": {{
                        "type": "string","enum": ["oui", "non"]
                    }}
                }},
                "required": ["raison", "verdict"]
                }}
            }}
        }},
    "required": ["verdicts"]
}}

Voici un exemple de Réponse: 
{{
    "verdicts" : [
    {{  
        "raison": "L'âge de stabilisation du patrimoine a changé et s'est décalé", 
        "verdict": "oui"
    }}, 
    {{
        "raison": "Entre 50 et 59 ans en 1998, il est maintenant entre 70 et 74 ans",
        "verdict": "oui"
    }}
    ]
}}

Ecrivez UNIQUEMENT en format JSON et en Français.
Puisque vous allez générer un verdict pour chaque phrase, le nombre de « verdict » DEVRAIT ÊTRE STRICTEMENT ÉGAL au nombre de phrases de la « Réponse attendue ».

Réponse attendue:
{expected_output}

Contexte:
{retrieval_context}
<|end|>
<|assistant|>

Réponse:  
"""

def load_judge(llm_judge_name: str) -> pipeline:
    # Load LLM config 
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=llm_judge_name, trust_remote_code=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=llm_judge_name)
    
    # Load quantization config 
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=False,
    )

    # Load LLM 
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=llm_judge_name,
        config=config,
        quantization_config=quantization_config,
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True,
    )

    return pipeline("text-generation", model=model, tokenizer=tokenizer)


# Custom parser
def extract_json(text) -> list[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    # Define the regular expression pattern to match JSON blocks
    pattern = r"```json(.*?)```"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception as e:
        raise ValueError(f"Failed to parse: {text}") from e


def calculate_recall_score(verdicts):
    # Check if verdicts is a dictionary with the key "verdicts"

    if isinstance(verdicts, list):
        list_verdicts = verdicts
    elif isinstance(verdicts, dict): 
        if "verdicts" in verdicts:
            list_verdicts = verdicts["verdicts"]
        else:
            return np.nan
    else:
        return np.nan
    
    number_of_verdicts = 0
    justified_sentences = 0

    for verdict in list_verdicts:
        if "verdict" in verdict and "raison" in verdict: 
            number_of_verdicts +=1
            if verdict["verdict"].lower() == "oui":
                justified_sentences += 1

    score = np.nan if number_of_verdicts == 0 else justified_sentences / number_of_verdicts
    return score

def clean_results(results: dict):
    df = pd.DataFrame.from_dict(results)
    df_filtered = df.replace(to_replace='None', value=np.nan)
    df_filtered = df_filtered[df_filtered["verdicts"].apply(lambda x : not isinstance(x, list) or len(x) > 0 )]
    df_filtered = df_filtered.dropna(inplace= False)
    return df_filtered.to_dict(orient='list')

def compute_context_recall_score(results: dict):
    if "verdicts" in results:
        # compute the context recall 
        context_recall_score = []
        for verdicts in results["verdicts"]:
            s = calculate_recall_score(verdicts)
            context_recall_score.append(s) 
        results["score"] = context_recall_score
        print("Context Recall have been sucessfully computed")
    else: 
        print("Error no key 'verdicts' in results")