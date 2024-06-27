from langchain.agents import Agent, AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda 

# Prompt Template with In-Context Learning and Justification
EVAL_INSTRUCTION = """
L'INSEE (Institut National de la Statistique et des Études Économiques) est l'organisme national chargé de la production, de l'analyse et de la publication des statistiques officielles en France. Les domaines d'expertise de l'INSEE incluent, mais ne sont pas limités à :

1. Démographie et population
2. Emploi et chômage
3. Revenus et niveaux de vie
4. Entreprises et économie
5. Comptes nationaux et finances publiques
6. Prix et indices des prix
7. Conditions de vie et pauvreté
8. Éducation et formation

Évaluez la pertinence de la question utilisateur suivante en vous assurant qu'elle est liée à l'un des domaines d'expertise de l'INSEE. Répondez par "Oui" ou "Non" et fournissez une brève justification.

Exemples :
Question: "Quel est le taux de chômage en France pour l'année 2023 ?"
Réponse: "Oui. Cette question est pertinente car elle concerne les statistiques sur l'emploi et le chômage, qui sont des domaines d'expertise de l'INSEE."

Question: "Quelle est la recette pour faire un gâteau au chocolat ?"
Réponse: "Non. Cette question n'est pas pertinente car elle concerne la cuisine, qui n'est pas un domaine d'expertise de l'INSEE."
"""

USER_INSTRUCTION = """Question utilisateur : {query}
Cette question est-elle pertinente par rapport aux domaines d'expertise de l'INSEE ? Répondez par "Oui" ou "Non" et fournissez une brève justification."""


EVAL_TEMPLATE = [
    {"role": "user", "content": """Tu es un assistant spécialisé dans la statistique publique qui filtre des requêtes entrantes."""},
    {"role": "assistant", "content": EVAL_INSTRUCTION},
    {"role": "user", "content": USER_INSTRUCTION},
]

def build_chain_validator(evaluator_llm=None, tokenizer=None):
    """
    defining a chain to check if a given query is related to INSEE expertise.  
    """
    is_query_related_to_public_statistics = lambda generation : generation.lower().find("oui") != -1

    prompt_template = tokenizer.apply_chat_template(EVAL_TEMPLATE, 
                                    tokenize=False,
                                    add_generation_prompt=True
                                    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])

    return (
        prompt 
        | evaluator_llm 
        | RunnableLambda(func=is_query_related_to_public_statistics)
    )