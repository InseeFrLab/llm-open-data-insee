import os

# SSPCLOUD RELATED PARAMETERS ----------------------------

S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
S3_BUCKET = "projet-llm-insee-open-data"
MLFLOW_TRACKING_URI = "https://projet-llm-insee-open-data-mlflow.user.lab.sspcloud.fr"
MLFLOW_S3_ENDPOINT_URL = "https://minio.lab.sspcloud.fr"


# LOCAL FILES -------------------------------------

RELATIVE_DATA_DIR = "data"
RELATIVE_LOG_DIR = "logs"
LOG_FILE_PATH = f"{RELATIVE_LOG_DIR}/conversation_logs.json"


# VECTOR DATABASE ------------------------------------

DB_DIR_S3 = "data/chroma_database/chroma_db/"
CHROMA_DB_LOCAL_DIRECTORY = "data/chroma_db"
COLLECTION_NAME = "insee_data"

# MODELS USED   ---------------------------------------

# Embedding model
EMB_DEVICE = "cuda"
EMB_MODEL_NAME = "OrdalieTech/Solon-embeddings-large-0.1"

# Model
MODEL_DEVICE = {"": 0}
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# PARSING  ----------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MARKDOWN_SEPARATORS = ["\n\n", "\n", ".", " ", ""]


# INSTRUCTION PROMPT --------------------------------

BASIC_RAG_PROMPT_TEMPLATE = """
<s>[INST]
Instruction: Réponds à la question en te basant sur le contexte donné:

{context}

Question:
{question}
[/INST]
"""

RAG_PROMPT_TEMPLATE = """
<s>[INST]
Tu es un assistant spécialisé dans la statistique publique répondant aux questions d'agent de l'INSEE.
Réponds en Français seulement.
Utilise les informations obtenues dans le contexte, réponds de manière argumentée à la question posée.
La réponse doit être développée et citer ses sources.

Si tu ne peux pas induire ta réponse du contexte, ne réponds pas.
Voici le contexte sur lequel tu dois baser ta réponse :
Contexte: {context}
        ---
Voici la question à laquelle tu dois répondre :
Question: {question}
[/INST]
"""

# CHATBOT CONFIGURATION --------------------------------------

CHATBOT_INSTRUCTION = """
En utilisant UNIQUEMENT les informations présentes dans le contexte, réponds de manière argumentée à la question posée.
La réponse doit être développée et citer ses sources.

Si tu ne peux pas induire ta réponse du contexte, ne réponds pas.
"""

USER_INSTRUCTION = """Voici le contexte sur lequel tu dois baser ta réponse :
Contexte:
{context}
---
Voici la question à laquelle tu dois répondre :
Question: {question}"""

CHATBOT_TEMPLATE = [
    {
        "role": "user",
        "content": """Tu es un assistant spécialisé dans la statistique publique.
    Tu réponds à des questions concernant les données de l'Insee, l'institut national statistique Français.
    Réponds en FRANCAIS UNIQUEMENT.""",
    },
    {"role": "assistant", "content": CHATBOT_INSTRUCTION},
    {"role": "user", "content": USER_INSTRUCTION},
]
