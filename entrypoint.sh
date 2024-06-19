#!/bin/bash

# Import vector DB
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT
export S3_BUCKET='projet-llm-insee-open-data'
# mc cp --recursive s3/$S3_BUCKET/data/chroma_database/chroma_db/ data/chroma_db

# Cache LLM
export LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
export EMB_MODEL_NAME=OrdalieTech/Solon-embeddings-large-0.1
python -m src.model_building.fetch_llm_model 2>&1

# Run app
chainlit run app.py --host 0.0.0.0 --port 8000 -h
