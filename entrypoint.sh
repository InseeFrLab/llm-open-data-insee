#!/bin/bash

# Import vector DB
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT
mc cp --recursive s3/$S3_BUCKET/data/chroma_database/chroma_db/ data/chroma_db

# Cache LLM
python -m src.model_building.fetch_llm_model 2>&1

# Run app
chainlit run app.py --host 0.0.0.0 --port 8000 -h

# Run dev app 
chainlit run dev_app.py --host 0.0.0.0 --port 8000 -h -w

