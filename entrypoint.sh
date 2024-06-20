#!/bin/bash

# Fetch vector DB from S3
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT
mc cp --recursive s3/$S3_BUCKET/data/chroma_database/chroma_db/ data/chroma_db

# Fetch cached LLM from S3 if available
LLM_MODEL_NAME_HF=$(echo "$LLM_MODEL_NAME" | sed 's|/|--|g' | sed 's|^|models--|')
MODEL_PATH_S3=s3/$S3_BUCKET/models/hf_hub/$LLM_MODEL_NAME_HF
if mc ls $MODEL_PATH_S3 > /dev/null 2>&1; then
    echo "Fetching cached LLM $MODEL_NAME from S3."
    mc cp --recursive $MODEL_PATH_S3/ $HOME/.cache/huggingface/hub/$LLM_MODEL_NAME_HF
fi

# Run app
chainlit run app.py --host 0.0.0.0 --port 8000 -h
