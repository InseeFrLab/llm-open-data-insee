#!/bin/bash

# S3 configuration
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT

# Fetch vector DB from S3
S3_DB_PATH=s3/$S3_BUCKET/data/chroma_database/chroma_db/
LOCAL_DB_PATH=data/chroma_db
if [ ! -d "$LOCAL_DB_PATH" ]; then
    echo "Fetching vector DB from S3."
    mc cp --recursive $S3_DB_PATH $LOCAL_DB_PATH 1> /dev/null
else
    echo "Vector DB is already present locally."
fi

# Fetch models from S3 cache if available
MODELS=("$LLM_MODEL_NAME" "$EMB_MODEL_NAME")
for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_NAME_HF=$(echo "$MODEL_NAME" | sed 's|/|--|g' | sed 's|^|models--|')
    S3_MODEL_PATH=s3/$S3_BUCKET/models/hf_hub/$MODEL_NAME_HF
    LOCAL_MODEL_PATH=$HOME/.cache/huggingface/hub/$MODEL_NAME_HF

    if mc ls $S3_MODEL_PATH 1> /dev/null; then
        if [ ! -d "$LOCAL_MODEL_PATH" ]; then
            echo "Fetching model $MODEL_NAME from S3."
            mc cp --recursive $S3_MODEL_PATH/ $LOCAL_MODEL_PATH 1> /dev/null
        else
            echo "Model $MODEL_NAME is already present locally."
        fi
    fi
done

# Run app
chainlit run app.py --host 0.0.0.0 --port 8000 -h
