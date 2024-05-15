#!/bin/bash

# Import vector DB
MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT
mc cp --recursive s3/$S3_BUCKET/data/chroma_database/chroma_db/ data/chroma_db

# Run app
chainlit run app.py --host 0.0.0.0 --port 8000 -h
