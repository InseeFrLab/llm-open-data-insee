#!/bin/bash

# Import vector DB
mc cp --recursive s3/$S3_BUCKET/data/chroma_database/chroma_db/ data/chroma_db

# Run app
chainlit run app.py --host 0.0.0.0 --port 8000 -h
