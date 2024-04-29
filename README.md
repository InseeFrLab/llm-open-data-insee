# llm-open-data

## install the requirements

> cd llm-open-data-insee
> pip install -r requirements.txt
> pre-commit install

## Build complete INSEE dataset based on parquet files stored in S3 bucket (Need S3 credential and SSP Cloud Access)

> python src/db_building/insee_data_processing.py

## To load a first version of Vectorial Database from S3 bucket
> mc cp s3/projet-llm-insee-open-data/data/chroma_database/chroma_db  ./src/data --recursive
