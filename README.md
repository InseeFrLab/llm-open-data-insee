# SSPCloud open data chatbot

## Running the app in local

TBD

## Evaluating best performing model

We use `MLFlow` to centralize all training performance. 
To reproduce our examples in your MLFlow store, use the 
following instructions:


1. Build the database (remove `--max_pages 20` if you want to build the whole database)

```python
python run_build_database.py --max_pages 20 --experiment_name "BUILD_CHROMA_TEST"
```

2. Evaluate model performance. If MLFlow has been used in previous example and you know the run id (see below for cases where you don't know it), you can use the following

```python
python run_evaluation.py --experiment_name BUILD_CHROMA_TEST --database_run_id ${your_run_id_here}
```



## Build complete INSEE dataset based on parquet files stored in S3 bucket (Need S3 credential and SSP Cloud Access)

> cd llm-open-data-insee
> pip install -r requirements.txt
> pre-commit install


> python src/db_building/insee_data_processing.py

## To load a first version of Vectorial Database from S3 bucket
> mc cp -r s3/projet-llm-insee-open-data/data/chroma_database/chroma_db/  data/chroma_db
