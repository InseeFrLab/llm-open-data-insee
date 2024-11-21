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
python run_evaluation.py --experiment_name BUILD_CHROMA_TEST --config_mlflow ${your_mlflow_run_id_here}
```


## Parameters

The project parameters can be specified in the following ways.
Each overrides the ones before.
- Default values are read from src/config/rag.toml
- A custom config file can be provided using the `RAG_CONFIG_FILE` environment variable
```python
RAG_CONFIG_FILE=/home/onyxia/work/myconfig.toml python XXX.py
```
- Environment variables prefixed with `RAG_` are interpreted as parameters
```python
RAG_LLM_TEMPERATURE=0.3 python XXX.py
```
- A custom config file can also be provided using the `--config_file` command line argument
```python
python XXX.py --config_file /home/onyxia/work/myconfig.toml
```
- Some parameters can be specified using command line arguments.
  Depending on the application, not all parameters may be specified this way.
  Refer to `python XXX.py --help` to know which flags are available.
```python
python XXX.py --emb_device cuda
```
- Finally parameters will be loaded from a previous MLFlow run
  if the `mlflow_run_id` and `mlflow_tracking_uri` are both properly set
  using any combination of the the previous methods.
```python
RAG_MLFLOW_TRACKING_URI=ZZZZZZ python XXX.py --rag_mlflow_run_id YYYY
```
These loaded parameters override all otherwise specified values.


## Build complete INSEE dataset based on parquet files stored in S3 bucket (Need S3 credential and SSP Cloud Access)

> cd llm-open-data-insee
> pip install -r requirements.txt
> pre-commit install


> python src/db_building/insee_data_processing.py

## To load a first version of Vectorial Database from S3 bucket
> mc cp -r s3/projet-llm-insee-open-data/data/chroma_database/chroma_db/  data/chroma_db
