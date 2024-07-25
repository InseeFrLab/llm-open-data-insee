
EXPERIMENT=BUILD_CHROMA_TEST
DATA_PATH=data/raw_data/applishare_solr_joined.parquet
COLLECTION_NAME=insee_data
EMBEDDING_MODEL=OrdalieTech/Solon-embeddings-large-0.1
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
MAX_PAGES=25
# SEPARATORS="['\n\n', '\n', '.', ' ', '']"
# MAX_NEW_TOKENS=2000

python run_pipeline.py --experiment_name ${EXPERIMENT} \
    --data_raw_s3_path ${DATA_PATH} \
    --collection_name ${COLLECTION_NAME} \
    --embedding_model ${EMBEDDING_MODEL} \
    --llm_model ${LLM_MODEL} \
    --max_pages ${MAX_PAGES}

    # --markdown_split \
    # --use_tokenizer_to_chunk \
    # --separators ${SEPARATORS} \
    # --quantization \
    # --max_new_tokens ${MAX_NEW_TOKENS} \
    # --chunk_size \
    # --chunk_overlap \
    # --reranking_method
