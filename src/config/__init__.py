from .loading_config import CHUNK_OVERLAP, CHUNK_SIZE
from .embed_config import EMB_MODEL_NAME, EMB_DEVICE
from .db_config import RELATIVE_DB_DIR
from .data_paths_config import RELATIVE_DATA_DIR
from .llm_config import MODEL_DEVICE, MODEL_NAME
from .s3_config import S3_ENDPOINT_URL

ROOT_DIR = "/home/onyxia/work/llm-open-data-insee-main"

DATA_DIR = f"{ROOT_DIR}/{RELATIVE_DATA_DIR}"
DB_DIR = f"{DATA_DIR}/{RELATIVE_DB_DIR}"



