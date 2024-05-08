import os
from pathlib import Path

from dotenv import load_dotenv
import s3fs
from fastapi import FastAPI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
import uvicorn

from config import RAG_PROMPT_TEMPLATE
from model_building import build_llm_model
from chain_building import load_retriever, build_chain


PROJECT_PATH = Path(__file__).resolve().parents[1]

# Load env variables from .env file
load_dotenv()

# S3 configuration
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

# Import Chroma DB from S3
DB_PATH_S3 = os.path.join(os.environ["S3_BUCKET"], os.environ["DB_KEY_S3"])
DB_PATH_LOCAL = os.path.join(PROJECT_PATH, "data", "chroma_db")
fs.get(DB_PATH_S3, DB_PATH_LOCAL, recursive=True)

# Generate prompt template
prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)

# Create a pipeline with tokenizer and LLM
retriever = load_retriever(DB_PATH_LOCAL)
llm = build_llm_model(quantization_config=True, config=True, token=os.environ["HF_TOKEN"])
chain = build_chain(retriever, prompt, llm)


# Queries objects
class RAGQueryInput(BaseModel):
    text: str


class RAGQueryOutput(BaseModel):
    context: list[str]
    question: str
    answer: str


# Build API
app = FastAPI()


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post('/chat')
async def query_rag(query: RAGQueryInput) -> RAGQueryOutput:
    response = await chain.ainvoke(query.text)
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info")
