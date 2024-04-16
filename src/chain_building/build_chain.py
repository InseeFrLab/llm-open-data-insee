from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import EMB_MODEL_NAME, EMB_DEVICE
from config import DB_DIR

from model_building import build_llm_model

def build_chain():
    """ """
    prompt_template = """
Instruction: Answer the question based on the relevant context:

{context}

Question:
{question}
 """
    hf_embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL_NAME, model_kwargs={"device": EMB_DEVICE})
    vectorstore = Chroma(embedding_function=hf_embeddings,persist_directory=str(DB_DIR))
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    hf_pipeline = build_llm_model()

    llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
    return rag_chain