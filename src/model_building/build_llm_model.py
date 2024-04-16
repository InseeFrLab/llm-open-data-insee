from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from config import MODEL_DEVICE, MODEL_NAME


def build_llm_model() -> HuggingFacePipeline:
    """
    Create the llm model
    """
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map=MODEL_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map=MODEL_DEVICE)
    return HuggingFacePipeline(
        pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2000)
    )