from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,  BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from config import MODEL_DEVICE, DEFAULT_MODEL_NAME


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

def build_llm_model(model_name=DEFAULT_MODEL_NAME, quantization=True) -> HuggingFacePipeline:
    """
    Create the llm model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='cuda')
    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                quantization_config=bnb_config,
                                                      )
        
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda')
       
    return HuggingFacePipeline(
        pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2000)
    )

