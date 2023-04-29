from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline ,BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

class LlamaModel:
    def __init__(self):
        self.tokenizer = LlamaTokenizer.from_pretrained("samwit/koala-7b")
        self.base_model = LlamaForCausalLM.from_pretrained(
            "samwit/koala-7b",
            load_in_8bit=True,
            device_map='auto',
            quantization_config=quantization_config,
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.base_model, 
            tokenizer=self.tokenizer, 
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
