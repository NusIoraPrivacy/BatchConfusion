from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import (
    get_peft_model, 
    LoraConfig,
    TaskType
)
from accelerate import infer_auto_device_map, init_empty_weights

def get_model_tokenizer(model_name, args):
    if "Qwen" in model_name or "Llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(model_name)
        device_map = infer_auto_device_map(model, max_memory={0: "12GiB", 1: "12GiB", 2: "12GiB", 3: "12GiB",}, 
                    no_split_module_classes=['MixtralDecoderLayer', "LlamaDecoderLayer", "Phi3DecoderLayer"])
        print(device_map)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = device_map)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        model = get_peft_model(model, peft_config)
    elif "bart" in model_name or "t5" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(args.device)
    return model, tokenizer