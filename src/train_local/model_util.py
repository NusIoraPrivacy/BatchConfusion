from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import (
    get_peft_model, 
    LoraConfig,
    TaskType
)
from accelerate import infer_auto_device_map, init_empty_weights
from collections import OrderedDict

def get_model_tokenizer(model_name, args):
    if "Qwen" in model_name or "Llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_pretrained(model_name)
        # device_map = infer_auto_device_map(model, max_memory={0: "0GiB", 1: "0GiB", 2: "14GiB", 3: "14GiB",}, 
        #             no_split_module_classes=['MixtralDecoderLayer', "LlamaDecoderLayer", "Phi3DecoderLayer"])
        # print(device_map)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto")
        # print(model)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)
    elif "gemma" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_pretrained(model_name)
        # # print(model)
        # device_map = infer_auto_device_map(model, max_memory={0: "15GiB", 1: "15GiB", 2: "0GiB", 3: "15GiB",}, 
        #             no_split_module_classes=["Gemma2DecoderLayer"])
        # print(device_map)
        device_map = OrderedDict([('model.embed_tokens', 0), ('model.layers.0', 0), ('model.layers.1', 0), ('model.layers.2', 0), ('model.layers.3', 0), ('model.layers.4', 0), ('model.layers.5', 0), ('model.layers.6', 0), ('model.layers.7', 0), ('model.layers.8', 0), ('model.layers.9', 0), ('model.layers.10', 0), ('model.layers.11', 1), ('model.layers.12', 1), ('model.layers.13', 1), ('model.layers.14', 1), ('model.layers.15', 1), ('model.layers.16', 1), ('model.layers.17', 1), ('model.layers.18', 1), ('model.layers.19', 1), ('model.layers.20', 1), ('model.layers.21', 1), ('model.layers.22', 1), ('model.layers.23', 1), ('model.layers.24', 1), ('model.layers.25', 1), ('model.layers.26', 1), ('model.layers.27', 1), ('model.layers.28', 1), ('model.layers.29', 1), ('model.layers.30', 1), ('model.layers.31', 3), ('model.layers.32', 3), ('model.layers.33', 3), ('model.layers.34', 3), ('model.layers.35', 3), ('model.layers.36', 3), ('model.layers.37', 3), ('model.layers.38', 3), ('model.layers.39', 3), ('model.layers.40', 3), ('model.layers.41', 3), ('model.norm', 3), ('lm_head', 0)])
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = device_map)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, peft_config)
    elif "bart" in model_name or "t5" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(args.device)
    return model, tokenizer