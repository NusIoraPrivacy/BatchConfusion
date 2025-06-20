from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_model_tokenizer(model_name, num_labels=2, args=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'qwen' in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, device_map="auto")
        base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.resize_token_embeddings(len(tokenizer))
    elif 'llama' in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        # config = AutoConfig.from_pretrained(model_name)
        # with init_empty_weights():
        #     model = LlamaForSequenceClassification._from_config(config)
        # # print(model)
        # device_map = infer_auto_device_map(model, max_memory={0: "20GiB", 1: "20GiB"}, no_split_module_classes=["LlamaDecoderLayer"])
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, device_map="auto")
    elif "roberta" in model_name.lower():
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                        num_labels=num_labels)
        base_model = base_model.to(args.device)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.pad_token
        tokenizer.eos_token_id = tokenizer.pad_token_id
    return base_model, tokenizer