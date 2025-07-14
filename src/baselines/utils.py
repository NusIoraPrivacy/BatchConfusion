from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from data_util.gpt_utils import get_response
import numpy as np
import torch

def load_para_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "bart" in model_name:
        torch_dtype = "float32"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto", 
            # torch_dtype=getattr(torch, torch_dtype)
        )
    elif "llama" in model_name:
        torch_dtype = "float16"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", 
            # torch_dtype=getattr(torch, torch_dtype)
        )
    return tokenizer, model

def load_emb_model(model_name, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'gpt2' in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    elif 'llama' in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    elif "bart" in model_name:
        base_model =  AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map)
    elif "bert" in model_name:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
    elif 't5' in model_name:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map)
    return tokenizer, base_model

def standard_ans(ans, labels):
    ans = ans.strip(".")
    ans = ans.strip()
    pred = 1
    for label in labels:
        if label in ans:
            pred = label
    return pred

def get_prediction(query, choices, client, args):
    prompt = (f"Question: {query}\n Please select one of the options, and output A-D only:\n"
                f"A: {choices[0]}\n B: {choices[1]}\n C: {choices[2]}\n D: {choices[3]}"
                "Remember to output only a single character from A to D!")
    # print(prompt)
    raw_pred = get_response(client, prompt, args, args.gpt_model)
    # print(raw_pred)
    pred = standard_ans(raw_pred, candidate_labels)
    return pred

logit_range_dict = {
    "eugenesiow/bart-paraphrase": (-3, 3),
    "meta-llama/Llama-2-7b-chat-hf": (-8, 8),
    "google/flan-t5-xl": (-80, 7.5)
        }

candidate_labels = ["A", "B", "C", "D"]

def paraphrase(tokenizer, model, question, para_model, epsilon, args):
    if "paraphrase" not in para_model:
        question = f"Please paraphrase the following question: {question}"
    input_ids = tokenizer.encode(question, return_tensors='pt').to(model.device)
    n = int(1.2 * len(input_ids[0]))
    cnt = 0
    genearated_tokens = []
    lower_bound = logit_range_dict[para_model][0]
    upper_bound = logit_range_dict[para_model][1]
    outputs = model.generate(input_ids, do_sample=True, temperature = (2*(upper_bound-lower_bound)*n/epsilon), max_new_tokens=n)
    sent = tokenizer.decode(token_ids = outputs[0], skip_special_tokens=True)
    return sent

def get_token_embedding(token_id, model, model_name):
    """get the token embedding given the input ids"""
    with torch.no_grad():
        if model_name =="stevhliu/my_awesome_model":
            embeddings = model.distilbert.embeddings.word_embeddings(token_id)
            # embeddings = model.distilbert.embeddings(token_id)
        elif "bert" in model_name:
            embeddings = model.bert.embeddings.word_embeddings(token_id)
        elif 'gpt2' in model_name:
            embeddings = model.transformer.wte(token_id)
        elif 'opt' in model_name:
            try:
                embeddings = model.model.decoder.embed_tokens(token_id)
            except:
                embeddings = model.decoder.embed_tokens(token_id)
        elif 'llama' in model_name:
            try:
                embeddings = model.model.embed_tokens(token_id)
            except:
                embeddings = model.embed_tokens(token_id)
        elif 't5' in model_name:
            embeddings = model.encoder.embed_tokens(token_id)
        elif "bart" in model_name:
            embeddings = model.model.shared.weight[token_id]
        embeddings = embeddings.squeeze()
    return embeddings

def custext_priv(tokenizer, model, question, model_name, top_k, epsilon, args):
    input_id = tokenizer.encode(question, return_tensors='pt').to(model.device)[0]
    init_embeddings = get_token_embedding(input_id, model, model_name)
    # acc = 0
    for i in range(1, len(init_embeddings)-1):
        embed = init_embeddings[i]
        topk_tokens, probs = get_topk_token(embed, tokenizer, model, model_name, top_k, epsilon)
        topk_tokens = topk_tokens.cpu().numpy()
        probs = probs.cpu().numpy()
        probs = probs / probs.sum()
        new_token = np.random.choice(topk_tokens, 1, p=probs)[0]
        input_id[i] = new_token
    sent = tokenizer.decode(token_ids = input_id, skip_special_tokens=True)
    return sent

def get_topk_token(embedding, tokenizer, model, model_name, top_k, epsilon):
    """Find the tokens with top-k closest embedding and obtain their sampling probabilities."""
    closest_token = None
    if 'gpt2' in model_name:
        vocabulary = tokenizer.get_vocab()
    else:
        vocabulary = tokenizer.vocab
    token_ids = [token_id for _, token_id in vocabulary.items()]
    token_ids = torch.tensor(token_ids).to(embedding.device)
    word_embeddings = get_token_embedding(token_ids, model, model_name)
    embedding = embedding.unsqueeze(dim=0)
    embedding = embedding.expand(word_embeddings.size())
    # get top-k token ids
    distance = torch.norm(embedding - word_embeddings, 2, dim=1)
    topk_idx = distance.argsort()[:top_k]
    topk_tokens = token_ids[topk_idx]
    topk_distance = distance[topk_idx]
    # compute normalize probabilities
    min_max_dist = max(topk_distance) - min(topk_distance)
    min_dist = min(topk_distance)
    norm_dist = (topk_distance-min_dist)/min_max_dist
    norm_dist = -norm_dist
    tmp = torch.exp(epsilon*norm_dist/2)
    probs = tmp/torch.sum(tmp)
    return topk_tokens, probs

def get_customized_mapping(eps,top_k):

    df_train = pd.read_csv(f"datasets/{args.dataset}/train.tsv",'\t')
    train_corpus = " ".join(df_train.sentence)
    dev_corpus = " ".join(df_train.question)
    corpus = train_corpus + " " + dev_corpus
    word_freq = [x[0] for x in Counter(corpus.split()).most_common()]


    embedding_path = f"./embeddings/{args.embedding_type}.txt"
    embeddings = []
    idx2word = []
    word2idx = {}
    with open(embedding_path,'r') as file:
        for i,line in enumerate(file):
            embedding = [float(num) for num in line.strip().split()[1:]]
            embeddings.append(embedding)
            idx2word.append(line.strip().split()[0])
            word2idx[line.strip().split()[0]] = i
    embeddings = np.array(embeddings)
    idx2word = np.asarray(idx2word)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.asarray(embeddings / norm, "float64")


    word_hash = defaultdict(str)
    sim_word_dict = defaultdict(list)
    p_dict = defaultdict(list)
    for i in trange(len(word_freq)):
        word = word_freq[i]
        if word in word2idx:
            if word not in word_hash:
                index_list = np.dot(embeddings[word2idx[word]], embeddings.T).argsort()[::-1][:top_k]
                word_list = [idx2word[x] for x in index_list]
                embedding_list = np.array([embeddings[x] for x in index_list])
                    
                for x in word_list:
                    if x not in word_hash:
                        word_hash[x] = word
                        sim_dist_list = np.dot(embeddings[word2idx[x]], embedding_list.T)
                        min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                        min_dist = min(sim_dist_list)
                        new_sim_dist_list = [(x-min_dist)/min_max_dist for x in sim_dist_list]
                        tmp = [np.exp(eps*x/2) for x in new_sim_dist_list]
                        norm = sum(tmp)
                        p = [x/norm for x in tmp]
                        p_dict[x] = p
                        sim_word_dict[x] =  word_list

                    inf_embedding = [0] * 300
                    for i in index_list:
                        embeddings[i,:] = inf_embedding
    try:
        with open(f"./p_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt", 'w') as json_file:
            json_file.write(json.dumps(p_dict, ensure_ascii=False, indent=4))
    except IOError:
        pass
    else:
        pass
    finally:
        pass

    try:
        with open(f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt", 'w') as json_file:
            json_file.write(json.dumps(sim_word_dict, ensure_ascii=False, indent=4))
    except IOError:
        pass
    else:
        pass
    finally:
        pass


    return sim_word_dict,p_dict