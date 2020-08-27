import pickle
from transformers import GPT2Tokenizer

def count_ngrams(data, n):
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    token_set = set()
    for item in data:
        g_out = tok.encode(item["gen_out"])
        token_set.update(get_ngrams(g_out, n))
    return len(token_set)

def get_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams
