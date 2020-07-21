import pickle
import constants
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import GPT2Tokenizer
import torch

class DialogDatasetPreprocessor(object):
    """Cleans and parses data from raw data file"""

    @staticmethod
    def preprocess_data_file(raw_data_path, train_output_path, validate_output_path, test_output_path, max_inp_length):
        
        #parse each line into fields
        data_list = DialogDatasetPreprocessor.parse_fields(raw_data_path)
        
        #filter data so that only justice-petitioner utterance pairs remain
        data_list = DialogDatasetPreprocessor.filter_utterance_pairs(data_list)
        
        DialogDatasetPreprocessor.tensorize_data_list(data_list, max_inp_length)
        
        #compile data into justice-petitioner utterance pairs
        data_pair_list = [(data_list[i], data_list[i+1]) for i in range(0, len(data_list), 2)]
        
        X, Y = zip(*data_pair_list)
        X_train, X_nontrain, Y_train, Y_nontrain = train_test_split(X, Y, test_size=0.3)
        X_validate, X_test, Y_validate, Y_test = train_test_split(X_nontrain, Y_nontrain, test_size=.333)
        
        test_data = list(zip(X_test, Y_test))
        validate_data = list(zip(X_validate, Y_validate))
        train_data = list(zip(X_train, Y_train))

        pickle.dump(train_data, open(train_output_path, 'wb'))
        pickle.dump(validate_data, open(validate_output_path, 'wb'))
        pickle.dump(test_data, open(test_output_path, 'wb'))

    @staticmethod
    def parse_fields(raw_data_path):
        data_list = []
        with open(raw_data_path, "r") as f:
            for line in tqdm(f, desc='parsing'):
                parsed_line = DialogDatasetPreprocessor.extract_data_elts(line)
                parsed_line["UTTERANCE"] = DialogDatasetPreprocessor.clean_utterance(parsed_line["UTTERANCE"])
                data_list += [parsed_line]
        return data_list

    @staticmethod
    def filter_utterance_pairs(data_list):
        return [data_list[i] for i in tqdm(range(len(data_list)), desc='filtering') if 
                     (DialogDatasetPreprocessor.is_justice(data_list[i]) 
                        and i+1 < len(data_list) 
                        and not DialogDatasetPreprocessor.is_justice(data_list[i+1]) 
                        and data_list[i]["CASE_ID"]==data_list[i+1]["CASE_ID"]) 
                     or 
                     (  not DialogDatasetPreprocessor.is_justice(data_list[i]) 
                        and i-1 >= 0 
                        and DialogDatasetPreprocessor.is_justice(data_list[i-1]) 
                        and data_list[i]["CASE_ID"]==data_list[i-1]["CASE_ID"])]
        
    @staticmethod
    def is_justice(data_entry):
        return data_entry["IS_JUSTICE"].upper().strip()=='JUSTICE'
    
    @staticmethod
    def extract_data_elts(raw_line, separator="+++$+++"):
        """Accepts a raw line from the data file, 
        and splits it into fields according to parse map."""

        elts = raw_line.split(separator)
        return {field: elts[i] for i, field in enumerate(constants.DATA_PARSE_MAP)}

    @staticmethod
    def tensorize_data_list(data_list, max_length):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        for item in tqdm(data_list, desc="tokenizing"):
            item["TENSOR"] = DialogDatasetPreprocessor.tensorize(item["UTTERANCE"], tokenizer, max_length)

    @staticmethod
    def tensorize(sent, tokenizer, max_length):
        return torch.cat((tokenizer.encode(sent, return_tensors="pt", truncation=True, max_length = max_length, add_special_tokens=False), 
                          torch.tensor([[tokenizer.eos_token_id]])), dim=1)

    #@staticmethod
    #def train_test_split(data, split_ratio):
    #    inds = list(range(len(data)))
    #    random.shuffle(inds)
    #    cutoff = int(split_ratio * len(data))
    #    train_inds = set(inds[0:cutoff])
    #    train = []
    #    test = []
    #    for i in range(len(data)):
    #        if(i in train_inds):
    #            train.append(data[i])
    #        else:
    #            test.append(data[i])
    #    return train, test

    @staticmethod
    def clean_utterance(utt):
        """Removes repeated words surrounding '--' token"""
        tokens = utt.split(" ")
        delete_tokens = []
        for i, token in enumerate(tokens):
            if(token=="--"):
                for j in range(5,0,-1):
                    temp_delete_tokens = DialogDatasetPreprocessor.find_repeats(tokens, i, j)
                    if(len(temp_delete_tokens) > 0):
                        delete_tokens += temp_delete_tokens
                        break
        cleaned_tokens = [token for i, token in enumerate(tokens) if i not in delete_tokens]
        clean_utt = " ".join(cleaned_tokens)
        return clean_utt

    @staticmethod
    def find_repeats(tokens, ind, window_size):
        """Finds repeating token sequences of a given length surrounding a given index"""
        for i in range(window_size):
            a = ind - window_size + i
            b = ind + 1 + i
            if(a < 0 or b >= len(tokens) or tokens[a].upper()!=tokens[b].upper()):
                return []
        return list(range(ind, ind + window_size + 1))


