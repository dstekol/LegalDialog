import pickle
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import GPT2Tokenizer
import torch

class DialogDatasetPreprocessor(object):
    """Cleans and parses data from raw data file"""

    # stores the order in which fields appear in each line of the raw dataset
    DATA_PARSE_MAP = ["CASE_ID", "UTTERANCE_ID", "AFTER_PREVIOUS", "SPEAKER", "IS_JUSTICE", "JUSTICE_VOTE", "PRESENTATION_SIDE", "UTTERANCE"]

    @staticmethod
    def preprocess_data_file(raw_data_path, train_output_path, validate_output_path, test_output_path, max_inp_length):
        """Given the path to the raw Supreme Court dialog dataset, extracts fields, 
        truncates utterances if needed, creates train-val-test split, and saves processed data files."""

        # parse each line into fields
        data_list = DialogDatasetPreprocessor.parse_fields(raw_data_path)
        
        # filter data so that only justice-petitioner utterance pairs remain
        data_list = DialogDatasetPreprocessor.filter_utterance_pairs(data_list)
        
        DialogDatasetPreprocessor.tensorize_data_list(data_list, max_inp_length)
        
        # compile data into justice-petitioner utterance pairs
        data_pair_list = [(data_list[i], data_list[i+1]) for i in range(0, len(data_list), 2)]
        
        # dataset split: train=0.7, validation=0.2, test=0.1 
        X, Y = zip(*data_pair_list)
        # split data into training data and nontraining (test & validation) data
        X_train, X_nontrain, Y_train, Y_nontrain = train_test_split(X, Y, test_size=0.3)
        # split nontraining data into test data and validation data
        X_validate, X_test, Y_validate, Y_test = train_test_split(X_nontrain, Y_nontrain, test_size=.333)
        
        test_data = list(zip(X_test, Y_test))
        validate_data = list(zip(X_validate, Y_validate))
        train_data = list(zip(X_train, Y_train))

        pickle.dump(train_data, open(train_output_path, 'wb'))
        pickle.dump(validate_data, open(validate_output_path, 'wb'))
        pickle.dump(test_data, open(test_output_path, 'wb'))

    @staticmethod
    def parse_fields(raw_data_path):
        """ Iterates through file and parses each line. """
        data_list = []
        with open(raw_data_path, "r") as f:
            for line in tqdm(f, desc='parsing'):
                parsed_line = DialogDatasetPreprocessor.extract_data_elts(line)
                parsed_line["UTTERANCE"] = DialogDatasetPreprocessor.clean_utterance(parsed_line["UTTERANCE"])
                data_list += [parsed_line]
        return data_list

    @staticmethod
    def filter_utterance_pairs(data_list):
        """ Removes all justice utterances not followed by petitioner utterances,
       and all petitioner utterances not preceded by justice utterances.
       This allows us to split the entire dataset into justice-petitioner query-response pairs."""
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
        """Returns whether or not a given data entry is an utterance from a justice."""
        return data_entry["IS_JUSTICE"].upper().strip()=='JUSTICE'
    
    @staticmethod
    def extract_data_elts(raw_line, separator="+++$+++"):
        """Accepts a raw line from the data file, 
        and splits it into fields according to parsing map."""

        elts = raw_line.split(separator)
        return {field: elts[i] for i, field in DATA_PARSE_MAP}

    @staticmethod
    def tensorize_data_list(data_list, max_length):
        """ Converts utterances to pytorch tensors using huggingface tokenizer. """
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        for item in tqdm(data_list, desc="tokenizing"):
            item["TENSOR"] = DialogDatasetPreprocessor.tensorize(item["UTTERANCE"], tokenizer, max_length)

    @staticmethod
    def tensorize(sent, tokenizer, max_length):
        """ Converts a given sentence to a pytorch tensor using the given tokenizer, 
        truncating the tensor if needed. Used by tensorize_data_list() method. """
        return torch.cat((tokenizer.encode(sent, return_tensors="pt", truncation=True, max_length = max_length, add_special_tokens=False), 
                          torch.tensor([[tokenizer.eos_token_id]])), dim=1)

    @staticmethod
    def clean_utterance(utt):
        """Removes repeated word sequences surrounding the '--' token"""
        tokens = utt.split(" ")
        delete_tokens = []
        for i, token in enumerate(tokens):
            if(token=="--"):
                # checks for repeating word sequences around '--' token with window sizes from 5 to 1
                for j in range(5,0,-1):
                    temp_delete_tokens = DialogDatasetPreprocessor.find_repeats(tokens, i, j)
                    
                    # if repeating sequence found, add indices to list of tokens to delete
                    # (do not delete immediately since we are still iterating through sentence)
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


