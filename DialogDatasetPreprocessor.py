import pickle
import constants
import random

class DialogDatasetPreprocessor(object):
    """Cleans and parses data from raw data file"""

    @staticmethod
    def preprocess_data_file(raw_data_path, train_output_path, test_output_path, split_ratio):
        
        #parse each line into fields
        data_list = DialogDatasetPreprocessor.parse_fields(raw_data_path)
        
        #filter data so that only justice-petitioner utterance pairs remain
        data_list = DialogDatasetPreprocessor.filter_utterance_pairs(data_list)
        
        #compile data into justice-petitioner utterance pairs
        data_pair_list = [(data_list[i], data_list[i+1]) for i in range(0, len(data_list), 2)]

        train_data, test_data = DialogDatasetPreprocessor.train_test_split(data_pair_list, split_ratio)
        
        pickle.dump(train_data, open(train_output_path, 'wb'))
        pickle.dump(test_data, open(test_output_path, 'wb'))

    @staticmethod
    def parse_fields(raw_data_path):
        data_list = []
        with open(raw_data_path, "r") as f:
            for line in f:
                parsed_line = DialogDatasetPreprocessor.extract_data_elts(line)
                parsed_line["UTTERANCE"] = DialogDatasetPreprocessor.clean_utterance(parsed_line["UTTERANCE"])
                data_list += [parsed_line]
        return data_list

    @staticmethod
    def filter_utterance_pairs(data_list):
        return [data_list[i] for i in range(len(data_list)) if 
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
    def train_test_split(data, split_ratio):
        inds = list(range(len(data)))
        random.shuffle(inds)
        cutoff = int(split_ratio * len(data))
        train_inds = set(inds[0:cutoff])
        train = []
        test = []
        for i in range(len(data)):
            if(i in train_inds):
                train.append(data[i])
            else:
                test.append(data[i])
        return train, test

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


