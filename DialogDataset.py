import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pickle


class DialogDataset(Dataset):
    """Supreme Court conversation dataset"""

    def __init__(self, prep_data_path, in_tok, out_tok):
        self.data = pickle.load(open(prep_data_path, 'rb'))
        self.in_tok = in_tok
        self.out_tok = out_tok

    def __len__(self):
        return len(self.data)

    def collate(self, data_list):
        x_pad_value = self.in_tok.eos_token_id
        y_pad_value = self.out_tok.eos_token_id
        max_x_len = max([data_entry[0].size(1) for data_entry in data_list])
        max_y_len = max([data_entry[1].size(1) for data_entry in data_list])
        X = torch.empty(len(data_list), max_x_len, dtype=torch.long).fill_(x_pad_value)
        Y = torch.empty(len(data_list), max_y_len, dtype=torch.long).fill_(y_pad_value)
        for i, (x, y) in enumerate(data_list):
            X[i,0:x.size(1)] = x.squeeze(0)
            Y[i,0:y.size(1)] = y.squeeze(0)
        return X, Y

    def get_loader(self, batch_size):
        return DataLoader(
            self,
            batch_size = batch_size,
            collate_fn = self.collate,
            shuffle = True,
            pin_memory = True
        )

    def __getitem__(self, ind):
        data_pair = self.data[ind]
        utt_pair = (data_pair[0]["UTTERANCE"], data_pair[1]["UTTERANCE"])
        vect_pair = (self.in_tok.encode(utt_pair[0], return_tensors='pt'), 
                     self.out_tok.encode(utt_pair[1], return_tensors='pt'))
        return vect_pair


