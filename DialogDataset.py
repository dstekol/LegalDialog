import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pickle
from transformers import GPT2Tokenizer

class DialogDataset(Dataset):
    """Supreme Court conversation dataset"""

    pad_value = None

    def __init__(self, prep_data_path, pad_value):
        self.data = pickle.load(open(prep_data_path, 'rb'))
        DialogDataset.pad_value = pad_value
        
    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate(data_list):
        """ Static collate function """
        return DialogDataset.collate_with_padding(data_list, DialogDataset.pad_value)
    
    @staticmethod
    def collate_with_padding(data_list, pad_value):
        """ Pads list of tensors to the same length with the given pad value, 
        and returns a batched input-output tuple."""
        max_x_len = max([data_entry[0].size(1) for data_entry in data_list])
        max_y_len = max([data_entry[1].size(1) for data_entry in data_list])
        X = torch.empty(len(data_list), max_x_len, dtype=torch.long).fill_(pad_value)
        Y = torch.empty(len(data_list), max_y_len, dtype=torch.long).fill_(pad_value)
        for i, (x, y) in enumerate(data_list):
            X[i,-x.size(1):] = x.squeeze(0)
            Y[i,0:y.size(1)] = y.squeeze(0)
        return X, Y

    @staticmethod
    def filter_token(vect, filter_token):
        """ Given a vector, removes all instances of a particular token index.
        This method is used for filtering out <EOS> tokens."""
        mask = vect != filter_token
        inds = torch.nonzero(mask).squeeze()
        new_vect = vect[inds]
        if(new_vect.dim()==0):
            new_vect = new_vect.unsqueeze(0)
        return new_vect

    def get_loader(self, batch_size, shuffle):
        """Returns a DataLoader for the dataset instance."""
        return DataLoader(
            self,
            batch_size = batch_size,
            collate_fn = DialogDataset.collate,
            shuffle = shuffle,
            pin_memory = True
        )

    def __getitem__(self, ind):
        item = self.data[ind]
        return (item[0]["TENSOR"], item[1]["TENSOR"])


