from transformers import BertModel, BertTokenizer
import pickle
class TransformerDiscriminator(DialogDiscriminator):
    """description of class"""
    def __init__(self, trained_path, meta_path, save_path):
        super(TransformerDiscriminator, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.disc_model = BertModel.from_pretrained("bert-base-uncased", 
            pad_token_id=self.tokenizer.eos_token_id)
        self.step = 0
        self.epoch = 0
        if(trained_path is None):
            self.disc_model.load_state_dict(torch.load(trained_path))
        if(meta_path is not None):
            meta = pickle.load(open(meta_path, 'rb'))
            self.step = meta.step
            self.epoch = meta.epoch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.disc_model.to(self.device)
        self.disc_model.train()

    def save_checkpoint(self):
        torch.save(self.disc_model.state_dict, self.save_path + "epoch_" + str(self.epoch) + "_gen.torch")
        pickle.dump({epoch: self.epoch, step: self.step}, self.save_path + "epoch_" + str(self.epoch) + "_meta.pkl")

    def retokenize(self, vect, from_tok, to_tok):
        retokenized = vect.clone().detach().cpu()
        retokenized.apply_(lambda x: to_tok.convert_tokens_to_ids(from_tok.convert_ids_to_tokens(x)))
        return retokenized.to(self.device)
        
   def weight_losses(self, gen_output, losses, gen_tokenizer):
        retokenized = self.retokenize(output, gen_tokenizer, self.tokenizer)
        scores = torch.empty_like(gen_output)
        for i in range(gen_output.size(1)):
            pass
        return scores
        

    def update(self, output, y, writer):
        pass

