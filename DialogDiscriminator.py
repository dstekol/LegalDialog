from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from train_utils import create_optimizer, create_scheduler
import pickle
from DialogDataset import DialogDataset
import torch

class DialogDiscriminator:
    """description of class"""
    def __init__(self, model_type, trained_path, save_path, opt_params):
        super(DialogDiscriminator, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.disc_model = AutoModelForSequenceClassification.from_pretrained(model_type if trained_path is None else trained_path)
        self.step = 0
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu") #remove _______________________________________
        self.optimizer = create_optimizer(self.disc_model, opt_params["weight_decay"], opt_params["lr"], opt_params["epsilon"])
        self.scheduler = create_scheduler(self.optimizer, opt_params["warmup_steps"], opt_params["total_steps"])
        self.disc_model.to(self.device)
        self.disc_model.train()
        self.save_path = save_path

    def save_checkpoint(self):
        save_file = self.save_path + "epoch_" + str(self.epoch) + "_disc"
        self.disc_model.save_pretrained(save_file)

    def retokenize_batch(self, x, y1, y2, from_tok):
        retokenized_list1 = []
        retokenized_list2 = []
        sep = torch.tensor([[self.tokenizer.sep_token_id]]).to(self.device)
        cls = torch.tensor([[self.tokenizer.cls_token_id]]).to(self.device)
        for x_old, y1_old, y2_old in zip(x, y1, y2):
            x_new = torch.cat((cls, self.retokenize_vect(x_old, from_tok), sep), dim=1)
            y1_new = torch.cat((self.retokenize_vect(y1_old, from_tok), sep), dim=1)
            y2_new = torch.cat((self.retokenize_vect(y2_old, from_tok), sep), dim=1)
            retokenized_list1.append((x_new, y1_new))
            retokenized_list2.append((x_new, y2_new))
        retokenized_list = retokenized_list1 + retokenized_list2
        X, Y = DialogDataset.collate_with_padding(retokenized_list, self.tokenizer.pad_token_id)
        return torch.cat((X,Y), dim=1)

    def retokenize_vect(self, vect, from_tok):
        filtered = DialogDataset.filter_token(vect, from_tok.eos_token_id)
        return self.tokenizer.encode(from_tok.decode(filtered), add_special_tokens=False, return_tensors='pt')
        
    def weight_losses(self, x, y, gen_output, gen_losses, gen_tokenizer):
        scores = torch.empty(gen_output.size(0)*2, gen_output.size(1)).to(self.device)
        disc_losses = torch.empty(0).to(self.device)
        for i in range(gen_output.size(1)):
            gen_input = torch.cat((y[:,0:i],gen_output[:,i].unsqueeze(1)), dim=1)
            real_input = y[:,0:i+1]
            input = self.retokenize_batch(x, gen_input, real_input, gen_tokenizer)
            labels = torch.cat((
                torch.ones(gen_input.size(0)), 
                torch.zeros(real_input.size(0))), dim=0).long()
            disc_loss, logits = self.disc_model(input, labels=labels)
            disc_losses = torch.cat((disc_losses, disc_loss.unsqueeze(0)))
            scores[:,i] = logits[:,0] - logits[:,1]
        gen_scores = scores[:gen_output.size(0),:]
        true_scores = scores[-gen_output.size(0):,:]
        weights = (true_scores - gen_scores).exp()
        gen_loss = (gen_losses * weights).sum()
        return gen_loss, disc_losses.sum()
       

