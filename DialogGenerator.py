import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Softmax
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tensorboardX
import pickle
from DialogDataset import DialogDataset
from train_utils import create_optimizer, create_scheduler, step_model
from tqdm import tqdm
import math

class DialogGenerator:
    """description of class"""

    def __init__(self, trained_path, save_path):
        super(DialogGenerator, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gen_model = GPT2LMHeadModel.from_pretrained("gpt2" if trained_path is None else trained_path, 
                                                         pad_token_id=self.tokenizer.eos_token_id)
        self.step = 0
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        #self.device = torch.device("cpu") #remove______________________________________
        self.gen_model.to(self.device)
        self.save_path = save_path
        if(save_path is not None):
            self.writer = tensorboardX.SummaryWriter(save_path + "tensorboard/")
        
    def set_optimizer(self, opt_params):
        self.optimizer = create_optimizer(self.gen_model, opt_params["weight_decay"], opt_params["lr"], opt_params["epsilon"])
        self.scheduler = create_scheduler(self.optimizer, opt_params["warmup_steps"], opt_params["total_steps"])

        
    def train_traditional(self, trainloader, num_epochs, max_out_length):
        self.gen_model.train()
        for epoch in tqdm(range(num_epochs), desc="epochs"):
            for x, y in tqdm(trainloader, desc="batches"):
                x, y = x.to(self.device), y.to(self.device)
                output, losses, true_output = self.generate_with_forcing(x, y, 1, max_out_length)
                loss = losses.sum()
                step_model(self, loss, False, self.writer, "gen_loss")
            self.save_checkpoint(x, y, output, true_output) # deindent___________________________
            self.epoch += 1

    def train_adversarial(self, trainloader, num_epochs, max_out_length, discriminator, train_disc_only_steps):
        self.gen_model.train()
        for epoch in tqdm(range(num_epochs), desc="epochs"):
            for x, y in tqdm(trainloader, desc="batches"):
                x, y = x.to(self.device), y.to(self.device)
                output, losses, true_output = self.generate_with_forcing(x, y, 1, max_out_length)
                gen_loss, disc_loss = discriminator.weight_losses(x, y, true_output, losses, self.tokenizer)
                gen_loss, disc_loss = gen_loss.to(self.device), disc_loss.to(self.device)
                if(self.step >= train_disc_only_steps):
                    step_model(self, gen_loss, True, self.writer, "gen_loss")
                step_model(discriminator, disc_loss, False, self.writer, "disc_loss")
            self.save_checkpoint(x, y, output, true_output) # unindent________________________
            discriminator.save_checkpoint() #unindent___________________________________________
            self.epoch += 1
            discriminator.epoch += 1

    def save_checkpoint(self, x, y, output, true_output):
        save_file = self.save_path + "epoch_" + str(self.epoch) + "_gen"
        self.gen_model.save_pretrained(save_file)
        with open(self.save_path + "epoch_" + str(self.epoch) + "sample.txt", "w") as f:
            for inp, label, out, true_out in zip(x, y, output, true_output):
                text_inp = self.tokenizer.decode(DialogDataset.filter_token(inp, self.tokenizer.eos_token_id))
                text_label = self.tokenizer.decode(DialogDataset.filter_token(label, self.tokenizer.eos_token_id))
                text_out = self.tokenizer.decode(DialogDataset.filter_token(out, self.tokenizer.eos_token_id))
                f.write("Input:\n" + text_inp + "\n")
                f.write("Label:\n" + text_label + "\n")
                f.write("True Output:\n" + self.get_generated_seqs(out, true_out) + "\n\n")

    def get_generated_seqs(self, out, true_out):
        sents = []
        for i, token_id in enumerate(true_out):
            if(token_id==self.tokenizer.eos_token_id):
                break
            sent_ids = torch.cat((out[0:i], true_out[i].unsqueeze(0)))
            sents.append(self.tokenizer.decode(sent_ids).strip())
        return "\n".join(sents)

    def generate_with_forcing(self, x, y, forcing_ratio, max_length):
        ce_loss = CrossEntropyLoss()
        batches = y.size(0)
        max_length = min(max_length, y.size(1))
        #max_length = 3 # remove__________________________________________________________
        generated = torch.zeros(batches, 0, dtype=torch.long).to(self.device)
        true_generated = torch.zeros(batches, max_length, dtype=torch.long).to(self.device)
        losses = torch.empty(batches, max_length).to(self.device)
        for i in range(max_length):
            input = torch.cat((x, generated), dim=1)
            output = self.gen_model(input)
            logits = output[0][:, -1, :]
            gen_words = torch.argmax(logits, dim=-1)
            losses[:, i] = ce_loss(logits, y[:, i])
            teacher_words = y[:, i]
            rand = torch.rand(gen_words.size()).to(self.device)
            word_selection = (rand < forcing_ratio).float()
            output_words = word_selection * teacher_words + (1 - word_selection) * gen_words
            output_words = output_words.long().unsqueeze(1)
            generated = torch.cat((generated, output_words), dim=1)
            true_generated[:,i] = gen_words
        return generated, losses, true_generated

    def eval(self, test_loader, max_length):
        with torch.no_grad():
            self.gen_model.eval()
            word_repetitions = []
            perplexities = torch.empty(0, dtype=torch.float).to(self.device)
            for x, y in tqdm(test_loader, desc="eval"):
                probs = self.get_probs(x.to(self.device), y.to(self.device))
                perplexities = torch.cat((perplexities, self.calc_perplexities(probs)))
                out = generator.gen_model.generate(x, max_length=max_length, early_stopping=True)[:,x.size(1):][0].tolist()
                out_words = set(out)
                word_repetitions.append(len(out) - len(out_words))
            avg_reps = sum(word_repetitions) / len(word_repetitions)
            return perplexities.mean().item(), avg_reps

    def calc_perplexities(self, probs):
        perps = (probs.log().sum(dim=1) * -1 / probs.size(1)).exp()
        return perps

    def get_probs(self, x, y):
        softmax_fn = Softmax(dim=1)
        batches, max_length = y.size()
        probs = torch.ones_like(y).float().to(self.device)
        for i in range(max_length):
            input = torch.cat((x, y[:,0:i]), dim=1)
            output = self.gen_model(input)
            logits = output[0][:, -1, :]
            vocab_dist = softmax_fn(logits)
            inds = y[:,i].unsqueeze(1)
            probs[:,i] = vocab_dist.gather(1, inds).squeeze(1).clamp(max = 1 - 1e-5)
        return probs

    def generate(self, sent, max_length, num_beams):
        with torch.no_grad():
            x = self.tokenizer.encode(sent, return_tensors='pt').to(self.device)
            if(num_beams==0):
                output = self.gen_model.generate(x, max_length=max_length)
            else:
                beams = self.gen_model.generate(x, max_length=max_length, num_beams=num_beams, early_stopping=True)
                output = beams[0]
            output = DialogDataset.filter_token(output.squeeze(), self.tokenizer.eos_token_id)
            return self.tokenizer.decode(output)
    


