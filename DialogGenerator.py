import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import DialogDiscriminator
import tensorboardX
import pickle

class DialogGenerator(torch.nn.Module):
    """description of class"""

    def __init__(self, trained_path, meta_path, save_path):
        super(DialogGenerator, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gen_model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self.tokenizer.eos_token_id)
        self.step = 0
        self.epoch = 0
        if(trained_path is not None):
            self.gen_model.load_state_dict(torch.load(trained_path))
        if(meta_path is not None):
            meta = pickle.load(open(meta_path, 'rb'))
            self.step = meta.step
            self.epoch = meta.epoch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.gen_model.to(self.device)
        self.save_path = save_path
        self.writer = tensorboardX.SummaryWriter(save_path + "tensorboard/")

    def train_traditional(self, trainloader, num_epochs, forcing_ratio, optimizer, scheduler):
        self.gen_model.train()
        
        for epoch in range(num_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                output, losses, true_output = self.generate_with_forcing(x, y, forcing_ratio)
                loss = losses.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.writer.add_scalar("gen_loss", loss, self.step)
                self.step += 1
                self.save_checkpoint(x, y, output, true_output) # deindent
            self.epoch += 1

    def train_adv(self, trainloader, num_epochs, forcing_ratio, optimizer, scheduler, discriminator):
        self.gen_model.train()
        for epoch in range(num_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                output, losses, true_output = self.generate_with_forcing(x, y, forcing_ratio)
                loss = discriminator.weight_losses(output, losses, self.tokenizer)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.writer.add_scalar("gen_loss", loss, self.step)
                self.step += 1
                discriminator.update(output, y)
                self.save_checkpoint(x, y, output, true_output) # deindent
                discriminator.save_checkpoint()
            self.epoch += 1
        optimizer = self.create_optimizer(args)
        scheduler = self.create_scheduler(optimizer, args)
        for epoch in range(num_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                output, losses, true_output = self.generate_with_forcing(x, y, forcing_ratio)
                loss = discriminator.weight_losses(output, losses)
                optimizer.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()
                discriminator.update(output, y)

            self.save_checkpoint(epoch, x, y, output, true_output)
            discriminator.save_model(epoch)

    def save_checkpoint(self, x, y, output, true_output):
        torch.save(self.gen_model.state_dict, self.save_path + "epoch_" + str(self.epoch) + "_gen.torch")
        pickle.dump({epoch: self.epoch, step: self.step}, self.save_path + "epoch_" + str(self.epoch) + "_meta.pkl")
        with open(self.save_path + "epoch_" + str(self.epoch) + "sample.txt", "w") as f:
            for inp, label, out, true_out in zip(x, y, output, true_output):
                text_inp = self.tokenizer.decode(self.filter_token(inp, self.tokenizer.eos_token_id))
                text_label = self.tokenizer.decode(self.filter_token(label, self.tokenizer.eos_token_id))
                text_out = self.tokenizer.decode(self.filter_token(out, self.tokenizer.eos_token_id))
                f.write("Input:\n" + text_inp + "\n")
                f.write("Label:\n" + text_label + "\n")
                f.write("True Output:\n" + self.get_generated_seqs(out, true_out) + "\n\n")

    def get_generated_seqs(self, out, true_out):
        sents = []
        for i, token_id in enumerate(true_out):
            if(token_id==self.tokenizer.eos_token_id):
                break
            sent_ids = torch.cat((out[0:i], true_out[i].unsqueeze(0)))
            sents.append(self.tokenizer.decode(sent_ids))
        return "\n".join(sents)

    def filter_token(self, vect, filter_token):
        mask = vect != filter_token
        inds = torch.nonzero(mask).squeeze()
        return vect[inds]

    def generate_with_forcing(self, x, y, forcing_ratio):
        ce_loss = CrossEntropyLoss()
        batches, max_length = y.size()
        max_length = 3 # remove
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
            rand = torch.rand(gen_words.size())
            word_selection = (rand < forcing_ratio).float()
            output_words = word_selection * teacher_words + (1 - word_selection) * gen_words
            output_words = output_words.long().unsqueeze(1)
            generated = torch.cat((generated, output_words), dim=1)
            true_generated[:,i] = gen_words
        return generated, losses, true_generated

    
    def create_optimizer(self, weight_decay, lr, epsilon):
        param_optimizer = list(self.gen_model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=epsilon)
        return optimizer

    def create_scheduler(self, optimizer, warmup_steps, total_steps):
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return scheduler

    def eval(self, test_loader):
        #perps = []
        #with torch.no_grad():
        #    for x, y in test_loader:
        #        probs = self.get_probs(x, y)
        #        for prob_list in probs:
        #            perps.append(self.calc_perplexity(prob_list))
        #return sum(perps) / len(perps)
        perps = []
        with torch.no_grad():
            for x, y in test_loader:
                loss = self.gen_model(x, lm_labels=y)
                for prob_list in probs:
                    perps.append(self.calc_perplexity(prob_list))
        return sum(perps) / len(perps)

    def calc_perplexity(self, prob_list):
        return prob_list.prod().item() ** (-1 / len(prob_list))

    def get_probs(x, y):
        batches, max_length = y.size()
        probs = torch.empty_like(y)
        for i in range(max_length):
            input = torch.cat((x, y[:,0:i]), dim=1)
            output = self.gen_model(input)
            logits = output[0][:, -1, :]
            probs[:,i] = logits[y[:,i]]
        return probs

    def generate(self, sent, max_length):
        x = self.tokenizer.encode(sent, return_tensors='pt')
        output, _, _ = self.generate_with_forcing(x, y = torch.zeros(1, max_length), forcing_ratio = 0)
        output = self.filter_token(output.squeeze(), self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output)
    


