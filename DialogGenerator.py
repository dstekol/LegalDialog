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

    def __init__(self, trained_path, save_path, opt_params):
        super(DialogGenerator, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # instantiate pretrained model from filepath if given, otherwise instantiate default pretrained model
        self.gen_model = GPT2LMHeadModel.from_pretrained("gpt2" if trained_path is None else trained_path, 
                                                         pad_token_id=self.tokenizer.eos_token_id)
        # logging variables
        self.step = 0
        self.epoch = 0

        # switch to cuda if possible, and explicitly print which device is being used
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gen_model.to(self.device)
        print(self.device)

        # instantiates optimizer and scheduler with given configuration options
        if(opt_params is not None):
            self.optimizer = create_optimizer(self.gen_model, opt_params["weight_decay"], opt_params["lr"], opt_params["epsilon"])
            self.scheduler = create_scheduler(self.optimizer, opt_params["warmup_steps"], opt_params["total_steps"])

        # save checkpoint path
        self.save_path = save_path
        if(save_path is not None):
            self.writer = tensorboardX.SummaryWriter(save_path + "tensorboard/")
                
    def train_traditional(self, trainloader, num_epochs, max_out_length):
        """ Trains model via standard teacher forcing. """
        self.gen_model.train()
        for epoch in tqdm(range(num_epochs), desc="epochs"):
            for x, y in tqdm(trainloader, desc="batches"):
                x, y = x.to(self.device), y.to(self.device)

                # get teacher forcing losses and outputs
                losses, forced_output, gen_output = self.generate_with_forcing(x, y, max_out_length)

                # compute loss as sum of token losses
                loss = losses.sum()

                # update generator
                step_model(self, loss, False, self.writer, "gen_loss")
            self.save_checkpoint(x, y, forced_output, gen_output)
            self.epoch += 1

    def train_adversarial(self, trainloader, num_epochs, max_out_length, discriminator, train_disc_only_steps):
        """ Trains a model with adversarial weighting, using the discriminator provided. """
        self.gen_model.train()
        for epoch in tqdm(range(num_epochs), desc="epochs"):
            for x, y in tqdm(trainloader, desc="batches"):
                x, y = x.to(self.device), y.to(self.device)

                # get teacher forcing losses and outputs
                losses, forced_output, gen_output = self.generate_with_forcing(x, y, max_out_length)
                
                # get weighted loss from discriminator
                gen_loss, disc_loss = discriminator.weight_losses(x, y, gen_output, losses, self.tokenizer)
                gen_loss, disc_loss = gen_loss.to(self.device), disc_loss.to(self.device)

                # Update generator if minimum number of discriminiator warmup steps have elapsed.
                # This option allows the discriminator to be trained by itself for a certain number of steps,
                # without updating the generator.
                if(self.step >= train_disc_only_steps):
                    step_model(self, gen_loss, True, self.writer, "gen_loss")

                # udpate discriminator
                step_model(discriminator, disc_loss, False, self.writer, "disc_loss")

            # save generator and discriminator checkpoints
            self.save_checkpoint(x, y, forced_output, gen_output)
            discriminator.save_checkpoint()

            # update logging variables
            self.epoch += 1
            discriminator.epoch += 1

    def save_checkpoint(self, x, y, output, gen_output):
        # save model checkpoint
        save_file = self.save_path + "epoch_" + str(self.epoch) + "_gen"
        self.gen_model.save_pretrained(save_file)
        
        # print sample inputs and outputs
        with open(self.save_path + "epoch_" + str(self.epoch) + "sample.txt", "w") as f:
            for inp, label, out, gen_out in zip(x, y, output, gen_output):
                text_inp = self.tokenizer.decode(DialogDataset.filter_token(inp, self.tokenizer.eos_token_id))
                text_label = self.tokenizer.decode(DialogDataset.filter_token(label, self.tokenizer.eos_token_id))
                text_out = self.tokenizer.decode(DialogDataset.filter_token(out, self.tokenizer.eos_token_id))
                f.write("Input:\n" + text_inp + "\n")
                f.write("Label:\n" + text_label + "\n")
                f.write("True Output:\n" + self.get_generated_seqs(out, gen_out) + "\n\n")

    def get_generated_seqs(self, out, gen_out):
        """ Given a true sentence and a list of words generated at each index, 
            outputs list of sentence fragments such that the last word is the generated word,
            and all preceding words are derived from the true label.
            For example, given the true label out=['She', 'went', 'to', 'school'] 
            and words generated at each step gen_out=['I', 'was', 'away', 'lunch'], 
            this function outputs the list of strings
            I                   [the first word generated by the model]
            She went            [Showing that the model generated 'went' as the most likely extension to 'She']
            She went away       [Showing that the model generated 'away' as the most extension to 'She went']
            She went to lunch   [Showing that the model generated 'lunch' as the most extension to 'She went to'] """
        sents = []
        for i, token_id in enumerate(gen_out):
            if(token_id==self.tokenizer.eos_token_id):
                break
            sent_ids = torch.cat((out[0:i], gen_out[i].unsqueeze(0)))
            sents.append(self.tokenizer.decode(sent_ids).strip())
        return "\n".join(sents)

    def generate_with_forcing(self, x, y, max_length, forcing_ratio = 1):
        """ Given an input vector x and a label vector y, generates a token sequence token-by-token,
            with teacher-forcing determined by the forcing ratio parameter (defaults to 1 - always force)."""
        ce_loss = CrossEntropyLoss()
        batches = y.size(0)
        max_length = min(max_length, y.size(1))

        # tensor for storing previously generated token sequences (with teacher forcing)
        forced_generated = torch.zeros(batches, 0, dtype=torch.long).to(self.device)

        # tensor for storing the maximum-likelihood tokens outputted by the generator at each step
        true_generated = torch.zeros(batches, max_length, dtype=torch.long).to(self.device)

        # placeholder tensor for losses for each generated token
        losses = torch.empty(batches, max_length).to(self.device)

        for i in range(max_length):
            # create input vector as combination of input query x and previously generated tokens
            input = torch.cat((x, forced_generated), dim=1)

            # perform next-token prediction
            output = self.gen_model(input)

            # extract losses and token probability distribution
            logits = output[0][:, -1, :]
            gen_words = torch.argmax(logits, dim=-1)
            losses[:, i] = ce_loss(logits, y[:, i])
            
            # decide whether to use teacher forcing via random sampling based on forcing ratio
            teacher_words = y[:, i]
            rand = torch.rand(gen_words.size()).to(self.device)
            word_selection = (rand < forcing_ratio).float()
            output_words = word_selection * teacher_words + (1 - word_selection) * gen_words
            output_words = output_words.long().unsqueeze(1)

            # record the output tokens after the (probabilistic) teacher forcing has been applied
            forced_generated = torch.cat((forced_generated, output_words), dim=1)

            # record the actual (unforced) output tokens of the generator
            true_generated[:,i] = gen_words
        return losses, forced_generated, true_generated

    @staticmethod
    def get_ngrams(toks, n):
        """ Given a set of tokens, extracts a set of all n-grams present (with given length n). """
        ngrams = set()
        for i in range(len(toks) - n + 1):
            ngrams.add(tuple(toks[i:i+n]))
        return ngrams

    def eval(self, test_loader, max_length):
        with torch.no_grad():
            self.gen_model.eval()

            # initialize evaluation variables
            tok_repetitions = []
            unique_ngrams = { 1: set(), 2: set(), 3: set(), 4: set() }
            all_outs = []
            perplexities = torch.empty(0, dtype=torch.float).to(self.device)

            for x, y in tqdm(test_loader, desc="eval"):
                x, y = x.to(self.device), y.to(self.device)

                # calculate perplexity
                probs = self.get_probs(x, y)
                perplexities = torch.cat((perplexities, self.calc_perplexities(probs)))

                # get generated text (trim to exclude input)
                max_total_length = x.size(1) + max_length
                out = self.gen_model.generate(x, max_length=max_total_length, early_stopping=True)[:,x.size(1):][0].tolist()
                
                # compute intra-utterance token repetition for given sequence
                out_toks = set(out)
                tok_repetitions.append(len(out) - len(out_toks))

                # update set of unique n-grams
                for n in unique_ngrams:
                    ngrams = DialogGenerator.get_ngrams(out, n)
                    unique_ngrams[n] = unique_ngrams[n].union(set(ngrams))
                
                # store generated text, along with input query and ground truth
                all_outs.append({"in":self.tokenizer.decode(x.squeeze()),
                            "real_out": self.tokenizer.decode(y.squeeze()),
                            "gen_out": self.tokenizer.decode(out)})
            
            # compute average intra-utterance token repetition
            avg_reps = sum(tok_repetitions) / len(tok_repetitions)

            # compute average perplexity
            avg_perp = perplexities.mean().item()

            return avg_perp, avg_reps, unique_ngrams, all_outs

    def calc_perplexities(self, probs):
        """ Calculates the perplexity of a sentence given a 
            vector of probabilities for each word. """
        perps = (probs.log().sum(dim=1) * -1 / probs.size(1)).exp()
        return perps

    def get_probs(self, x, y):
        """ Given an input query x, returns a vector of the probability of 
            each word in the output query y. """
        softmax_fn = Softmax(dim=1)
        batches, max_length = y.size()
        probs = torch.ones_like(y).float().to(self.device)
        
        for i in range(max_length):
            # prepare input as concatenation of input query, and output query up to index i
            input = torch.cat((x, y[:,0:i]), dim=1)
            output = self.gen_model(input)
            
            # calculate and extract probabilities
            logits = output[0][:, -1, :]
            vocab_dist = softmax_fn(logits)
            inds = y[:,i].unsqueeze(1)

            # clamp values to ensure rounding errors do not result in invalid output
            probs[:,i] = vocab_dist.gather(1, inds).squeeze(1).clamp(max = 1 - 1e-5)
        return probs

    def generate(self, sent, max_length, num_beams):
        """ Given an input sentence, generate an extension of the sentence, 
            optionally using beamsearch. """
        with torch.no_grad():
            x = self.tokenizer.encode(sent, return_tensors='pt').to(self.device)
            if(num_beams==0):
                output = self.gen_model.generate(x, max_length=max_length)
            else:
                beams = self.gen_model.generate(x, max_length=max_length, num_beams=num_beams, early_stopping=True)
                output = beams[0]

            # remove EOS token
            output = DialogDataset.filter_token(output.squeeze(), self.tokenizer.eos_token_id)

            return self.tokenizer.decode(output)
    


