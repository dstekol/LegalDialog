from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from train_utils import create_optimizer, create_scheduler
import pickle
from DialogDataset import DialogDataset
import torch
from tqdm import tqdm

class DialogDiscriminator:
    """ Discriminator used for adversarial weighting. """
    def __init__(self, model_type, trained_path, save_path, opt_params):
        super(DialogDiscriminator, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)

        # instantiate pretrained model from filepath if given, otherwise instantiate default pretrained model
        self.disc_model = AutoModelForSequenceClassification.from_pretrained(model_type if trained_path is None else trained_path)
        
        # logging variables
        self.step = 0
        self.epoch = 0

        # switch to cuda if possible
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if(torch.cuda.device_count()>=2):
            self.device = torch.device("cuda:1")
        self.disc_model.to(self.device)

        # instantiates optimizer and scheduler with given configuration options
        self.optimizer = create_optimizer(self.disc_model, opt_params["weight_decay"], opt_params["lr"], opt_params["epsilon"])
        self.scheduler = create_scheduler(self.optimizer, opt_params["warmup_steps"], opt_params["total_steps"])
        
        # discriminator is only used in training
        self.disc_model.train()

        self.save_path = save_path

    def save_checkpoint(self):
        save_file = self.save_path + "epoch_" + str(self.epoch) + "_disc"
        self.disc_model.save_pretrained(save_file)

    def retokenize_batch(self, x, gen_out, true_out, from_tok):
        """ Prepares output from generator for input into discriminator.
            Specifically, accepts input query x, generated output y1, and true output y2, 
            retokenizes each vector, inserts necessary SEP and CLS tokens, and generates a 
            batch tensor."""
        retokenized_gen = []
        retokenized_true = []

        sep = torch.tensor([[self.tokenizer.sep_token_id]]).to(self.device)
        cls = torch.tensor([[self.tokenizer.cls_token_id]]).to(self.device)

        for x_old, gen_out_old, true_out_old in zip(x, gen_out, true_out):
            # retokenize each of the vectors
            x_new = torch.cat((cls, self.retokenize_vect(x_old, from_tok), sep), dim=1)
            gen_out_retok = torch.cat((self.retokenize_vect(gen_out_old, from_tok), sep), dim=1)
            true_out_retok = torch.cat((self.retokenize_vect(true_out_old, from_tok), sep), dim=1)

            # append (input query, generated output) tuple to retokenized list
            retokenized_gen.append((x_new, gen_out_retok))

            # append (input query, true output) tuple to retokenized list
            retokenized_true.append((x_new, true_out_retok))

        # append retokenized generated vectors and retokenized true vectors into single batch (for parallel processing)
        retokenized_list = retokenized_gen + retokenized_true
        X, Y = DialogDataset.collate_with_padding(retokenized_list, self.tokenizer.pad_token_id)
        return torch.cat((X,Y), dim=1).to(self.device)

    def retokenize_vect(self, vect, from_tok):
        """ Accepts a vector representing a sentence tokenized with a different tokenizer,
            and retokenizes it with the discriminator's tokenizer. """
        filtered = DialogDataset.filter_token(vect, from_tok.eos_token_id)
        return self.tokenizer.encode(from_tok.decode(filtered), add_special_tokens=False, return_tensors='pt').long().to(self.device)
        
    def weight_losses(self, x, true_out, gen_out, gen_losses, gen_tokenizer):
        """ Weights the generator's token losses according to their 'believability',
            as described in Emulating Legal Dialog with Adversarial Weighting paper.
            In other words, this is where the magic happens. """
        x, true_out, gen_out, gen_losses = x.to(self.device), true_out.to(self.device), gen_out.to(self.device), gen_losses.to(self.device)
        scores = torch.empty(gen_out.size(0)*2, gen_out.size(1)).to(self.device)
        disc_losses = torch.empty(0).to(self.device)
        
        for i in range(gen_out.size(1)):
            gen_vects = torch.cat((true_out[:,0:i], gen_out[:,i].unsqueeze(1)), dim=1)
            true_vects = true_out[:,0:i+1]

            # obtain a batch for input into the discriminator:
            # the first half of the batch contains the 'generated' inputs: each vector begins with the input query x, then the first i-1 ground-truth tokens, and ends with the token outputted by the generator at the i-th step
            # the second half of the batch contains 'true' inputs: each vector begins with the input query x, and ends with the ground-truth token sequence, up to the i-th token
            # these vectors are appended into a single batch so the believability scores for generated and ground-truth sequences can be computed simultaneously
            disc_input = self.retokenize_batch(x, gen_vects, true_vects, gen_tokenizer)
            
            # create label vector: ones for the first half of the batch (which contains the generated sequences),
            # and zeros for the second half of the batch (which contains the ground-truth sequences). 
            labels = torch.cat((
                torch.ones(gen_vects.size(0)), 
                torch.zeros(true_vects.size(0))), dim=0).long().to(self.device)
            
            # extract discriminator output
            disc_output = self.disc_model(disc_input, labels=labels)
            disc_loss = disc_output[0]
            logits = disc_output[1]
            disc_losses = torch.cat((disc_losses, disc_loss.unsqueeze(0)))
            scores[:,i] = logits[:,0] - logits[:,1]

        # compute believability
        gen_scores = scores[:gen_out.size(0),:]
        true_scores = scores[-gen_out.size(0):,:]
        weights = (true_scores - gen_scores).exp()

        # weight generator losses for each token by believability score of each token
        gen_loss = (gen_losses * weights).sum()
        return gen_loss, disc_losses.sum()
       

