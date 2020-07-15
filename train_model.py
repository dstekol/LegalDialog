import torch
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import argparse
from DialogGenerator import DialogGenerator
from DialogDataset import DialogDataset
import os

#args: decoder (none, gru, gpt2), epochs, data_path, pretrained encoder

#def train_models_adversarial():
#    pass

#def train_models_teacher_forcing():
#    generator = DialogGenerator()

#def generate():
#    pass

#--test_path test_data.pkl --split_ratio 0.8

#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#tokenizer.encode("Justice Stevens, and may it please the Court: These two contract cases concern whether the Government is liable in money damages under the Contract Disputes Act and section 110 of the Indian Self-Determination Act when the Secretary fails to fully pay a contract price for the --")

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, dest="epochs", help='Number of epochs to run')
    parser.add_argument('--batch-size', type=int, default=10, dest="batch_size", help='Batch size')
    parser.add_argument('--adversarial', default=False, action='store_true')

    parser.add_argument('--gen_weight_decay', type=float, default=.001, dest="gen_weight_decay", help='Weight decay for the generator\'s training scheduler')
    parser.add_argument('--forcing-ratio', type=float, default=1, dest="forcing_ratio", help='How often to use teacher forcing')
    parser.add_argument('--gen_lr', type=float, default=.0001, dest="gen_lr", help='Learning rate for generator')
    parser.add_argument('--gen_epsilon', type=float, default=.001, dest="gen_epsilon", help='Epsilon parameter for generator optimizer')
    parser.add_argument('--gen_warmup_steps', type=int, default=1000, dest="gen_warmup_steps", help='Number of warmup steps for training generator')

    parser.add_argument('--train-data-path', type=str, dest="train_data_path", help="Filepath to preprocessed data")
    parser.add_argument('--save-folder', type=str, dest="save_folder", help="Filepath to folder where checkpoints should be saved")
    parser.add_argument('--pretrained-gen', type=str, default=None, dest="pretrained_gen", help="Filepath to pretrained generator")
    parser.add_argument('--pretrained-gen-meta', type=str, default=None, dest="pretrained_gen_meta", help="Filepath to pretrained generator metadata")
    parser.add_argument('--pretrained-disc', type=str, default=None, dest="pretrained_disc", help="Filepath to pretrained discriminator")

    args = parser.parse_args()

    if(args.adversarial):
        pass
    else:
        if(not os.path.isdir(args.save_folder)):
            os.mkdir(args.save_folder)
        if(args.save_folder[-1]!='/'):
            args.save_folder += '/'
        generator = DialogGenerator(args.pretrained_gen, args.pretrained_gen_meta, args.save_folder)
        train_dataset = DialogDataset(args.train_data_path, generator.tokenizer, generator.tokenizer)
        train_loader = train_dataset.get_loader(args.batch_size)
        optimizer = generator.create_optimizer(args.gen_weight_decay, args.gen_lr, args.gen_epsilon)
        scheduler = generator.create_scheduler(optimizer, args.gen_warmup_steps, int(len(train_dataset) / args.batch_size))
        generator.train_traditional(train_loader, args.epochs, args.forcing_ratio, optimizer, scheduler)
