import torch
import argparse
from DialogGenerator import DialogGenerator
from DialogDataset import DialogDataset
from DialogDiscriminator import DialogDiscriminator
import os

def prep_folder(args):
    if(args.save_folder[-1]!='/'):
        args.save_folder += '/'
    if(not os.path.isdir(args.save_folder)):
            os.mkdir(args.save_folder)

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3, dest="epochs", help='Number of epochs to run')
    parser.add_argument('--batch-size', type=int, default=50, dest="batch_size", help='Batch size')
    parser.add_argument('--max-out-length', type=int, default=128, dest="max_out_length", help='Maximum output length (outputs truncated if longer)')
    parser.add_argument('--adversarial-model', type=str, default=None, dest="adv_model", help='Type of adversarial model to use. Will use traditional teacher forcing if None.')
    parser.add_argument('--train-disc-only-steps', type=int, default=0, dest="train_disc_only_steps", help='Number of steps for which to train discriminator only (without updating generator)')

    parser.add_argument('--gen_weight_decay', type=float, default=0, dest="gen_weight_decay", help='Weight decay for the generator\'s training scheduler')
    parser.add_argument('--gen_lr', type=float, default=2e-5, dest="gen_lr", help='Learning rate for generator')
    parser.add_argument('--gen_epsilon', type=float, default=1e-8, dest="gen_epsilon", help='Epsilon parameter for generator optimizer')
    parser.add_argument('--gen_warmup_steps', type=int, default=0, dest="gen_warmup_steps", help='Number of warmup steps for training generator')

    parser.add_argument('--disc_weight_decay', type=float, default=0, dest="disc_weight_decay", help='Weight decay for the discriminator\'s training scheduler')
    parser.add_argument('--disc_lr', type=float, default=2e-5, dest="disc_lr", help='Learning rate for discriminator')
    parser.add_argument('--disc_epsilon', type=float, default=1e-8, dest="disc_epsilon", help='Epsilon parameter for discriminator optimizer')
    parser.add_argument('--disc_warmup_steps', type=int, default=0, dest="disc_warmup_steps", help='Number of warmup steps for training discriminator')

    parser.add_argument('--train-data-path', type=str, dest="train_data_path", help="Filepath to preprocessed data")
    parser.add_argument('--save-folder', type=str, dest="save_folder", help="Filepath to folder where checkpoints should be saved")
    parser.add_argument('--pretrained-gen', type=str, default=None, dest="pretrained_gen", help="Filepath to pretrained generator")
    parser.add_argument('--pretrained-disc', type=str, default=None, dest="pretrained_disc", help="Filepath to pretrained discriminator")

    args = parser.parse_args()

    assert args.train_data_path is not None
    assert args.save_folder is not None

    prep_folder(args)
    
    
    generator = DialogGenerator(args.pretrained_gen, args.save_folder)

    train_dataset = DialogDataset(args.train_data_path, generator.tokenizer.eos_token_id)
    train_loader = train_dataset.get_loader(args.batch_size, shuffle=True)

    gen_opt_params = {"weight_decay": args.gen_weight_decay, 
                      "lr": args.gen_lr, 
                      "warmup_steps": args.gen_warmup_steps,
                      "epsilon": args.gen_epsilon,
                      "total_steps": int(len(train_dataset) / args.batch_size) * args.epochs }
    generator.set_optimizer(gen_opt_params)

    if(args.adv_model is not None):
        disc_opt_params = {"weight_decay": args.disc_weight_decay, 
                           "lr": args.disc_lr, 
                           "warmup_steps": args.disc_warmup_steps,
                            "epsilon": args.disc_epsilon,
                           "total_steps": int(len(train_dataset) / args.batch_size) * args.epochs }
        discriminator = DialogDiscriminator(args.adv_model, args.pretrained_disc, args.save_folder, disc_opt_params)
        
        generator.train_adversarial(train_loader, args.epochs, args.max_out_length, discriminator, args.train_disc_only_steps)
    else:
        generator.train_traditional(train_loader, args.epochs, args.max_out_length)
