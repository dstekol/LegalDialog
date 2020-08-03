import argparse
from DialogGenerator import DialogGenerator
from DialogDataset import DialogDataset

from tqdm import tqdm
import pickle
import torch

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data-path', type=str, dest="test_data_path", help="Filepath to preprocessed test data file")
    parser.add_argument('--pretrained-gen', type=str, dest="pretrained_gen", help="Filepath to pretrained generator")
    parser.add_argument('--batch-size', type=int, default=1, dest="batch_size", help='Batch size')
    parser.add_argument('--save-path', type=str, default=None, dest="save_path", help='Path where sample outputs should be saved')
    parser.add_argument('--max-length', type=int, default=80, dest="max_length", help='Maximum length of generated sentences')
    
    args = parser.parse_args()

    generator = DialogGenerator(args.pretrained_gen, None)
    test_dataset = DialogDataset(args.test_data_path, generator.tokenizer.eos_token_id)
    test_loader = test_dataset.get_loader(args.batch_size, shuffle=False)

    avg_perplexity, avg_repetition, all_outs = generator.eval(test_loader, args.max_length)
    print("Average perplexity:\n" + str(avg_perplexity))
    print("Average perplexity:\n" + str(avg_repetition))
    pickle.dump(all_outs, open(args.save_path, "wb"))

