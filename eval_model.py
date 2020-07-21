import argparse
from DialogGenerator import DialogGenerator
from DialogDataset import DialogDataset

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data-path', type=str, dest="test_data_path", help="Filepath to preprocessed test data file")
    parser.add_argument('--pretrained-gen', type=str, dest="pretrained_gen", help="Filepath to pretrained generator")
    parser.add_argument('--batch-size', type=int, default=10, dest="batch_size", help='Batch size')
    args = parser.parse_args()
    generator = DialogGenerator(args.pretrained_gen, None)
    test_dataset = DialogDataset(args.test_data_path, generator.tokenizer.eos_token_id)
    test_loader = test_dataset.get_loader(args.batch_size)
    avg_perplexity = generator.eval(test_loader)
    print(avg_perplexity)

