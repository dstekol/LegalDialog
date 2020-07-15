import argparse
from DialogGenerator import DialogGenerator
from DialogDataset import DialogDataset

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data-path', type=str, dest="test_data_path", help="Filepath to preprocessed test data file")
    parser.add_argument('--pretrained-gen', type=str, dest="pretrained_gen", help="Filepath to pretrained generator")
    args = parser.parse_args()
    generator = DialogGenerator(args.pretrained_gen, None, None)
    test_dataset = DialogDataset(args.test_data_path, generator.tokenizer, generator.tokenizer)
    test_loader = test_dataset.get_loader(100)
    avg_perplexity = generator.eval(test_loader)
    print(avg_perplexity)

