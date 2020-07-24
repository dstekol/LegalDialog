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
    parser.add_argument('--batch-size', type=int, default=10, dest="batch_size", help='Batch size')
    args = parser.parse_args()
    generator = DialogGenerator(args.pretrained_gen, None)
    test_dataset = DialogDataset(args.test_data_path, generator.tokenizer.eos_token_id)
    test_loader = test_dataset.get_loader(args.batch_size)

    #avg_perplexity = generator.eval(test_loader)
    #print(avg_perplexity)
    reps = []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(generator.device)
            out = generator.gen_model.generate(x, max_length=80, early_stopping=True)
            out2 = out[:,x.size(1):]
            outlist = out2[0].tolist()
            outset = set(outlist)
            reps.append(len(outlist) - len(outset))
        print("avg reps")
        print(sum(reps) / len(reps))
            #examples.append({"input": generator.tokenizer.decode(x.squeeze(0)), "output": generator.tokenizer.decode(out2.squeeze(0))})
        #pickle.dump(examples, open("unt_examples.pkl", "wb"))

