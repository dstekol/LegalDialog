import argparse
from DialogGenerator import DialogGenerator

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest="input", help="Input sentence")
    parser.add_argument('--pretrained-gen', type=str, dest="pretrained_gen", help="Filepath to pretrained generator")
    args = parser.parse_args()
    generator = DialogGenerator(args.pretrained_gen, None, None)
    output = generator.generate(args.input)
    print(output)
