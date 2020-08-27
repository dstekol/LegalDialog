import argparse
from DialogGenerator import DialogGenerator

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest="input", help="Input sentence")
    parser.add_argument('--pretrained-gen', type=str, dest="pretrained_gen", help="Filepath to trained generator")
    parser.add_argument('--beams', type=int, default=0, dest="num_beams", help="Number of beams to use if beamsearch desired")
    parser.add_argument('--max-length', type=int, default=50, dest="max_length", help="Maximum output length")
    
    args = parser.parse_args()

    generator = DialogGenerator(args.pretrained_gen, None)
    output = generator.generate(args.input, args.max_length, args.num_beams)

    print(output)
