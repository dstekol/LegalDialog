import argparse
from DialogGenerator import DialogGenerator

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest="input", help="Input sentence")
    parser.add_argument('--pretrained-gen', type=str, dest="pretrained_gen", help="Filepath to pretrained generator")
    parser.add_argument('--beams', type=int, default=0, dest="num_beams", help="Number of beams to use if beamsearch desired")
    parser.add_argument('--max-length', type=int, default=50, dest="max_length", help="Maximum output length")
    args = parser.parse_args()
    generator = DialogGenerator(args.pretrained_gen, None)
    #sents = ["What does the plaintiff wish to prove?",
    #         "It seems to me that full amount had already been paid.",
    #         "Are you familiar with the aforementioned regulations?",
    #         "Could you clarify your previous statement regarding the statute of limitations",
    #         "In this case, I think it would be sensible to make an exception.",
    #         "I am not sure I understand the premise of your assumption.",
    #         "That is correct.",
    #         "Do you consider the proposals legal permissible within the framework of the Constitution?"]
    #for sent in sents:
    #    #output = generator.generate(args.input, args.max_length, args.num_beams)
    #    output = generator.generate(sent, args.max_length, args.num_beams)
    #    #print(sent)
    #    print("___________________________________________")
    #    print(output)
    #    #print("___________________________________________")


    output = generator.generate(args.input, args.max_length, args.num_beams)
    print(output)
