from DialogDatasetPreprocessor import DialogDatasetPreprocessor
import argparse

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, dest="data_path", help="Filepath to raw data file")
    parser.add_argument('--train-path', type=str, dest="train_path", help="Where to save processed train data")
    parser.add_argument('--validate-path', type=str, dest="validate_path", help="Where to save processed validation data")
    parser.add_argument('--test-path', type=str, dest="test_path", help="Where to save processed test data")
    parser.add_argument('--max-inp-length', type=int, default=128, dest="max_inp_length", help='Maximum input length (inputs truncated if longer)')
    args = parser.parse_args()
    DialogDatasetPreprocessor.preprocess_data_file(args.data_path, 
                                                   args.train_path, 
                                                   args.validate_path,
                                                   args.test_path,
                                                   args.max_inp_length)
