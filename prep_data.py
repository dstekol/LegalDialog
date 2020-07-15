from DialogDatasetPreprocessor import DialogDatasetPreprocessor
import argparse

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, dest="data_path", help="Filepath to raw data file")
    parser.add_argument('--train_path', type=str, dest="train_path", help="Where to save processed train data")
    parser.add_argument('--test_path', type=str, dest="test_path", help="Where to save processed test data")
    parser.add_argument('--split_ratio', type=float, dest="split_ratio", help="What portion of data should be used as training data")
    args = parser.parse_args()
    DialogDatasetPreprocessor.preprocess_data_file(args.data_path, 
                                                   args.train_path, 
                                                   args.test_path, 
                                                   args.split_ratio)
