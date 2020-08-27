This project is based on the Emulating Legal Dialog with Adversarial Weighting paper by Daniel Stekol, written for the course Sequencing Legal DNA at ETH Zurich.


PREPROCESSING
This preprocessing code in project is designed for the Supreme Court Dialogs Corpus. 
It can be used with other corpuses, but would need additional preprocessing logic.
Before training, the prep_data.py file can be run to preprocess the raw corpus.
This will create a train-validation-test split and output 3 corresponding "pickle" files.
Command line options:
--data-path			Path to the raw data file (required).
--train-path		Where to save processed training data (required).
--validate-path		Where to save processed validation data (required).
--test-path option	Where to save processed testing data (required).
--max-inp-length	Maximum token length after which vectorized utterances will be truncated. This is helpful in circumventing memory constraints during training.

Example:
python prep_data.py --data-path "supreme_court_dialogs_corpus_v1.01/supreme.conversations.txt" --train-path "train_data.pkl" --validate-path "validate_data.pkl" --test-path "test_data.pkl"


TRAINING
Both generators and discriminators use the huggingface transformers package. 
Training is run using the train_model.py file, and can either be done via traditional teacher-forcing (simply do not specify --adversarial-model), or via adversarial weighting (specify --adversarial-model option with the name of a huggingface sequence classification model, such as 'bert-base-cased').
Command line options:
--epochs					Number of epochs to run training for.
--batch-size				Batch size.
--max-out-length			Maximum output length (outputs truncated if longer).
--adversarial-model			Type of adversarial model to use. Will use traditional teacher forcing if None. If specified, must be a huggingface transformer id such as 'bert-base-cased' or 'camember-base'
--train-disc-only-steps		Number of steps for which to train discriminator only (without updating generator). By default, the discriminator is not allocated any steps to train independently.

--gen_weight_decay			Weight decay for the generator's training scheduler
--gen_lr					Learning rate for generator
--gen_epsilon				Epsilon parameter for generator optimizer
--gen_warmup_steps			Number of warmup steps for training generator

--disc_weight_decay			Weight decay for the discriminator's training scheduler
--disc_lr					Learning rate for discriminator
--disc_epsilon				Epsilon parameter for discriminator optimizer
--disc_warmup_steps			Number of warmup steps for training discriminator

--train-data-path			Filepath to preprocessed training data (required).
--save-folder				Filepath to folder where checkpoints should be saved (required).
--pretrained-gen			Filepath to trained generator. If None, will instantiate a default pretrained GPT-2 generator.
--pretrained-disc			Filepath to trained discriminator. If None, will instantiate a default pretrained discriminator of type specified by --adversarial-model option.

Example:
python train_model.py  --train-data-path "train_data.pkl" --epochs 3 --batch-size 4 --save-folder adversarial_cam/ --max-out-length 30 --adversarial-model "camembert-base" 


AUTOMATIC EVALUATION
The eval_model.py file can be run to evaluate a model on a (preprocessed) dataset. 
This will output average perplexity and average intra-utterance token repetition, and save the generator outputs to a specified file.
Command line options:
--test-data-path		Filepath to preprocessed test (or validation) data file. (required)
--pretrained-gen		Filepath to trained generator. If not specified, will use default pretrained GPT-2 generator.
--save-path				Path where sample outputs should be saved. (required)
--max-length			Maximum length of generated token sequences, past which the sequences will be automatically truncated.

Example: 
python eval_model.py  --test-data-path "test_data.pkl" --pretrained-gen adversarial_bert/epoch_0_gen --max-length 30


MANUAL INPUT EXPERIMENTATION
In addition to performing automatic evaluation, it is possible to manually provide an input query to the generator and view the resulting output via the run_model.py file.
Command line options:
--input					Input sentence. (required)
--pretrained-gen		Filepath to trained generator. If not specified, will use default pretrained GPT-2 generator.
--beams					Number of beams to use if beamsearch desired
--max-length			Maximum length of generated token sequences, past which the sequences will be automatically truncated.

Example:
python run_model.py --input "What is the plaintiff's complaint?" --pretrained-gen "test_model/epoch_0_gen" --beams 3 --max-length 40
