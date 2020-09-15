import transformers
import numpy as np

max_len = 128
batch_size = 16
EPOCHS = 10

ROBERTA_PATH = 'xlm-roberta-large'
Training_file = 'train.csv'
tokenizer = transformers.AutoTokenizer.from_pretrained(ROBERTA_PATH, do_lower_case = True)

DEVICE = 'cuda'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
