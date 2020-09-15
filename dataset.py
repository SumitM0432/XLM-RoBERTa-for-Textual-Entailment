import config
import torch
from torch.utils.data import DataLoader, Dataset

class roberta_dataset(Dataset):
    
    def __init__(self, combined_thesis, target, tokenizer, max_len):
        self.combined_thesis = combined_thesis
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.combined_thesis)
    
    def __getitem__(self, item):
        # this combined will just take one premise and hypothesis combination from the whole list of list
        # as we are using the batch_encode_plus and we need p and h encoded both combined

        combined = [self.combined_thesis[item]]
        target = self.target[item]
        encoding_input = self.tokenizer.batch_encode_plus(
        combined,
        add_special_tokens = True,
        pad_to_max_length = True,
        max_length = self.max_len,
        return_tensors = 'pt',
        truncation = True
        )
        
        return {
            'combined_thesis':  combined[0],
            'input_ids': encoding_input['input_ids'].flatten(),
            'attention_mask': encoding_input['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype = torch.long)
        }
    

# DataLoader
def create_data_loader(df_, tokenizer, max_len, batch_size):
    ds = roberta_dataset(
            combined_thesis = df_.combined_thesis.to_numpy(),
            target = df_.label.to_numpy(),
            tokenizer = tokenizer,
            max_len = max_len
        )
    return DataLoader(
        ds,
        batch_size = batch_size,
        )