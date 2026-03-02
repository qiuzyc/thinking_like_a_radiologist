from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import jsonlines
import torch

class InterleavedDataset(Dataset):
    def __init__(self, filepath):
        self.tokenized_data = []
        
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                input_ids = torch.tensor(obj["tokens"], dtype=torch.long)
                
                if len(input_ids) < 5120:
                    attention_mask = torch.ones_like(input_ids)  # All tokens are valid
               
                    labels = input_ids.clone()

                    # Find the position of 8710 and set the tokens before it to -100.
                    sep_pos = (input_ids == 8710).nonzero(as_tuple=True)[0]
                    assert len(sep_pos) == 1  
                    labels[:sep_pos[0] + 1] = -100  
                    
                    self.tokenized_data.append({
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': labels
                    })
               
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def collate_fn(batch):
    """
    Simple collate function that pads the sequences in the batch.
    All other processing is handled in the dataset classes.
    """
    # Extract each tensor type from the batch
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad each tensor type
    return {
        'input_ids': pad_sequence(input_ids, batch_first=True, padding_value=1),
        'attention_mask': pad_sequence(attention_masks, batch_first=True, padding_value=0),
        'labels': pad_sequence(labels, batch_first=True, padding_value=1)
    }
