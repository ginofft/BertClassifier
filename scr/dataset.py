import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Dict

class SentenceLabelDataset(Dataset):
    """This class create a labeled dataset from a list - each element having two component: [text, label]

    Note: This class is intended to be used with dynamic padding. As such, tokenized_dataset consist of sentence with various length
    '''
    Attributes
    ----------
    labelSet : List[str]
        A list whose elements are our classes
    texts : List[str]
        A list whose elements is the text
    labels : List[int]
        A list whose element is the index of that class inside labelSet
    tokenizer : BertTokenizer
        Bert pre-trained tokenizer, inherits from Huggingface's PretrainedTokenizer
    tokenized_dataset : List[Dict[str, List[int]]]
        tokenized dataset, return as a list of dictionary with three keys:
        - input_ids : token's  idex
        - attention_mask : attention mask to feed into transformer
        - token_type_ids : not used for our purpose
    """

    def __init__(self, listData, labelSet, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
        self.labelSet = labelSet
        self.texts, self.labels = self._processList(listData)
        self.tokenizer = tokenizer

        self.tokenized_dataset = self.tokenizer(self.texts, 
                        padding = False, 
                        truncation = False)
                        
    def _processList(self, listData):
        texts = []
        labels = []
        for data in listData:
            texts.append(data[0]) 
            labels.append(self.labelSet.index(data[1]))
        return texts, labels
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return{
            "text": self.texts[idx],
            "class": self.labelSet[self.labels[idx]],
            "input_ids": self.tokenized_dataset['input_ids'][idx],
            "attention_mask": self.tokenized_dataset['attention_mask'][idx],
            "label": self.labels[idx]
        }

#Tokenizer pad_id
_pad_token_id = BertTokenizer.pad_token_id 

def pad_seq(seq:List[int], max_batch_len: int, pad_value:int)->List[int]:
    return seq + (max_batch_len - len(seq)) * [pad_value]

def collate_dynamic_padding(batch) -> Dict[str, torch.Tensor]:
    """This function padd all sentence to the longest sentence in the batch - used to do dynamic padding
    """

    batch_input = list()
    batch_attention_mask = list()
    labels = list()
    max_size = max([len(ex['input_ids']) for ex in batch])
    for item in batch:
        batch_input += [pad_seq(item['input_ids'], max_size, _pad_token_id)]
        batch_attention_mask += [pad_seq(item['attention_mask'], max_size, 0)]
        labels.append(item['label'])
    return {
        'input_ids': torch.tensor(batch_input, dtype = torch.long),
        'attention_mask': torch.tensor(batch_attention_mask, dtype = torch.long),
        'labels': torch.tensor(labels, dtype = torch.long),
    }
