import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Dict
from dataclasses import dataclass
import json

class SentenceLabelDataset(Dataset):
    """This class create a labeled dataset from a list - each element is: [text, [labels]]
    Note: This class is intended to be used with dynamic padding. As such, tokenized_dataset consist of sentence with various length

    Attributes
    ----------
    labelSet : Dict{str : int}
        A dictionary with key being the text label and value being the indices in classifier
    texts : List[str]
        A list whose elements is the sentence
    labels : List[List[str]]
        A list whose elements is a list of the classes associated with this the sentence
    tokenizer : BertTokenizer
        Bert pre-trained tokenizer, inherits from Huggingface's PretrainedTokenizer
    tokenized_dataset : List[Dict[str, List[int]]]
        tokenized dataset, return as a list of dictionary with three keys: input_ids - token's  idex, attention_mask - attention mask to feed into transformer, token_type_ids - not used for our purpose
    """

    def __init__(self, listData, labelSet, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
        """Passing list of [text, label] and the label set into our dataset

        Parameters
        ----------
        listData : List[Dict]
            A list whose element are Dictionaries - each with two keys: `dialog` and `label`
        labelSet : Dict
            A dictionary with keys being the classes, and value being the classifier's index
        tokenizer : PretrainedTokenizer
            Huggingface's tokenizer
        """
        
        self.labelSet = labelSet
        self.nClasses = len(labelSet)

        self.texts, self.labels = self._processList(listData)
        self.tokenizer = tokenizer

        self.tokenized_dataset = self.tokenizer(self.texts, 
                        padding = False, 
                        truncation = False)
                        
    def _processList(self, listData):
        temp_data = listData[0]
        if ('dialog' not in temp_data) or ('label' not in temp_data):
            raise ValueError(f"{listData} do not have the correct format")
        texts = []
        labels = []
        for data in listData:
            texts.append(data['dialog'])     
            labels.append(data['label'])
        return texts, labels
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        """Get item from dataset

        Parameters
        ----------
        idx : int
            Item index in this dataset
        
        Return
        ------
        A dictionary, whose keys are:
        - text : str
            the sentence in text form
        - class : str
            the class in text form
        - input_ids : List[int]
            tokenized sentenced
        - attention_mask : List[int]
            the attention mask
        - label : List[int]
            the class index in labelSet
        """
        return{
            "text": self.texts[idx],
            "class": self.labels[idx],
            "input_ids": self.tokenized_dataset['input_ids'][idx],
            "attention_mask": self.tokenized_dataset['attention_mask'][idx],
            "label": [self.labelSet[key] for key in self.labels[idx]]
        }
@dataclass
class SmartCollator():
    """ This class provide methods for dynamic padding

    Attributes
    ----------
    pad_token_id : int
        the token id of [PAD] token, which are usually set by the tokenizer
    
    Methods
    -------
    collate_dynamic_paddding(batch)
        take in a batch of tokenized sentences with various length. Then pad them all to the longest sentence in the pad
    """
    pad_token_id: int
    nClasses : int
    def pad_seq(self, seq:List[int], max_batch_len: int, pad_value:int)->List[int]:
        return seq + (max_batch_len - len(seq)) * [pad_value]

    def collate_dynamic_padding(self, batch) -> Dict[str, torch.Tensor]:
        """This function padd all sentence to the longest sentence in the batch - used to do dynamic padding

        Parameters
        ----------
        batch : List[Dict]
            A list of dictionaries, where each dict must contains three keys: input_ids, attention_mask, label
        
        Return
        ------
        A dictionary whose keys are:
            - input_ids : 2D tensor of [batch_size, max_batch_length]
            - attention_mask : 2D tensor of [batch_size, max_batch_length]
            - labels : 1D tensor of labels
        """

        batch_input = list()
        batch_attention_mask = list()
        batch_targets = list()
        max_size = max([len(ex['input_ids']) for ex in batch])

        batch_targets = torch.zeros(len(batch), self.nClasses)
        for idx, item in enumerate(batch):
            batch_input += [self.pad_seq(item['input_ids'], max_size, self.pad_token_id)]
            batch_attention_mask += [self.pad_seq(item['attention_mask'], max_size, 0)]
            
            for label in item['label']:
              batch_targets[idx, label] = 1
        return {
            'input_ids': torch.tensor(batch_input, dtype = torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype = torch.long),
            'labels' : batch_targets
        }
