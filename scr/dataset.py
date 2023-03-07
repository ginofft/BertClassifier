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
    labelSet : List[str]
        A list whose elements are our classes
    texts : List[str]
        A list whose elements is the sentence
    labels : List[List[int]]
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
        """Passing list of [text, label] and the label set into our dataset

        Parameters
        ----------
        listData : List[str, List[str]]
            A list whose elements are in the form: [text, [labels]]
        labelSet : List[str]
            A list whose element are the labels
        tokenizer : PretrainedTokenizer
            Huggingface's tokenizer
        """
        
        self.labelSet = labelSet
        self.texts, self.labels = self._processList(listData)
        self.tokenizer = tokenizer

        self.tokenized_dataset = self.tokenizer(self.texts, 
                        padding = False, 
                        truncation = False)
                        
    def _processList(self, listData):
        for sentence, labels_instance in listData:
            self.texts.append(sentence) 
            
            labels = []
            for label in labels_instance:
                labels.append(self.labelSet.index(label))
            self.labels.append(labels)
    
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
            "class": self.labelSet[self.labels[idx]],
            "input_ids": self.tokenized_dataset['input_ids'][idx],
            "attention_mask": self.tokenized_dataset['attention_mask'][idx],
            "label": self.labels[idx]
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
        labels = list()
        max_size = max([len(ex['input_ids']) for ex in batch])
        for item in batch:
            batch_input += [self.pad_seq(item['input_ids'], max_size, self.pad_token_id)]
            batch_attention_mask += [self.pad_seq(item['attention_mask'], max_size, 0)]
            labels.append(item['label'])
        return {
            'input_ids': torch.tensor(batch_input, dtype = torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype = torch.long),
            'labels': torch.tensor(labels, dtype = torch.long),
        }

def get_data_from_json(path):
    """Return a dictionary from json file, where each keys contain a list.
    """
    with open(path, 'r') as f:
        dataDict = json.load(f)
    print('The keys found in this json are: ', dataDict.keys())
    return dataDict
    
def get_label_set(*lists):
    """Return a List[str] of label, used to map 'label' into indices.
    """
    labelSet = []
    for currentList in lists:
        for data in currentList:
            if data[1] not in labelSet:
                labelSet.append(data[1])
    return labelSet