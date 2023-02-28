from torch.utils.data import Dataset
from transformers import BertTokenizer

class SentenceLabelDataset(Dataset):
    """This class create a labeled dataset from a list - each element having two component: [text, label]

    '''
    Attributes
    ----------
    listData : list[str, str]
        A list whose elements consist of two component: text and label
    labelSet : 
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
