import torch
from pathlib import Path
from typing import List, Tuple
import json

def save_checkpoint(state, path:Path, filename='latest.pth.tar'):
    out_path = path/filename
    torch.save(state, out_path)

def load_checkpoint(path, model, optimizer = None):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    state = torch.load(path, map_location=device)
    epoch = state['epoch']
    train_loss = state['train_loss']
    val_loss = state['val_loss']
    classifier_threshold = state['classifier_threshold']

    model.load_state_dict(state['model'])
    if 'label_set' in state.keys():
        model.labelSet = state['label_set']
    if optimizer !=  None:
        optimizer.load_state_dict(state['optimizer'])

    print("=> loaded checkpoint '{}' (epoch {})".format(True, epoch))
    print("Checkpoint's train loss is: {:.4f}".format(train_loss))
    print("Checkpoint's validation loss is: {:.4f}".format(val_loss))
    return epoch, train_loss, val_loss, classifier_threshold
    #return epoch, train_loss, val_loss

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(filename, data):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

def get_label_set(*lists):
    label_set = {}
    count = 0
    for current_lst in lists:
        for dict in current_lst:
            for labels in dict['label']:
                if labels not in label_set:
                    label_set[labels] = count 
                    count += 1
    return label_set

class Predictor():
    """This class contains the predictor given a classifier, its tokenizer and a labelSet

    Attributes 
    ----------
    model : torch.nn.Module
        The classifier
    tokenizer : transformers.PretrainedTokenizer
        The associated tokenizer
    labelSet : List[str] 
        The set of possible classes

    Methods
    -------
    get_class(texts) -> List[str]
        get the highest classes of each sentence in `texts`
    get_topk_classes(texts, k) -> List[List[str]]
        get the top `k` classes of each sentence in `texts`
    get_classes_at_percent(texts, p) -> List[List[str]]
        get the classes whose probability is higher than `p` (0<=p<1)
    """

    def __init__(self, model, tokenizer, labelSet, device = torch.device('cuda')):
        """
        Parameters
        ----------
        """
        self.model = model
        self.tokenizer = tokenizer
        self.labelSet = labelSet
        self.device = device

        if self.model.nClasses != len(labelSet):
            raise Exception("Number of model's classifier ({}) differ from length of label set {}"
                            .format(self.model.nClasses, len(labelSet)))
        
    def get_class(self, texts: List[str]):
        tokenized_texts = self.tokenizer(texts, padding = True, truncation = False, return_tensors = 'pt')
        input_ids = tokenized_texts['input_ids'].to(self.device)
        masks = tokenized_texts['attention_mask'].to(self.device)

        embedding = self.model(input_ids, masks)
        preds = torch.argmax(embedding,dim=1)
        results = []
        for pred in preds:
            results.append(self.labelSet[pred])
        return results
    
    def get_topk_classes(self, texts: List[str], k : int):        
        tokenized_texts = self.tokenizer(texts, padding = True, truncation = False, return_tensors = 'pt')
        input_ids = tokenized_texts['input_ids'].to(self.device)
        masks = tokenized_texts['attention_mask'].to(self.device)

        embedding = self.model(input_ids, masks)
        predsMatrix = torch.topk(embedding, dim =1, k = k).indices
        results = []
        for preds in predsMatrix:
            results.append([self.labelSet[pred] for pred in preds])
        return results
    
    def get_classes_at_percent(self, texts: List[str], p : float):
        tokenized_texts = self.tokenizer(texts, padding = True, truncation = False, return_tensors = 'pt')
        input_ids = tokenized_texts['input_ids'].to(self.device)
        masks = tokenized_texts['attention_mask'].to(self.device)

        embedding =  self.model(input_ids, masks)
        predsMatrix = torch.where(embedding > p, 1, 0)
        results = []
        for preds in predsMatrix:
            results.append([self.labelSet[pred] for pred in torch.nonzero(preds)])
        return results

    def get_classes_based_on_thresholds(self, texts : List[str], thresh : torch.Tensor):
        tokenized_texts = self.tokenizer(texts, padding = True, truncation = False, return_tensors = 'pt')
        input_ids = tokenized_texts['input_ids'].to(self.device)
        masks = tokenized_texts['attention_mask'].to(self.device)
        
        embedding = self.model(input_ids, masks)
        if embedding.size()[1] != thresh.size()[0]:
            raise Exception('The number of thresholds differ from the number of classes!!')
        
        predsMatrix = (embedding > thresh).int()
        results = []
        for preds in predsMatrix:
            results.append([self.labelSet[pred] for pred in torch.nonzero(preds)])
        return results  
    
#---------------------------------------------------Depricated--------------------------------------------------
def read_MixSNIPs_file(filePath) -> List[Tuple[str, List[str]]]:
    """read MixSNIPs file into a list of tuple, whose element is: sentence, [label]

    Return
    -------
    list : List[Tuple[str, List[str]]]
        A list of tuple, whose are: a sentence and a list of labels
    """
    texts, slots, intents = [], [], []
    text, slot = [], []
    with open(filePath, 'r', encoding="utf8") as fr:
        for line in fr.readlines():
            items = line.strip().split()
            if len(items) == 1:
                texts.append(text)
                slots.append(slot)
                if "/" not in items[0]:
                    intents.append(items)
                else:
                    new = items[0].split("/")
                    intents.append([new[1]])
                # clear buffer lists.
                text, slot = [], []
            elif len(items) == 2:
                text.append(items[0].strip())
                slot.append(items[1].strip())
    sentences = []
    labels = []
    space = ' '
    for i, txt in enumerate(texts):
        sentences.append(space.join(txt))
        
        intent_instance = intents[i][0]
        if '#' in intent_instance:
            label = intent_instance.split('#')
        else:
            label = [intent_instance]
        labels.append(label)
    return list(zip(sentences, labels))

def read_CLINC150_file(filePath):
    """Return a dictionary from json file, where each keys contain a list.
    """
    with open(filePath, 'r') as f:
        dataDict = json.load(f)
    print('The keys found in this json are: ', dataDict.keys())
    return dataDict

def turn_single_label_to_multilabels(*lists):
    for l in lists:
        for element in l:
            element[1] = [element[1]]
