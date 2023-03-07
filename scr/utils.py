import torch
from pathlib import Path
import json

def save_checkpoint(state, path:Path, filename='latest.pth.tar'):
    out_path = path/filename
    torch.save(state, out_path)

def load_checkpoint(path, model, optimizer = None):
    state = torch.load(path)
    epoch = state['epoch']
    train_loss = state['train_loss']
    val_loss = state['val_loss']

    model.load_state_dict(state['model'])
    if optimizer !=  None:
        optimizer.load_state_dict(state['optimizer'])
        
    print("=> loaded checkpoint '{}' (epoch {})".format(True, epoch))
    print("Checkpoint's train loss is: {:.4f}".format(train_loss))
    print("Checkpoint's validation loss is: {:.4f}".format(val_loss))
    return epoch, train_loss, val_loss

def predict_class(texts, model, tokenizer, labelSet):
    device = model.bert.device
    tokenized_texts = tokenizer(texts, padding = True, truncation = False, return_tensors = 'pt')
    input_ids = tokenized_texts['input_ids'].to(device)
    masks = tokenized_texts['attention_mask'].to(device)

    embedding = model(input_ids, masks)
    preds = torch.argmax(embedding,dim=1)
    results = []
    for pred in preds:
        results.append(labelSet[pred])
    return results

def read_MixSNIPs_file(filePath):
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

def get_label_set(*lists):
    """Return a List[str] of label, used to map 'label' into indices.
    """
    labelSet = []
    for currentList in lists:
        for sentence, labels in currentList:
            for label in labels:
                if label not in labelSet:
                    labelSet.append(label)
    return labelSet