import torch
from pathlib import Path

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

def predict_class(*texts, model, tokenizer, labelSet):
    device = model.bert.device
    txt = [text for text in texts]
    tokenized_texts = tokenizer(txt, padding = False, truncation = False, return_tensors = 'pt')
    input_ids = tokenized_texts['input_ids'].to(device)
    masks = tokenized_texts['attention_mask'].to(device)

    embedding = model(input_ids, masks)
    preds = torch.argmax(dim=1)
    results = []
    for pred in preds:
        results.append(labelSet[pred])
    return results