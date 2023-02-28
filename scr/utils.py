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
