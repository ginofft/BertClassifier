import argparse
from pathlib import Path
import torch

from scr.dataset import SentenceLabelDataset
from scr.train import train, inference
from scr.models import BertMLPClassifier
from scr.utils import save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser(description='Bert-Sentence-Classifier')

if __name__ == "__main__":
    opt = parser.parse_args()
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('No GPU found, running on CPU!!')
    
    model = BertMLPClassifier()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    if opt.mode.lower() == 'train':
        startEpoch = 0
        val_loss = float('inf')
        train_loss = float('inf')

        if opt.loadPath:
            startEpoch, train_loss, val_loss = load_checkpoint(
                                                Path(opt.loadPath),
                                                model,
                                                optimizer)
        
        for epoch in range(startEpoch+1, opt.nEpochs+1):
            epoch_train_loss = train(trainSet, model, criterion, 
                                    optimizer, device, opt.batch_size, epoch)
            if opt.validationPath:
                epoch_val_loss = inference(valSet, model, criterion, device)
            else:
                epoch_val_loss = float('inf')
            
            if (epoch_val_loss < val_loss):
                val_loss = epoch_val_loss
                save_checkpoint({
                    'epoch' : epoch,
                    'train_loss' : epoch_train_loss,
                    'val_loss' : epoch_val_loss,
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, Path(opt.savePath), 'best.pth.tar')
            if (epoch % opt.saveEvery) == 0:
                save_checkpoint({
                    'epoch' : epoch,
                    'train_loss' : epoch_train_loss,
                    'val_loss' : epoch_val_loss,
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, Path(opt.savePath), 'epoch{}.pth.tar'.format(epoch))
    else:
        if opt.loadPath:
            startEpoch, train_loss, val_loss = load_checkpoint(
                                            Path(opt.loadPath),
                                            model,
                                            optimizer)
        else:
            raise Exception('Please point to a model using ---loadPath')
        
        