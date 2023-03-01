import argparse
from pathlib import Path
import torch

from scr.dataset import SentenceLabelDataset, get_data_from_json, get_label_set
from scr.train import train, inference
from scr.models import BertMLPClassifier
from scr.utils import save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser(description='Bert-Sentence-Classifier')

parser.add_argument('--batch_size', type=int, default = 16, help='batch size')
parser.add_argument('--nEpochs', type = int, default=50, help='No. epochs')
parser.add_argument('--mode', type=str, default='train', 
                    help='training mode or inference mode',
                    choices=['train', 'test'],
                    required=True)
parser.add_argument('--bertVariation', type=str, default='distilbert-base-uncased',
                    help='pretrained Bert checkpoint on HuggingFace')

parser.add_argument('--saveEvery', type = int, default = 10, 
                    help='no. epoch before a save is created')

parser.add_argument('--datasetPath', type = str, default='',
                    help='Path to dataset json')

parser.add_argument('--savePath', type = str, default = '',
                    help = 'Path to save checkpoint to')
parser.add_argument('--loadPath', type = str, default = '',
                    help = 'Path to load checkpoint from')

if __name__ == "__main__":
    opt = parser.parse_args()
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('No GPU found, running on CPU!!')
    
    model = BertMLPClassifier(checkpoint = opt.bertVariation)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    dataDict = get_data_from_json(opt.datasetPath)
    trainList = dataDict['train']
    valList = dataDict['val']
    testList = dataDict['test']
    labelSet = get_label_set(trainList, valList, testList)

    trainSet = SentenceLabelDataset(trainList, labelSet, tokenizer = opt.bertVariation)
    valSet = SentenceLabelDataset(valList, labelSet, tokenizer = opt.bertVariation)
    testSet = SentenceLabelDataset(testList, labelSet, tokenizer = opt.bertVariation)

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
            epoch_train_loss, train_metrics = train(trainSet, model, criterion, 
                                    optimizer, device, opt.batch_size, epoch)
            epoch_val_loss, val_metrics = inference(valSet, model, criterion, device)

            print('Epoch {} completed: \nTrain loss: {:.4f} - Train Metrics: {}\nValidation loss: {:.4f} - Validation Metrics {}'.format(
                epoch, epoch_train_loss, train_metrics, epoch_val_loss, val_metrics))
            
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

        print('---------------------------Running Inferenece---------------------------')
        test_loss, test_metrics = inference(testSet, model, criterion, device)
        print('Test loss: {:.4f} - Test Metrics: {}'.format(test_loss, test_metrics))
