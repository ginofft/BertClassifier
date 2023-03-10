import argparse
from pathlib import Path
from typing import List
import torch
from transformers import BertTokenizer

from scr.dataset import SentenceLabelDataset
from scr.train import train, inference
from scr.models import BertMLPClassifier
from scr.utils import save_checkpoint, load_checkpoint, read_CLINC150_file, read_MixSNIPs_file, get_label_set
from scr.evaluate import MultiLabelEvaluator

parser = argparse.ArgumentParser(description='Bert-Sentence-Classifier')

#model, optimizer and criterion parameters
parser.add_argument('--lr', type = float, default=1e-6, help='learning rate')
parser.add_argument('--bertVariation', type=str, default='bert-base-uncased',
                    help='pretrained Bert checkpoint on HuggingFace')

#training parameters
parser.add_argument('--mode', type=str, default='train', 
                    help='training mode or inference mode',
                    choices=['train', 'inference'],
                    required=True)
parser.add_argument('--nEpochs', type = int, default=50, help='No. epochs')
parser.add_argument('--saveEvery', type = int, default = 10, 
                    help='no. epoch before a save is created')
parser.add_argument('--metrics', type = List[str], nargs = '+', 
                    default=['Accuracy'], choices=['Accuracy', 'Recall', 'Precision', 'F1'],
                    help='The evaluation metric for multi-label classification')

#Data paremters
parser.add_argument('--dataFormat', type=str, default = 'MixSNIPs', 
                    help="Input data format, currently support CLINC150 and MixSNIPs", 
                    choices=['CLINC150', 'MixSNIPs'])
parser.add_argument('--batch_size', type=int, default = 16, help='batch size')
parser.add_argument('--datasetPath', type = str, default='',
                    help='Path to dataset json')

#check point parameters
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

    if opt.dataFormat.lower() == 'clinc150':
        dataDict = read_CLINC150_file(opt.datasetPath)
        trainList = dataDict['train'] + dataDict['oos_train']
        valList = dataDict['val'] + dataDict['oos_val']
        testList = dataDict['test'] + dataDict['oos_test']

    
    if opt.dataFormat.lower() == 'mixsnips':
        trainPath = opt.datasetPath + '/train.txt'
        valPath = opt.datasetPath + '/dev.txt'
        testPath = opt.datasetPath + '/test.txt'

        trainList = read_MixSNIPs_file(trainPath)
        valList = read_MixSNIPs_file(valPath)
        testList = read_MixSNIPs_file(testPath)
    
    labelSet = get_label_set(trainList, valList, testList)
    trainSet = SentenceLabelDataset(trainList, labelSet, tokenizer = BertTokenizer.from_pretrained(opt.bertVariation))
    valSet = SentenceLabelDataset(valList, labelSet, tokenizer = BertTokenizer.from_pretrained(opt.bertVariation))
    testSet = SentenceLabelDataset(testList, labelSet, tokenizer = BertTokenizer.from_pretrained(opt.bertVariation))

    model = BertMLPClassifier(checkpoint = opt.bertVariation, nClasses=trainSet.nClasses)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    metrics = opt.metrics
    evaluator = MultiLabelEvaluator()
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
            # epoch_train_loss, train_metrics = train(trainSet, model, criterion, 
            #                         optimizer, device, opt.batch_size, epoch)
            # epoch_val_loss, val_metrics = inference(valSet, model, criterion, device)

            # print('Epoch {} completed: \nTrain loss: {:.4f} - Train Metrics: {}\nValidation loss: {:.4f} - Validation Metrics {}'.format(
            #     epoch, epoch_train_loss, train_metrics, epoch_val_loss, val_metrics))
            
            epoch_train_loss, train_metrics = train(trainSet, model, criterion, optimizer,
                                     evaluator, metrics, 
                                     device, opt.batch_size, epoch)
            epoch_val_loss, val_metrics = inference(valSet, model, criterion, 
                                       evaluator, metrics,
                                       device, opt.batch_size)
            
            print('Epoch {} completed: \nTrain loss: {:.4f} \nValidation loss: {:.4f}'.format(
                epoch, epoch_train_loss, epoch_val_loss))
            print('Train metrics: {} \nValidation metrics: {}'.format(
                train_metrics, val_metrics))
            
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
        test_loss, test_metrics = inference(testSet, model, criterion, 
                                            evaluator, metrics,
                                            device, opt.batch_size)
        print('Test loss: {:.4f}'.format(test_loss))
        print('Test metrics: {}'.format(test_metrics))
