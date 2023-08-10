import argparse
from pathlib import Path
from typing import List
import torch
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer

from src.dataset import SentenceLabelDataset
from src.train import train, inference
from src.models import BertMLPClassifier, DistilBertMLPClassifier, RoBertaMLPClassifier
from src.utils import save_checkpoint, load_checkpoint, read_json, get_label_set
from src.evaluate import MultiLabelEvaluator

MODEL_MAPPING = {
    'Bert' : {'model' : BertMLPClassifier , 'tokenizer' : BertTokenizer.from_pretrained('bert-base-uncased')},
    'DistilBert' : {'model' : DistilBertMLPClassifier, 'tokenizer' : DistilBertTokenizer.from_pretrained('distilbert-base-uncased')},
    'RoBerta' : {'model' : RoBertaMLPClassifier, 'tokenizer' : RobertaTokenizer.from_pretrained('roberta-base')}
}

parser = argparse.ArgumentParser(description='Sentence-Classifier')

#model, optimizer and criterion parameters
parser.add_argument('--lr', type = float, default=1e-6, help='learning rate')
parser.add_argument('--encoder', type=str, default='Bert',
                    help='which text encoder to use',
                    choices= ['Bert', 'DistilBert', 'RoBerta'])
#training parameters
parser.add_argument('--mode', type=str, default='train', 
                    help='training mode or inference mode',
                    choices=['train', 'inference'],
                    required=True)
parser.add_argument('--nEpochs', type = int, default=50, help='No. epochs')
parser.add_argument('--saveEvery', type = int, default = 10, 
                    help='no. epoch before a save is created')
parser.add_argument('--metrics', nargs = '+', 
                    default=['macro f1'], 
                    choices=['macro accuracy',
                             'micro accuracy',
                             'macro precision',
                             'micro precision',
                             'macro recall',
                             'micro recall', 
                             'macro f1',
                             'micro f1'],
                    help='The evaluation metric for multi-label classification')

#Data paremters
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

    if opt.datasetPath == '':
        raise Exception('Please provide a path to the dataset')
    else:
        trainList = read_json(opt.datasetPath + '/train.json')
        valList = read_json(opt.datasetPath + '/val.json')
        testList = read_json(opt.datasetPath + '/test.json')
    
    labelSet = get_label_set(trainList, valList)

    tokenizer = MODEL_MAPPING[opt.encoder]['tokenizer']

    trainSet = SentenceLabelDataset(trainList, labelSet, tokenizer = tokenizer)
    valSet = SentenceLabelDataset(valList, labelSet, tokenizer = tokenizer)
    testSet = SentenceLabelDataset(testList, labelSet, tokenizer = tokenizer)

    model = MODEL_MAPPING[opt.encoder]['model']
    model = model(labelSet = labelSet)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = opt.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    metrics = opt.metrics
    evaluator = MultiLabelEvaluator()
    if opt.mode.lower() == 'train':
        print('---------------------------Training---------------------------')
        startEpoch = 0
        val_loss = float('inf')
        train_loss = float('inf')
        
        if opt.loadPath:
            startEpoch, train_loss, val_loss, _ = load_checkpoint(
                                                    Path(opt.loadPath),
                                                    model,
                                                    optimizer)
        
        for epoch in range(startEpoch+1, opt.nEpochs+1):
            epoch_train_loss = train(trainSet, model, 
                                    criterion, optimizer, 
                                    device, opt.batch_size, epoch)
            epoch_val_loss, val_metrics = inference(valSet, model, criterion, 
                                                    evaluator, metrics,
                                                    device, opt.batch_size)
            
            optimal_val_thresholds = val_metrics['Classifier thresholds']
            metric_results = {key : val_metrics[key] for key in metrics}
            
            print('Epoch {} completed: \nTrain loss: {:.4f} \nValidation loss: {:.4f}'.format(
                epoch, epoch_train_loss, epoch_val_loss))
            print('Validation metrics: {}'.format(metric_results))
            
            if (epoch_val_loss < val_loss):
                val_loss = epoch_val_loss
                save_checkpoint({
                    'epoch' : epoch,
                    'train_loss' : epoch_train_loss,
                    'val_loss' : epoch_val_loss,
                    'classifier_threshold': optimal_val_thresholds,
                    'label_set' : model.labelSet,
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, Path(opt.savePath), 'best.pth.tar')
            
            if (epoch % opt.saveEvery) == 0:
                save_checkpoint({
                    'epoch' : epoch,
                    'train_loss' : epoch_train_loss,
                    'val_loss' : epoch_val_loss,
                    'classifier_threshold': optimal_val_thresholds,
                    'label_set' : model.labelSet,
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, Path(opt.savePath), 'epoch{}.pth.tar'.format(epoch))
    else:
        if opt.loadPath:
            startEpoch, train_loss, val_loss, classifier_threshold = load_checkpoint(
                                                                    Path(opt.loadPath),
                                                                    model,
                                                                    optimizer)
        else:
            raise Exception('Please point to a model using ---loadPath')

        print('---------------------------Running Inferenece---------------------------')
        evaluator.percent = classifier_threshold
        test_loss, test_metrics = inference(testSet, model, criterion, 
                                            evaluator, metrics,
                                            device, opt.batch_size)
        metric_results = {key : test_metrics[key] for key in metrics}

        print('Test loss: {:.4f}'.format(test_loss))
        print('Test metrics: {}'.format(metric_results))
        print('Classifier thresholds: ',classifier_threshold)
