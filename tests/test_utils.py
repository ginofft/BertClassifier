from src.utils import *
from src.models import BertMLPClassifier
from src.dataset import SentenceLabelDataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dataDict = read_CLINC150_file('D:/Projects/BertClassifier/data/CLINC150/data_oos_plus.json')
trainList = dataDict['train'] + dataDict['oos_train']
valList = dataDict['val'] + dataDict['oos_val']
testList = dataDict['test'] + dataDict['oos_test']
turn_single_label_to_multilabels(trainList, valList, testList)
labelSet = get_label_set(trainList, valList, testList)
trainSet = SentenceLabelDataset(trainList, labelSet)
tokenizer = trainSet.tokenizer

modelPath = 'output/CLINC150/multiLabel.pth.tar'
model =  BertMLPClassifier(nClasses = trainSet.nClasses)
model.to(device)
load_checkpoint(modelPath, model)

predictor = Predictor(model, tokenizer, labelSet, device)

def test_get_class() -> None:
    texts = ['tell me my bank balance',
             'whats 2+5',
             'how to say goodbye in mandarin',
             'how to make fried eggs']

    results = predictor.get_class(texts)
    assert all(type(r)==str for r in results)
    assert len(results) == len(texts)

def test_get_topk_classes() -> None:
    texts = ['tell me my bank balance',
             'whats 2+5',
             'how to say goodbye in mandarin',
             'how to make fried eggs']
    k = 2
    results = predictor.get_topk_classes(texts, k)
    assert all(len(r)==k for r in results)
    assert len(results) == len(texts)

def test_get_classes_at_percent() -> None:
    texts = ['tell me my bank balance',
             'whats 2+5',
             'how to say goodbye in mandarin',
             'how to make fried eggs']
    p = 0.5
    results = predictor.get_classes_at_percent(texts, p)
    assert all(type(r) is list for r in results)
    assert len(results) == len(texts)