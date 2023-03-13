from scr.utils import *
from scr.models import BertMLPClassifier
from scr.dataset import SentenceLabelDataset

def test_increment() -> None:
    assert increment(3) == 4

def test_get_classes() -> None:
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

    modelPath = 'output/CLINC150/best.pth.tar'
    model =  BertMLPClassifier(nClasses = trainSet.nClasses)
    model.to(device)
    load_checkpoint(modelPath, model)

    texts = ['tell me my bank balance',
             'whats 2+5',
             'how to say goodbye in mandarin',
             'how to make fried eggs']

    results = predict_class(texts, model, tokenizer, labelSet)
    assert len(results) == len(texts)    