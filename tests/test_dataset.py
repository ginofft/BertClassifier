from src.dataset import SmartCollator, SentenceLabelDataset
from src.utils import read_MixSNIPs_file, get_label_set

from torch.utils.data import DataLoader
from transformers import BertTokenizer
import random

dataPath = 'data/MixSNIPs'

trainList = read_MixSNIPs_file(dataPath + '/train.txt')
testList = read_MixSNIPs_file(dataPath + '/test.txt')
valList = read_MixSNIPs_file(dataPath + '/dev.txt')

labelSet = get_label_set(trainList, testList, valList)
trainSet = SentenceLabelDataset(trainList, labelSet, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'))

collator = SmartCollator(trainSet.tokenizer.pad_token_id, trainSet.nClasses)

def test_len_() -> None:
    assert len(trainSet) == len(trainList)

def test_label_set() -> None:
    assert len(labelSet) == len(trainSet.labelSet)

def test_get_item() -> None:
    index = random.randint(0, len(labelSet))
    data = trainSet[index]
    assert type(trainSet[index]) == dict 

# def test_dataloader() -> None:
#     dataloader = DataLoader(trainSet, batch_size=64, shuffle=True, collate_fn=SmartCollator.collate_dynamic_padding)