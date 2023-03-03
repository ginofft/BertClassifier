# BERT Intent Classifier for CLINC150 Dataset
Pytorch and Huggingface implementation of a intent classifier with BERT as the encoder and a MLP as the classification head.
While developed for [CLINC150](https://github.com/clinc/oos-eval) dataset, simply pass new dataset in form of .json work just fine.

## References
* **Stefan Larson et al**. *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction (2019)*. https://doi.org/10.48550/arXiv.1909.02027

## Quick Start
This repo will work with any .json dataset with similar format to CLINC150. Specifically:
```json
    {
    "oos_val": [
        [
            "set a warning for when my bank account starts running low", 
            "oos"
        ], 
        [
            "set up a 52 minute timer", 
            "timer"
        ],
            ...
        ],
    "train":[
        [
            "you shall address me as nick", 
            "change_user_name"
        ], 
        [
            "my name is nick", 
            "change_user_name"
        ], 
        ]
    }
```
You might need to change the dataloader to suit your need (or dataset), the code you need to modify is found in main: 
```
    dataDict = get_data_from_json(opt.datasetPath)
    trainList = dataDict['train'] + dataDict['oos_train']
    valList = dataDict['val'] + dataDict['oos_val']
    testList = dataDict['test'] + dataDict['oos_test']
    labelSet = get_label_set(trainList, valList, testList)
```

To train:
```
python main.py --mode train --mode train --nEpochs 500 --saveEvery 10 \
    --datasetPath data/data_oos_plus.json \
    --savePath output
```

To inference:
```
python main.py --mode inference \
    --datasetPath data/data_oos_plus.json \
    --loadPath output/best.pth.tar
```
