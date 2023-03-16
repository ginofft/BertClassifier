# BERT Intent Classifier for CLINC150 Dataset
Pytorch and Huggingface implementation of a multi label intent classifier with BERT as the encoder and a MLP as the classification head.

This repo currently works for two dataset: [CLINC150](https://github.com/clinc/oos-eval) and [MixSNIPs](https://github.com/LooperXX/AGIF/tree/master/data/MixSNIPS_clean).

User are recommended to write custom scripts to write your data into `scr.dataset.SentenceLabelDataset`, everything else should works just fine.

Features:
- MLP With BCELoss : independent classifier for each classes.
- A Multi Label evaluator : with an method to get thresholding value that maximize **macro f1**.
- Dynamic Padding : for faster training and inference.
## References
* **Stefan Larson et al**. *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction (2019)*. https://doi.org/10.48550/arXiv.1909.02027

* **Libo Qin, Xiao Xu, Wanxiang Che and Ting Liu**. *AGIF: An Adaptive Graph-Interactive Framework for Joint Multiple Intent Detection and Slot Filling (2020)*. https://aclanthology.org/2020.findings-emnlp.163/
  
* **Rong-En Fan and Chih-Jen Lin**. *A Study on Threshold Selection for Multi-label Classification*. https://www.csie.ntu.edu.tw/~cjlin/papers/threshold.pdf

* **Michaël Benesty**. *Divide Hugging Face Transformers training time by 2 or more with dynamic padding and uniform length batching*. https://towardsdatascience.com/divide-hugging-face-transformers-training-time-by-2-or-more-21bf7129db9q-21bf7129db9e

## Quick Start
To train:
```
python main.py --mode train \
    --datasetFormat CLINC150 \
    --nEpochs 500 --saveEvery 10 \
    --datasetPath data/CLINC150/data_oos_plus.json \
    --metrics 'marco f1' \
    --savePath output
```

To inference:
```
python main.py --mode inference \
    --datasetFormat CLINC150 \
    --datasetPath data/data_oos_plus.json \
    --metrics 'macro f1' \
    --loadPath output/best.pth.tar
```
## Custom stuff 
You might need to change the dataloader to suit your need (or dataset), the code you need to modify is found in main: 
```
    if opt.dataFormat.lower() == 'clinc150':
        dataDict = read_CLINC150_file(opt.datasetPath)
        trainList = dataDict['train'] + dataDict['oos_train']
        valList = dataDict['val'] + dataDict['oos_val']
        testList = dataDict['test'] + dataDict['oos_test']
        turn_single_label_to_multilabels(trainList, valList, testList)
    
    if opt.dataFormat.lower() == 'mixsnips':
        trainPath = opt.datasetPath + '/train.txt'
        valPath = opt.datasetPath + '/dev.txt'
        testPath = opt.datasetPath + '/test.txt'

        trainList = read_MixSNIPs_file(trainPath)
        valList = read_MixSNIPs_file(valPath)
        testList = read_MixSNIPs_file(testPath)
```
where `read_CLINE150_file()` and `read_MixSNIPs_file()` were my custom functions.