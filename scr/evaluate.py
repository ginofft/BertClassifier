import torch
from .patterns import Singleton
from typing import Optional
class MultiLabelEvaluator(metaclass = Singleton):
    """Singleton class used for calculating various metrics for multi-label classification, such as: accuracy, precision, recall and f1

    Attributes
    ----------
    `probs` : Torch.Tensor
        Matrix of size [no. data, no. classes], where `probs[i,j]` denote the probability of the i'th data contains the j'th class
    `targets` : Torch.Tensor
        Matrix of size [no. data, no. classes], where `targets[i,j] == 1` if the i'th data contains the j'th class
    `percent` : Torch.Tensor
        Vector of size [no. classes], where `percent[j]` is the classification threshold for the j'th class
    `truePositive` : Torch.Tensor
        Matrix of size [no. data, no. classes] denoting true positive.
    `falsePositive` : Torch.Tensor
        Matrix of size [no. data, no. classes] denoting false positive.
    `trueNegative` : Torch.Tensor
        Matrix of size [no. data, no. classes] denoting true negative.
    `falseNegative` : Torch.Tensor
        Matrix of size [no. data, no. classes] denoting false negative.
    
    Methods
    -------
    :meth:`add_batch(probs, targets)` -> None
        add probs and targets into self.probs and self.targets respectively
    :meth:`clean()` -> None
        make all attribute into None, as this is a Singleton class
    :meth:`get_class_accuracy()` -> torch.Tensor :
        Vector of size [no. classes], containing accuracy of each class
    :meth:`get_class_precision()` -> torch.Tensor :
        Vector of size [no. classes], containing precision of each class     
    :meth:`get_class_recall()` -> torch.Tensor :
        Vector of size [no. classes], containing recall of each class
    :meth:`get_class_f1()` -> torch.Tensor :
        Vector of size [no. classes], containing f1 of each class
    :meth:`get_micro_accuracy()` -> float :

    :meth:`get_macro_accuracy()` -> float :

    :meth:`get_micro_precision()` -> float : 

    :meth:`get_macro_precision()` -> float :

    :meth:`get_micro_recall()` -> float :

    :meth:`get_macro_recall()` -> float :

    :meth:`get_micro_f1()` -> float :

    :meth:`get_macro_f1()` -> float :


    """
    def __init__(self, probs : Optional[torch.Tensor] = None, 
                 targets : Optional[torch.Tensor] = None, percent : Optional[torch.Tensor] = None):
        self.probs = probs
        self.targets = targets
        self.percent = percent

        self.truePositive = None
        self.falsePositive = None
        self.trueNegative = None
        self.falseNegative = None

    def clean(self):
        self.probs = None
        self.targets = None
        self.percent = None

        self.truePositive = None
        self.falsePositive = None
        self.trueNegative = None
        self.falseNegative = None
    
    def add_batch(self, probs, targets):
        if self.probs is None or self.targets is None:
            self.probs = probs
            self.targets = targets
        else:
            self.probs = torch.cat((self.probs, probs), dim =0)
            self.targets = torch.cat((self.targets, targets), dim =0)

    def _check(self) -> None:
        if self.probs == None or self.targets == None:
            raise Exception("empty input tensor")
        if self.probs.size() != self.targets.size():
            raise Exception("input tensors must have the same size")
        if self.probs.dim() != 2:
            raise Exception("input tensors must have two dimension")

    def _get_positives_and_negatives(self):
        self._check()
        if  (self.truePositive != None) and \
            (self.falsePositive != None) and \
            (self.trueNegative != None) and \
            (self.falseNegative != None):
            return self.truePositive, self.falsePositive, self.trueNegative, self.falseNegative
        
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        
        #god i love my brain
        self.truePositive = torch.logical_and(preds, self.targets)
        self.falsePositive = torch.logical_and(preds, torch.logical_not(self.targets))
        self.trueNegative = torch.logical_and(torch.logical_not(preds), 
                                            torch.logical_not(self.targets))
        self.falseNegative = torch.logical_and(torch.logical_not(preds), self.targets) 
        return self.truePositive, self.falsePositive, self.trueNegative, self.falseNegative

    def get_class_accuracy(self) -> torch.Tensor:
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        classTP = torch.sum(truePositive, dim=0)
        classFP = torch.sum(falsePositive, dim =0)
        classTN = torch.sum(trueNegative, dim=0)
        classFN = torch.sum(falseNegative, dim =0)
        classAccuracy = (classTP + classTN)/(classTP+classTN+classFP+classFN)
        return classAccuracy
 
    def get_micro_accuracy(self) -> float:
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        correct = torch.sum(truePositive) + torch.sum(trueNegative)
        all = correct + torch.sum(falsePositive) + torch.sum(falseNegative)
        return (correct/all).item()

    def get_macro_accuracy(self) -> float:
        classAccuracy = self.get_class_accuracy()
        return torch.mean(classAccuracy).item()

    def get_class_precision(self) -> torch.Tensor:
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        classPrecsion = torch.sum(truePositive, dim =0) / torch.sum(preds, dim =0)
        return classPrecsion
    
    def get_micro_precision(self) -> float:
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        return (torch.sum(truePositive) / (torch.sum(preds))).item()
    
    def get_macro_precision(self) -> float:
        classPrecsion = self.get_class_precision()
        return torch.mean(classPrecsion).item()

    def get_class_recall(self) -> torch.Tensor:
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        classRecall = torch.sum(truePositive, dim =0) / torch.sum(self.targets, dim =0)
        return classRecall

    def get_micro_recall(self) -> float:
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        return (torch.sum(truePositive) / (torch.sum(self.targets))).item()

    def get_macro_recall(self) -> float:        
        classRecall = self.get_class_recall()
        return torch.mean(classRecall).item()

    def get_class_f1(self) -> torch.Tensor:
        classPrecision = self.get_class_precision()
        classRecall = self.get_class_recall()
        classF1 = (2*classPrecision*classRecall) / (classPrecision+classRecall)
        return classF1

    def get_micro_f1(self) -> float:
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        precision = (torch.sum(truePositive) / torch.sum(preds)).item()
        recall = (torch.sum(truePositive) / torch.sum(self.targets)).item()
        return (2*precision*recall)/(precision+recall)

    def get_macro_f1(self) -> float:
        classF1 = self.get_class_f1()
        return torch.mean(classF1).item()
    
class ThresholdOptimizer():
    pass