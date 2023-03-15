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
    `preds` : Torch.Tensor
        Matrix of size [no. data, no. classes], where `preds[i,j] == 1` if `probs[i,j] > percent[j]`.
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

    def clean(self):
        self.probs = None
        self.targets = None
        self.percent = None
    
    @property
    def preds(self):
        return (self.probs >= self.percent.squeeze(0)).int()

    @property
    def truePositive(self):
        return torch.logical_and(self.preds, self. targets)
    
    @property
    def falsePositive(self):
        return torch.logical_and(self.preds, torch.logical_not(self.targets))

    @property
    def trueNegative(self):
        return torch.logical_and(torch.logical_not(self.preds),
                                 torch.logical_not(self.targets))

    @property 
    def falseNegative(self):
        return torch.logical_and(torch.logical_not(self.preds),
                                 self.targets)    

    def add_batch(self, probs, targets):
        if self.probs is None or self.targets is None:
            self.probs = probs
            self.targets = targets
        else:
            self.probs = torch.cat((self.probs, probs), dim =0)
            self.targets = torch.cat((self.targets, targets), dim =0)
        self._check()

    def _check(self) -> None:
        if self.probs == None or self.targets == None:
            raise Exception("empty input tensor")
        if self.probs.size() != self.targets.size():
            raise Exception("input tensors must have the same size")
        if self.probs.dim() != 2:
            raise Exception("input tensors must have two dimension")

    def get_class_accuracy(self) -> torch.Tensor:
        classTP = torch.sum(self.truePositive, dim=0)
        classFP = torch.sum(self.falsePositive, dim =0)
        classTN = torch.sum(self.trueNegative, dim=0)
        classFN = torch.sum(self.falseNegative, dim =0)
        classAccuracy = (classTP + classTN)/(classTP+classTN+classFP+classFN)
        return classAccuracy
 
    def get_micro_accuracy(self) -> float:
        correct = torch.sum(self.truePositive) + torch.sum(self.trueNegative)
        all = correct + torch.sum(self.falsePositive) + torch.sum(self.falseNegative)
        return (correct/all).item()

    def get_macro_accuracy(self) -> float:
        classAccuracy = self.get_class_accuracy()
        return torch.mean(classAccuracy).item()

    def get_class_precision(self) -> torch.Tensor:
        classPrecsion = torch.sum(self.truePositive, dim =0) / torch.sum(self.preds, dim =0)
        return classPrecsion
    
    def get_micro_precision(self) -> float:
        return (torch.sum(self.truePositive) / (torch.sum(self.preds))).item()
    
    def get_macro_precision(self) -> float:
        classPrecsion = self.get_class_precision()
        return torch.mean(classPrecsion).item()

    def get_class_recall(self) -> torch.Tensor:
        classRecall = torch.sum(self.truePositive, dim =0) / torch.sum(self.targets, dim =0)
        return classRecall

    def get_micro_recall(self) -> float:
        return (torch.sum(self.truePositive) / (torch.sum(self.targets))).item()

    def get_macro_recall(self) -> float:        
        classRecall = self.get_class_recall()
        return torch.mean(classRecall).item()

    def get_class_f1(self) -> torch.Tensor:
        tp = torch.sum(self.truePositive, dim=0)
        fp = torch.sum(self.falsePositive, dim=0)
        fn = torch.sum(self.falseNegative, dim=0) 
        classF1 = (tp/(tp+0.5*(fp+fn)))
        return classF1

    def get_micro_f1(self) -> float:
        truePositive = torch.sum(self.truePositive)
        falsePositive = torch.sum(self.falsePositive)
        falseNegative = torch.sum(self.falseNegative)
        return (truePositive/(truePositive+0.5*(falsePositive+falseNegative))).item()

    def get_macro_f1(self) -> float:
        classF1 = self.get_class_f1()
        return torch.mean(classF1).item()
    
    def get_optimal_percent(self):
        highestClassF1 = torch.zeros(self.probs.size()[1])
        result = self.probs[0]
        for prob in self.probs:
            self.percent = prob
            classF1 = self.get_class_f1()
            changeVector = classF1 > highestClassF1
            highestClassF1 = torch.where(changeVector, classF1, highestClassF1)
            result = torch.where(changeVector, prob, result)
        self.percent = result
        return self.percent
