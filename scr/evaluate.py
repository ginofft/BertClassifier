import torch
from .patterns import Singleton
from typing import Optional
class MultiLabelEvaluator(metaclass = Singleton):
     
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
    
    def get_micro_accuracy(self):
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        correct = torch.sum(truePositive) + torch.sum(trueNegative)
        all = correct + torch.sum(falsePositive) + torch.sum(falseNegative)
        return correct/all

    def get_macro_accuracy(self):
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        classTP = torch.sum(truePositive, dim=0)
        classFP = torch.sum(falsePositive, dim =0)
        classTN = torch.sum(trueNegative, dim=0)
        classFN = torch.sum(falseNegative, dim =0)
        classAccuracy = (classTP + classTN)/(classTP+classTN+classFP+classFN)
        return torch.mean(classAccuracy)

    def get_precision(self):
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        classPrecsion = torch.sum(truePositive, dim =0) / torch.sum(preds, dim =0)
        return classPrecsion
    
    def get_recall(self):
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        classRecall = torch.sum(truePositive, dim =0) / torch.sum(self.targets, dim =0)
        return classRecall

    def get_micro_precision(self):
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        return torch.sum(truePositive) / (torch.sum(preds))

    def get_micro_recall(self):
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        return torch.sum(truePositive) / (torch.sum(self.targets))
         
    def get_micro_f1(self):
        truePositive, falsePositive, trueNegative, falseNegative = self._get_positives_and_negatives()
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        precision = (torch.sum(truePositive) / torch.sum(preds)).item()
        recall = (torch.sum(truePositive) / torch.sum(self.targets)).item()
        return (2*precision*recall)/(precision+recall)

    def get_macro_f1(self):
        classPrecision = self.get_precision()
        classRecall = self.get_recall()
        classF1 = (2*classPrecision*classRecall) / (classPrecision+classRecall)
        f1 = torch.mean(classF1).item()
        return f1