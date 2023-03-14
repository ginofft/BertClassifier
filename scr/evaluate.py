import torch
from .patterns import Singleton
from typing import Optional
class MultiLabelEvaluator(metaclass = Singleton):
     
    def __init__(self, probs : Optional[torch.Tensor] = None, 
                 targets : Optional[torch.Tensor] = None, percent : Optional[torch.Tensor] = None):
        self.probs = probs
        self.targets = targets
        self.percent = percent

        self.positive = None
        self.truePositive = None
        self.falsePositive = None

    def clean(self):
        self.probs = None
        self.targets = None
        
        self.positive = None
        self.truePositive = None
        self.falsePositive = None
    
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

    def _get_positives(self):
        self._check()
        if (self.positive != None) and (self.truePositive != None):
            return self.positive, self.truePositive, self.falsePositive
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        self.positive = torch.logical_or(preds, self.targets)
        self.truePositive = torch.logical_and(preds, self.targets)
        self.falsePositive = torch.where(self.truePositive==0, preds, 0)
        return self.positive, self.truePositive, self.falsePositive
    
    def get_accuracy(self):
        positive, truePositive, _ = self._get_positives()
        classAccuracy = torch.sum(truePositive, dim=0) / torch.sum(positive, dim=0)
        return classAccuracy
    
    def get_precision(self):
        positive, truePositive, falsePositive = self._get_positives()
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        classPrecsion = torch.sum(truePositive, dim =0) / torch.sum(preds, dim =0)
        return classPrecsion
    
    def get_recall(self):
        positive, truePositive, falsePositive = self._get_positives()
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        classRecall = torch.sum(truePositive, dim =0) / torch.sum(self.targets, dim =0)
        return classRecall
    
    def get_F1(self):
        classPrecision, precision = self.get_precision()
        classRecall, recall = self.get_recall()

        classF1 = (2*classPrecision*classRecall)/(classPrecision+classRecall)
        f1 = (2*precision*recall)/(precision+recall)
        return classF1, f1

    def get_micro_precision(self):
        pass
    def get_micro_recall(self):
        pass
         
    def get_micro_f1(self):
        positive, truePositive, falsePositive = self._get_positives()
        preds = (self.probs > self.percent.unsqueeze(0)).int()
        precision = (torch.sum(truePositive) / torch.sum(preds)).item()
        recall = (torch.sum(truePositive) / torch.sum(self.targets)).item()
        return (2*precision*recall)/(precision+recall)

    def get_macro_f1(self):
        classF1, _ = self.get_F1()
        f1 = torch.mean(classF1) 
        return f1