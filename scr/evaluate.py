import torch
from .patterns import Singleton
class MultiLabelEvaluator(metaclass = Singleton):

    def __init__(self, probs, targets, percent = 0.5):
        self.probs = probs
        self.targets = targets
        self.percent = percent

        self.positive = None
        self.truePositive = None
        self.falsePositive = None

        self._check_size()

    def _check_size(self) -> None:
        if self.probs.size() != self.targets.size():
            raise Exception("input tensors must have the same size")
        if self.probs.dim() != 2:
            raise Exception("input tensors must have two dimension")

    def _get_positives(self):
        self._check_size()
        if (self.positive != None) and (self.truePositive != None):
            return self.positive, self.truePositive, self.falsePositive
        preds = (self.probs > self.percent).int()
        self.positive = torch.logical_or(preds, self.targets)
        self.truePositive = torch.logical_and(preds, self.targets)
        self.falsePositive = torch.where(self.truePositive==0, preds, 0)
        return self.positive, self.truePositive, self.falsePositive
    
    def get_accuracy(self):
        positive, truePositive, _ = self._get_positives()
        classAccuracy = torch.sum(truePositive, dim=0) / torch.sum(positive, dim=0)
        accuracy = (torch.sum(truePositive) / torch.sum(positive)).item()
        return classAccuracy, accuracy
    
    def get_precision(self):
        positive, truePositive, falsePositive = self._get_positives()
        preds = (self.probs > self.percent).int()
        classPrecsion = torch.sum(truePositive, dim =0) / torch.sum(preds, dim =0)
        precision = torch.sum(truePositive) / torch.sum(preds).item()
        return classPrecsion, precision
    
    def get_recall(self):
        positive, truePositive, falsePositive = self._get_positives()
        preds = (self.probs > self.percent).int()
        classRecall = torch.sum(truePositive, dim =0) / torch.sum(self.targets, dim =0)
        recall = torch.sum(truePositive) / torch.sum(self.targets).item()
        return classRecall, recall