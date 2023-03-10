import torch
from .patterns import Singleton
class MultiLabelEvaluator(metaclass = Singleton):

    def __init__(self, probs, targets, percent = 0.5):
        self.probs = probs
        self.targets = targets
        self.percent = percent

        self.positive = None
        self.truePositive = None

        self._check_size()

    def _check_size(self) -> None:
        if self.probs.size() != self.targets.size():
            raise Exception("input tensors must have the same size")
        if self.probs.dim() != 2:
            raise Exception("input tensors must have two dimension")

    def _get_positives(self):
        self._check_size()
        if (self.positive != None) and (self.truePositive != None):
            return self.positive, self.truePositive
        preds = (self.probs > self.percent).int()
        self.positive = torch.logical_or(preds, self.targets)
        self.truePositive = torch.logical_and(preds, self.targets)
        return self.positive, self.truePositive
    
    def get_accuracy(self):
        positive, truePositive = self._get_positives()
        classAccuracy = torch.sum(truePositive, dim=1) / torch.sum(positive, dim=1)
        accuracy = (torch.sum(truePositive) / torch.sum(positive)).item()
        return classAccuracy, accuracy
    