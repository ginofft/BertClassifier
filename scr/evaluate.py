import torch
from patterns import Singleton
class MultiLabelEvaluator(metaclass = Singleton):
    
    preds : torch.Tensor
    targets : torch.Tensor
    percent = 0.5

    def _check_size(self) -> None:
        if self.preds.size() != self.targets.size():
            raise Exception("preds and targets must be the same size")
        if self.preds.dim() != 2:
            raise Exception("preds and targets must have two dimensions")

    def get_confusion_matrix(self) -> torch.Tensor:
        pass

