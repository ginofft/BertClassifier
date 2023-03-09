import torch

def calculate_multi_label_accuracy(preds: torch.Tensor, targets: torch.Tensor):
    if preds.device != targets.device:
        raise TypeError('input tensors must be on the same device')
    assert preds.shape == targets.shape, "the two input tensors must have the same shape"
    return (torch.sum(preds*targets).item() / torch.numel(preds))

