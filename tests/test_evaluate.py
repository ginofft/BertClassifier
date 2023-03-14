from scr.evaluate import MultiLabelEvaluator
import torch
import math

#dummy test values
targets = torch.as_tensor([[1,1,0], [0,1,1], [0,0,1], [1,1,1], [1,0,0]])
preds = torch.as_tensor([[1,0,0], [0,1,1], [0,1,0], [1,1,0], [1,1,0]])
percent = torch.as_tensor([0.5,0.5,0.5])
#precision(each class): [1, 1/2, 1]
#micro precision: 6/8
#recall(each class): [1, 2/3, 1/3]
#micro recall: 6/9
#f1 (each class): [1, 4/7, 1/2]
#micro f1: 12/17
#macro f1: 29/42


def test_get_micro_f1()-> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent=percent)
   assert math.isclose(evaluator.get_micro_f1(), (12/17), rel_tol = 1e-6)

def test_get_macro_f1() -> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent=percent)
   assert math.isclose(evaluator.get_macro_f1(),(29/42), rel_tol = 1e-6)