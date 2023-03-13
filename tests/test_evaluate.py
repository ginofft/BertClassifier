from scr.evaluate import MultiLabelEvaluator
import torch
import math

#dummy test values
targets = torch.as_tensor([[1,1,0], [0,1,1], [0,0,1], [1,1,1], [1,0,0]])
preds = torch.as_tensor([[1,0,0], [0,1,1], [0,1,0], [1,1,0], [1,1,0]])
#precision(each class): 

def test_micro_f1()-> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent=0.5)
   assert evaluator.get_micro_f1() > 0