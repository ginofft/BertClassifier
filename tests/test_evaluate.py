from scr.evaluate import MultiLabelEvaluator
import torch
import math

#dummy test values
targets = torch.as_tensor([[1,1,0], [0,1,1], [0,0,1], [1,1,1], [1,0,0]])
preds = torch.as_tensor([[1,0,0], [0,1,1], [0,1,0], [1,1,0], [1,1,0]])
percent = torch.as_tensor([0.5,0.5,0.5])
#micro accuracy: 2/3
#macro accuracy: 2/3
#precision(each class): [1, 1/2, 1]
#micro precision: 6/8
#recall(each class): [1, 2/3, 1/3]
#micro recall: 6/9
#f1 (each class): [1, 4/7, 1/2]
#micro f1: 12/17
#macro f1: 29/42

def test_get_micro_accuracy() -> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent=percent)
   assert math.isclose(evaluator.get_micro_accuracy(), (2/3), rel_tol=1e-6)

def test_get_macro_accuracy() -> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent= percent)
   assert math.isclose(evaluator.get_macro_accuracy(), (2/3), rel_tol=1e-6)

def test_get_precision() -> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent=percent)
   result = torch.as_tensor([1,1/2,1])
   isClose = torch.isclose(result, evaluator.get_precision())
   assert all(isClose)

def test_get_recall() -> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent=percent)
   result = torch.as_tensor([1,2/3,1/3])
   isClose = torch.isclose(result, evaluator.get_recall())
   assert all(isClose)

def test_get_micro_precision() -> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent=percent)
   assert math.isclose(evaluator.get_micro_precision(), (3/4), rel_tol=1e-6)

def test_get_micro_recall() -> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent=percent)
   assert math.isclose(evaluator.get_micro_recall(), (2/3), rel_tol=1e-6)

def test_get_micro_f1()-> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent=percent)
   assert math.isclose(evaluator.get_micro_f1(), (12/17), rel_tol = 1e-6)

def test_get_macro_f1() -> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent=percent)
   assert math.isclose(evaluator.get_macro_f1(),(29/42), rel_tol = 1e-6)

def test_clean() -> None:
   evaluator = MultiLabelEvaluator(preds, targets, percent)
   evaluator.clean()
   assert all([attr is None for attr in evaluator.__dict__.values()])

def test_add_batch() -> None:
   evaluator = MultiLabelEvaluator()
   evaluator.add_batch(preds, targets)
   evaluator.add_batch(preds, targets)
   assert (evaluator.probs.shape[0] == 2*preds.shape[0])