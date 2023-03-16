import torch
from torch.utils.data import DataLoader
from .dataset import SmartCollator, SentenceLabelDataset
from .evaluate import MultiLabelEvaluator
from typing import List

EVALUATION_MAP = {
    'macro accuracy' : MultiLabelEvaluator.get_macro_accuracy,
    'micro accuracy' : MultiLabelEvaluator.get_micro_accuracy,
    'macro precision' : MultiLabelEvaluator.get_macro_precision,
    'micro precision' : MultiLabelEvaluator.get_micro_precision,
    'macro recall' : MultiLabelEvaluator.get_macro_recall,
    'micro recall' : MultiLabelEvaluator.get_micro_recall,
    'macro f1' : MultiLabelEvaluator.get_macro_f1,
    'micro f1' : MultiLabelEvaluator.get_micro_f1,
}
sigmoid = torch.nn.Sigmoid()

def train(
        train_set:SentenceLabelDataset,
        model,
        criterion,
        optimizer,
        device=torch.device("cuda"),
        batch_size=8,  
        epoch=1):
    
    collator = SmartCollator(pad_token_id = train_set.tokenizer.pad_token_id, nClasses=train_set.nClasses)
    dataloader = DataLoader(train_set, 
                            batch_size = batch_size, 
                            num_workers = 2, 
                            shuffle = True,
                            collate_fn = collator.collate_dynamic_padding,
                            pin_memory = True)
    n_batches = len(dataloader)

    epoch_loss = 0
    start_iter = 1
    model.train()
    for batch_id, data in enumerate(dataloader, start_iter):
        input_ids = data['input_ids'].to(device)
        attention_masks = data['attention_mask'].to(device)
        targets = data['labels'].to(device)

        embeddings = model(input_ids, attention_masks)
        loss = criterion(embeddings, targets).to(device)
        loss.backward()
        optimizer.step()  

        batch_loss = loss.item()
        epoch_loss += batch_loss

        del input_ids, attention_masks, embeddings
        del loss

        if batch_id % 200 ==0 or n_batches <= 10:
            print('Epoch[{}]({}/{}) Loss: {:.6f}'.format(epoch,
                                                        batch_id, 
                                                        n_batches,
                                                        batch_loss))
            del batch_loss
    avg_loss = epoch_loss / n_batches
    # print('--> Epoch {} completed, train avg. loss: {:.6f}'.format(epoch, avg_loss))
    # print('Epoch {} Metrics: {}'.format(epoch,metrics.compute()))
    del dataloader
    
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()
    return avg_loss

def inference(testSet : SentenceLabelDataset,
        model,
        criterion, 
        evaluator : MultiLabelEvaluator,
        metrics : List[str]=['macro f1'],
        device = torch.device('cuda'),
        batch_size=8,):
    collator = SmartCollator(pad_token_id = testSet.tokenizer.pad_token_id, nClasses=testSet.nClasses)
    evalFuns = [EVALUATION_MAP[metric.lower()] for metric in metrics]
    dataloader = DataLoader(testSet, 
                            batch_size = batch_size, 
                            num_workers = 2, 
                            shuffle = True,
                            collate_fn = collator.collate_dynamic_padding,
                            pin_memory = True)
    epoch_loss = 0
    n_batches = len(dataloader)
    model.eval()
    start_iter = 1
    with torch.no_grad():
        for batch_id, data in enumerate(dataloader, start_iter):
            input_ids = data['input_ids'].to(device)
            attention_masks = data['attention_mask'].to(device)
            targets = data['labels'].to(device)

            embeddings = model(input_ids, attention_masks)
            loss = criterion(embeddings, targets).to(device)

            batch_loss = loss.item()
            epoch_loss += batch_loss

            probs = sigmoid(embeddings)
            evaluator.add_batch(probs = probs, targets = targets)

            del input_ids, attention_masks, embeddings
            del loss
            del batch_loss
    avg_loss = epoch_loss / n_batches
    
    if evaluator.percent == None:
        evaluator.percent = evaluator.probs[0]
        evaluator.get_optimal_percent()
    
    metricDict = {"Classifier thresholds" : evaluator.percent}

    metricResults = []
    for fun in evalFuns:
        metricResults.append(fun(evaluator))
    metricDict.update(dict(zip(metrics, metricResults)))
    
    evaluator.clean()
    del dataloader
    
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()
    return avg_loss, metricDict
