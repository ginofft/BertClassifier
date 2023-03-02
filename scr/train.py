import torch
from torch.utils.data import DataLoader
from .dataset import SmartCollator
import evaluate

def train(
        train_set,
        model,
        criterion,
        optimizer,
        device=torch.device("cuda"),
        batch_size=8,  
        epoch=1):
    
    collator = SmartCollator(pad_token_id = train_set.tokenizer.pad_token_id)
    metrics = evaluate.load("accuracy")
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
        labels = data['labels'].to(device)

        embeddings = model(input_ids, attention_masks)
        loss = criterion(embeddings, labels).to(device)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss

        # Calculate Metrics
        preds = torch.argmax(embeddings, dim=1)
        metrics.add_batch(references = labels, predictions = preds)

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
    return avg_loss, metrics.compute()

def inference(testSet,
        model,
        criterion, 
        device = torch.device('cuda'),
        batch_size=8,):
    collator = SmartCollator(pad_token_id = testSet.tokenizer.pad_token_id)
    metrics = evaluate.load("accuracy")
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
            labels = data['labels'].to(device)

            embeddings = model(input_ids, attention_masks)
            loss = criterion(embeddings, labels).to(device)  
            
            batch_loss = loss.item()
            epoch_loss += batch_loss

            # Calculate Metrics
            preds = torch.argmax(embeddings, dim=1)
            metrics.add_batch(references = labels, predictions = preds)

            del input_ids, attention_masks, embeddings
            del loss
            del batch_loss
    avg_loss = epoch_loss / n_batches
    
    # print('---> Inference loss: {:.6f}'.format(avg_loss), flush = True)
    # print('Inference Metrics: {}'.format(metrics.compute()), flush = True)
    del dataloader
    
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()
    return avg_loss, metrics.compute()
