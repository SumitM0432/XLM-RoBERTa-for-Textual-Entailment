import numpy as np
import torch
from torch import nn


def train_roberta_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    
    # Putting the model in training state from default eval mode
    model.train()
    
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        
        # Taking the inputs for the single batch that we created using the data_loader
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        
        # Feeding this to the model
        outputs = model(input_ids = input_ids,
                        attention_mask = attention_mask,
                        labels = targets
                       )
        
        # taking the output with the highest probability
        _, preds = torch.max(outputs, dim = 1)
        
        # 
        loss = loss_fn(outputs, targets)
        
        # predictions which are right
        correct_predictions +=torch.sum(preds == targets)
        
        # appending the losses in the list
        losses.append(loss.item())
        
        # back propagation
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        
        # this will take the step based on the gradient of the parameters
        optimizer.step()
        
        # this will change the learning rate after epochs if required or otherwise the lr remains the initial value
        scheduler.step()
        
        # zeroing the gradients for the next step or it will just accumulate
        optimizer.zero_grad()
        
    return correct_predictions.double()/n_examples, np.mean(losses)

# eval function mostly same as the train
def eval_roberta(model, data_loader, loss_fn, device, n_examples):
    
    # putting the model in evaluation mode
    model = model.eval()
    
    # saving this for the total correct predictions and mean losses
    losses = []
    correct_predictions = 0
    
    # this with torch.no_grad() will just disable the torch gradient which will be faster
    with torch.no_grad():
        
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            
            outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              labels = targets
                )
            
            _, preds = torch.max(outputs, dim=1)
            
            loss = loss_fn(outputs, targets)
            
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)
