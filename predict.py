import torch

def get_predictions(model, data_loader, device):
    model = model.eval()
    thesis = []
    predictions = []
    prediction_probs = []
    real_values = []
    
    with torch.no_grad():
        for d in data_loader:
            texts = d['combined_thesis']
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
            
            outputs = model(input_ids = input_ids,
                           attention_mask = attention_mask
                           )
            _, pred = torch.max(outputs, dim = 1)
            thesis.extend(texts)
            predictions.extend(pred)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
        
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        
        return thesis, predictions, prediction_probs, real_values

