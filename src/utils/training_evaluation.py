import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

def train_model_one_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(train_loader, desc="Training Batch", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels.float())
        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    train_loss = total_loss / len(train_loader)
    train_accuracy = (np.array(all_preds) == np.array(all_labels)).mean() if len(all_labels) > 0 else 0.0
    train_f1 = f1_score(all_labels, all_preds, zero_division=0) if len(all_labels) > 0 else 0.0
    return train_loss, train_accuracy, train_f1

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating Batch", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).int()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_val_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    if len(all_labels) == 0:
        return avg_val_loss, 0.0, 0.0, 0.0, 0.0

    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    return avg_val_loss, accuracy, precision, recall, f1

def train_fold_routine(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, max_grad_norm):
    warnings.filterwarnings("ignore", category=FutureWarning) 
    history = {
        'epoch': [], 'train_loss': [], 'train_accuracy': [], 'train_f1': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_f1 = train_model_one_epoch(
            model, train_loader, criterion, optimizer, device, max_grad_norm
        )
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(
            model, val_loader, criterion, device
        )

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch + 1}/{num_epochs} Summary:")
        print(f"  Train: Loss: {train_loss:.4f} | Acc: {train_accuracy*100:.2f}% | F1: {train_f1:.4f}")
        print(f"  Val:   Loss: {val_loss:.4f} | Acc: {val_accuracy*100:.2f}% | P: {val_precision:.4f} | R: {val_recall:.4f} | F1: {val_f1:.4f}")
    return history