import torch
from tqdm import tqdm

import warnings
import logging

logger = logging.getLogger(__name__)

def train_model_one_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm):
    model.train()
    total_loss = 0.0

    dataset_size = len(train_loader.dataset)    
    pbar = tqdm(total=dataset_size,
                desc="Training Samples",
                unit="img",
                leave=False)

    for batch in train_loader:
        images, labels, _ = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels.float())
        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=max_grad_norm)
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        pbar.update(batch_size)

    pbar.close()
    train_loss = total_loss / dataset_size
    
    return train_loss

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_sids = [], [], []

    with torch.no_grad():
        for imgs, labs, sids in tqdm(test_loader, desc="Test Eval", leave=False):
            imgs, labs = imgs.to(device), labs.to(device)
            outputs = model(imgs).squeeze(1)
            loss = criterion(outputs, labs.float())
            total_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy().tolist()

            all_probs.extend(probs)
            all_labels.extend(labs.cpu().numpy().tolist())
            all_sids.extend(sids)

    pat_outputs = {
        'patient_probabilities': all_probs,
        'patient_sids': all_sids,
        'patient_labels': all_labels
    }

    loss = total_loss / len(test_loader.dataset)
    return loss, pat_outputs

def train_fold_routine(model, 
                       train_loader, 
                       val_loader, 
                       criterion, 
                       optimizer, 
                       device, 
                       num_epochs, 
                       max_grad_norm,
                       config, 
                       fold_id, 
                       fold="val"):
    
    warnings.filterwarnings("ignore", category=FutureWarning) 

    seq_len = config.get('sequence_length', 'N/A')
    logger.info(f"\n Fold {fold_id} {fold}, Training:{config['model_type']}, lr:{config['learning_rate']}, eps:{config['num_epochs']} seq_len:{seq_len}")
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'patient_probabilities': [],
        'patient_sids': [],
        'patient_labels': []
    }
    
    for epoch in range(num_epochs):
        
        train_loss = train_model_one_epoch(model, 
                                        train_loader, 
                                        criterion, 
                                        optimizer, 
                                        device, 
                                        max_grad_norm)

        val_loss, pat_outputs, = evaluate_model(model, 
                                                val_loader,
                                                criterion, 
                                                device)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)

        history['val_loss'].append(val_loss)

        history['patient_probabilities'].append(pat_outputs['patient_probabilities'])
        history['patient_sids'].append(pat_outputs['patient_sids'])
        history['patient_labels'].append(pat_outputs['patient_labels'])

        logger.info(f"Epoch {epoch + 1}/{num_epochs} Summary:")
        logger.info(f"  Train Loss: {train_loss:.2f}")
        logger.info(f"  Val Loss: {val_loss:.2f}")

    return history
