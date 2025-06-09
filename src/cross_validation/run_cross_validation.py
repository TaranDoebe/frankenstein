import logging
import torch
from torch.utils.data import DataLoader

from datasets.fmri_dataset import fMRIDataset, fMRISequentialDataset
from models.cnn_models import (Simple3DCNN, 
                               HighAccuracy3DCNN, 
                               Base3DCNN, 
                               Base3DCNN2, 
                               Base3DCNN3, 
                               Base3DCNN4, 
                               Base3DCNN5, 
                               ResNetCNN, 
                               CNNGRUClassifier)

from cross_validation.run_routine import train_fold_routine


logger = logging.getLogger(__name__)

def select_model(name, device):
    d = {
        "Simple3DCNN":   Simple3DCNN,
        "HighAccuracy3DCNN": HighAccuracy3DCNN,
        "Base3DCNN":     Base3DCNN,
        "Base3DCNN2":     Base3DCNN2,
        "Base3DCNN3":     Base3DCNN3,
        "Base3DCNN4":     Base3DCNN4,
        "Base3DCNN5":     Base3DCNN5,
        "ResNetCNN":     ResNetCNN,
        "CNNGRUClassifier":     CNNGRUClassifier,
    }
    return d[name]().to(device)

def run_cv(
        run_id,
        config,
        train_df,
        data_dict_train_val,
        device,
        val_df=None,
        fold="val",
    ):
    """
    Initialize datalaoders and model, then execute a train fold routine.
    """ 

    # load data
    if config.get('model_type', '')=="CNNGRUClassifier":
        train_loader = DataLoader(fMRISequentialDataset(train_df, data_dict_train_val, augment=True, sequence_length=config['sequence_length']),
                                  batch_size=config['batch_size'], 
                                  shuffle=True,
                                  num_workers=config['num_workers'], 
                                  pin_memory=False, 
                                  persistent_workers=True)
        
        val_loader = DataLoader(fMRISequentialDataset(val_df, data_dict_train_val, augment=False, sequence_length=config['sequence_length']),
                                batch_size=config['batch_size'], 
                                shuffle=False, 
                                num_workers=1, 
                                pin_memory=False, 
                                persistent_workers=True)
    
    else: # CNN-Only
        logging.info(f"\\ {config.get("data_loading_mode")}")

        train_loader = DataLoader(fMRIDataset(train_df, data_dict_train_val,augment=True, raw_timepoints=False),
                                  batch_size=config['batch_size'], 
                                  shuffle=True,
                                  num_workers=config['num_workers'],
                                  pin_memory=False, 
                                  persistent_workers=True)

        val_loader = DataLoader(fMRIDataset(val_df, data_dict_train_val, augment=False, raw_timepoints=False),
                                batch_size=config['batch_size'], 
                                shuffle=False,
                                num_workers=1, 
                                pin_memory=False, 
                                persistent_workers=True)

    # Model
    model = select_model(config.get('model_type', ''), device)
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['learning_rate'])
    
    criterion = torch.nn.BCEWithLogitsLoss()

    # train
    history = train_fold_routine(model, 
                                 train_loader,
                                 val_loader,
                                 criterion, 
                                 optimizer, 
                                 device,
                                 config['num_epochs'], 
                                 config.get('max_grad_norm', 1.0),
                                 config=config,             
                                 fold_id=run_id,
                                 fold=fold)
    return history


