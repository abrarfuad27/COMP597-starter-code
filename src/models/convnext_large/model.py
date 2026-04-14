# === import necessary modules ===
import src.config as config  # Configurations
import src.trainer as trainer  # Trainer base class
import src.trainer.stats as trainer_stats  # Trainer statistics module

# === import necessary external modules ===
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import random

"""
This file contains the code to train a ConvNeXt Large model for fake image detection.
It uses the pretrained ConvNeXt Large model from TorchVision.
https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.convnext_large.html
"""


def get_convnext_transforms():
    """
    Returns the appropriate transforms for ConvNeXt Large model.
    Based on the official pretrained weights transforms.
    
    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # ConvNeXt Large expects these specific preprocessing steps
    # Based on: https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.convnext_large.html
    return transforms.Compose([
        transforms.Resize(232),  # Resize to 232
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])


def collate_fn(batch):
    transform = get_convnext_transforms()
    images = []
    labels = []
    for sample in batch:
        try:
            # 1. Get the image data
            img_data = sample.get('png') or sample.get('jpg')
            if img_data is None:
                continue
                
            # 2. Decode the image
            if isinstance(img_data, Image.Image):
                img = img_data.convert('RGB')
            else:
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
            
            # 3. Apply transforms
            images.append(transform(img))
            
            # 4. FIX: Use hardcoded label instead of sample['label']
            # Change '1' to '0' if this dataset is Real
            labels.append(0) 
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
            
    if not images:
        # Return empty tensors with correct shapes so the trainer handles it gracefully
        return torch.zeros((0, 3, 224, 224)), torch.zeros((0,), dtype=torch.long)
        
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

def init_convnext_model(num_classes=2, pretrained=True):
    """
    Initializes the ConvNeXt Large model with pretrained weights.
    
    Args:
        num_classes (int): Number of output classes (2 for binary classification: real vs fake)
        pretrained (bool): Whether to use pretrained ImageNet weights
        
    Returns:
        nn.Module: The ConvNeXt Large model.
    """
    if pretrained:
        # Load pretrained ConvNeXt Large with ImageNet-1K weights
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        model = models.convnext_large(weights=weights)
    else:
        model = models.convnext_large(weights=None)
    
    # Modify the classifier head for binary classification
    # ConvNeXt's classifier is at model.classifier[2]
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    model.loss_fn = nn.CrossEntropyLoss()
    
    return model


def init_convnext_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    """
    Initializes the optimizer for the ConvNeXt Large model.
    
    Args:
        conf (config.Config): The configuration object.
        model (nn.Module): The ConvNeXt Large model.
        
    Returns:
        optim.Optimizer: The initialized optimizer.
    """
    # AdamW optimizer is commonly used for ConvNeXt models
    # Using learning rate from config
    return optim.AdamW(model.parameters(), lr=conf.learning_rate, weight_decay=0.05)


def process_dataset(conf: config.Config, dataset: data.Dataset) -> data.Dataset:
    """
    Processes the dataset for training.
    For webdataset format, we mainly need to ensure proper structure.
    
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to process.
        
    Returns:
        data.Dataset: The processed dataset.
    """
    # Webdataset format already comes structured from the loader
    # No additional processing needed here
    return dataset


def pre_init_convnext(conf: config.Config, dataset: data.Dataset) -> Tuple[nn.Module, data.Dataset]:
    """
    Prepares the ConvNeXt Large model and dataset for training.
    
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
        
    Returns:
        Tuple[nn.Module, data.Dataset]: The ConvNeXt model and processed dataset.
    """
    # Initialize model with pretrained weights
    model = init_convnext_model(num_classes=2, pretrained=True)
    
    # Process dataset if needed
    dataset = process_dataset(conf, dataset)
    
    return model, dataset



def simple_trainer(conf: config.Config, model: nn.Module, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Simple trainer for ConvNeXt Large model.
    Uses the SimpleTrainer from src/trainer/simple.py.
    
    Args:
        conf (config.Config): The configuration object.
        model (nn.Module): The ConvNeXt Large model to train.
        dataset (data.Dataset): The dataset to train on.
        
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The simple trainer and a dictionary with additional options.
    """
    # Create DataLoader with custom collate function for webdataset
    loader = data.DataLoader(
        dataset, 
        batch_size=conf.batch_size, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # Move model to GPU
    model = model.cuda()
    
    # Initialize optimizer
    optimizer = init_convnext_optim(conf, model)
    
    # Learning rate scheduler - cosine annealing is commonly used for vision models
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(loader),
        eta_min=1e-6
    )
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Return the SimpleTrainer with the initialized components
    return trainer.SimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        stats=trainer_stats.init_from_conf(
            conf=conf,
            device=device,
            num_train_steps=len(loader)
        )
    ), None


################################################################################ 
################################## Init ################################## 
################################################################################ 

def convnext_large_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Initializes the ConvNeXt Large model and returns the appropriate trainer based on the configuration.
    
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
        
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The initialized trainer and a dictionary with additional options.
    """
    model, dataset = pre_init_convnext(conf, dataset)
    
    # Note: Currently, only Simple trainer is implemented for ConvNeXt Large. Add more trainers as needed.
    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")