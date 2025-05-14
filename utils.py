import logging
import os
from datetime import datetime

import torch
from tqdm import tqdm  # progress bar for training and evaluation loops
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch_ema import ExponentialMovingAverage
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

def sync_device(device):
    """Synchronize device operations for CUDA and MPS to avoid hangs."""
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

# Setup logger to console and file
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = datetime.now().strftime('train_%Y%m%d_%H%M%S.log')
log_path = os.path.join(log_dir, log_filename)

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path)
    ]
)

# Hyperparameters
NUM_WORKERS = 6
BATCH_SIZE = 512
EPOCHS = 800
LR = 0.00125
WEIGHT_DECAY = 0.05
EMA_DECAY = 0.9999


def train(student_model, device, loader, criterion, optimizer, ema, epoch, teacher_model=None):
    student_model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}", unit="batch")):
        # log progress at every batch
        logging.debug(f"Processing batch {batch_idx+1}/{len(loader)} for epoch {epoch}")
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

        student_outputs = student_model(inputs)
        if teacher_model is not None:
            loss = criterion(student_outputs, teacher_outputs, targets)
        else:
            loss = criterion(student_outputs, targets)
        
        loss.backward()
        optimizer.step()
        ema.update()
        # Synchronize after update to flush operations
        sync_device(device)

        total_loss += loss.item()
        _, predicted = student_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # Log more frequently to monitor progress
        if batch_idx % 10 == 0:
            logging.info(f'Epoch {epoch} | Step {batch_idx+1}/{len(loader)} | Loss: {total_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')
    # End of epoch logging
    logging.info(f'Epoch {epoch} training completed')

def evaluate(model, device, loader, criterion, ema):
    ema.store()
    ema.copy_to()
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating", unit="batch", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    logging.info(f'Test Loss: {total_loss/len(loader):.4f} | Test Acc: {acc:.2f}%')
    # Ensure all evaluation ops are completed
    sync_device(device)
    
    ema.restore()
    return acc

def get_optimizer(model):
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return optimizer

def get_scheduler(optimizer):
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    return scheduler

def get_ema(model, decay=EMA_DECAY):
    ema = ExponentialMovingAverage(model.parameters(), decay=EMA_DECAY)
    return ema

def get_dataset_loader():
    # Data transforms for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),  # random augmentation
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    # Dynamically set num_workers and pin_memory based on device
    device = get_device()
    num_workers = 0 if device == 'mps' else NUM_WORKERS
    pin_memory = True if device == 'cuda' else False

    # Datasets and loaders
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    return trainloader, testloader

def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'