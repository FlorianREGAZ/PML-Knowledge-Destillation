import logging
import os

import torch
import torch.nn as nn

from models.resnet import resnet18
from utils import (
    train,
    evaluate,
    get_device,
    get_dataset_loader,
    get_optimizer,
    get_scheduler,
    EPOCHS
)

def main():
    print("Starting default training.")
    torch.manual_seed(0)

    device = get_device()
    logging.info(f'Using device: {device}')

    trainloader, testloader = get_dataset_loader()

    # Model: ResNet18
    model = resnet18(device=device)
    model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, training_length=len(trainloader))

    # Resume from checkpoint if exists
    checkpoint_path = 'default_resnet18_cifar10_checkpoint.pth'
    start_epoch = 1
    best_acc = 0.0
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']}, best_acc {best_acc:.2f}%)")

    for epoch in range(start_epoch, EPOCHS + 1):
        train(model, device, trainloader, criterion, optimizer, scheduler, epoch)
        acc = evaluate(model, device, testloader, criterion)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'default_ghostnetv3_cifar10.pth')
            logging.info(f'New best accuracy: {best_acc:.2f}%, model saved.')

        # Save checkpoint to resume training
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(checkpoint, checkpoint_path)

    logging.info(f'Training complete. Best Test Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
