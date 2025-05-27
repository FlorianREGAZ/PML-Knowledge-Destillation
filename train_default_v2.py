import logging
import os

import torch
import torch.nn as nn
import timm

import models.ghostnetv3 as ghostnetv3
from scheduler.WarmupCosineLR import WarmupCosineLR
from utils import (
    train,
    evaluate,
    get_device,
    get_dataset_loader,
    EPOCHS
)

def init_weights_kaiming(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def main():
    print("Starting default training.")
    torch.manual_seed(0)

    device = get_device()
    logging.info(f'Using device: {device}')

    trainloader, testloader = get_dataset_loader()

    # Model: GhostNetV3
    model = timm.create_model('ghostnetv3', width=1.0, num_classes=10)
    init_weights_kaiming(model)
    model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        weight_decay=0.01,
        momentum=0.9,
        nesterov=True,
    )
    total_steps = EPOCHS * len(trainloader)
    scheduler = WarmupCosineLR(
        optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
    )

    # Resume from checkpoint if exists
    checkpoint_path = 'default_v2_ghostnetv3_cifar10_checkpoint.pth'
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
        acc = evaluate(model, device, testloader, criterion, None)

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
