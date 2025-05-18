import logging

import torch
import torch.nn as nn
import timm

import models.ghostnetv3 as ghostnetv3
from utils import (
    train,
    evaluate,
    get_device,
    get_dataset_loader,
    get_optimizer,
    get_scheduler,
    get_ema,
    EPOCHS
)

def main():
    torch.manual_seed(0)

    device = get_device()
    logging.info(f'Using device: {device}')

    trainloader, testloader = get_dataset_loader()

    # Model: GhostNetV3 pretrained
    model = timm.create_model('ghostnetv3', width=1.0, num_classes=10)
    checkpoint = torch.load('default_ghostnetv3_cifar10.pth', weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    #model = torch.compile(model)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    ema = get_ema(model)
    ema.to(device)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train(model, device, trainloader, criterion, optimizer, ema, epoch)
        acc = evaluate(model, device, testloader, criterion, ema)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'default_ghostnetv3_cifar10.pth')
            logging.info(f'New best accuracy: {best_acc:.2f}%, model saved.')

    logging.info(f'Training complete. Best Test Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
