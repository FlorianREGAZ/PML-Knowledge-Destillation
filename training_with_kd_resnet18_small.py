import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

import models.ghostnetv3_small as ghostnetv3_small
from models.resnet import resnet18
from loss.distillation_loss import DistillationLoss
from utils import (
    train,
    evaluate,
    get_device,
    get_dataset_loader,
    get_optimizer,
    get_scheduler,
    init_weights_kaiming,
    EPOCHS
)

def main():
    print("Starting KD training.")
    torch.manual_seed(0)
    device = get_device()
    logging.info(f'Using device: {device}')
    trainloader, testloader = get_dataset_loader()

    # Student model: GhostNetV3
    width = 1.0
    logging.info(f'Using GhostNetV3 with width {width}')
    student = timm.create_model('ghostnetv3_small', width=width, num_classes=10)
    init_weights_kaiming(student)
    student.to(device)

    # Teacher model: ResNet-18
    teacher = resnet18(pretrained=True, device=device).to(device)
    teacher.eval()

    # Loss, optimizer, scheduler
    criterion = DistillationLoss(temperature=1.0, alpha=0.5)
    optimizer = get_optimizer(student)
    scheduler = get_scheduler(optimizer, training_length=len(trainloader))

    # checkpoint loading
    checkpoint_path = 'kd_resnet18_ghostnetv3_cifar10_checkpoint.pth'
    start_epoch = 1
    best_acc = 0.0
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']}, best_acc {best_acc:.2f}%)")

    for epoch in range(start_epoch, EPOCHS + 1):
        train(student, device, trainloader, criterion, optimizer, scheduler, epoch, teacher_model=teacher)
        acc = evaluate(student, device, testloader, nn.CrossEntropyLoss())
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), 'kd_resnet_18_ghostnetv3_cifar10.pth')
            logging.info(f'New best accuracy: {best_acc:.2f}%, model saved.')

        # Save checkpoint to resume training
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
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
