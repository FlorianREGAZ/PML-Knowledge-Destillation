import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

import models.ghostnetv3_small as ghostnetv3_small
from train_efficientnetv2 import BaseVisionSystem, config
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

class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, targets):
        T = self.temperature
        alpha = self.alpha

        # Cross-Entropy Loss
        ce_loss = F.cross_entropy(student_logits, targets)

        # Distillation loss
        kd_loss = self.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1)
        ) * (T * T)

        return (1 - alpha) * ce_loss + alpha * kd_loss

def main():
    print("Starting KD training.")
    torch.manual_seed(0)
    device = get_device()
    logging.info(f'Using device: {device}')
    trainloader, _ = get_dataset_loader()
    trainloader_resized, testloader = get_dataset_loader(resize=(224, 224))

    # Student model: GhostNetV3
    width = 2.8
    logging.info(f'Using GhostNetV3 with width={width}')
    student = timm.create_model('ghostnetv3_small', width=width, num_classes=10)
    init_weights_kaiming(student)
    student.to(device)

    # Teacher model: EfficientNetV2
    config['model']['num_classes'] = 10
    config['model']['num_step'] = 150000
    config['model']['max_epochs'] = 100
    #teacher = BaseVisionSystem(**config['model'])
    teacher = BaseVisionSystem.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=99-step=140700.ckpt", **config['model'])
    teacher.eval()
    teacher.to(device)

    # Loss, optimizer, scheduler
    criterion = DistillationLoss(temperature=1.0, alpha=0.5)
    optimizer = get_optimizer(student)
    scheduler = get_scheduler(optimizer, training_length=len(trainloader))

    # checkpoint loading
    checkpoint_path = 'kd_efficientnetv2_ghostnetv3_cifar10_checkpoint.pth'
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
            torch.save(student.state_dict(), 'kd_efficientnetv2_ghostnetv3_cifar10.pth')
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
