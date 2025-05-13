import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import ghostnetv3
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
    torch.manual_seed(0)
    device = get_device()
    logging.info(f'Using device: {device}')
    trainloader, testloader = get_dataset_loader()

    # Student model: GhostNetV3
    student = timm.create_model('ghostnetv3', width=1.0, num_classes=10).to(device)

    # Teacher model: ResNet-50
    teacher = timm.create_model('resnet50', pretrained=True, num_classes=10).to(device) # TODO: have pretained resnet on cifar10
    teacher.eval()  # no gradients for teacher

    # Loss, optimizer, scheduler
    criterion = DistillationLoss(temperature=1.0, alpha=0.5)
    optimizer = get_optimizer(student)
    scheduler = get_scheduler(optimizer)
    ema = get_ema(student).to(device)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train(student, device, trainloader, criterion, optimizer, ema, epoch, teacher_model=teacher)
        acc = evaluate(student, device, testloader, nn.CrossEntropyLoss(), ema)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), 'distilled_ghostnetv3_cifar10.pth')
            logging.info(f'New best accuracy: {best_acc:.2f}%, model saved.')

    logging.info(f'Training complete. Best Test Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
