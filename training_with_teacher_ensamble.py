import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import models.ghostnetv3 as ghostnetv3
from models.resnet import resnet50
from models.densenet import densenet161
from models.vgg import vgg13_bn
from models.inception import inception_v3
from utils import (
    train,
    evaluate,
    get_device,
    get_dataset_loader,
    get_optimizer,
    get_scheduler,
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

# Add ensemble teacher wrapper
class EnsembleTeacher(nn.Module):
    def __init__(self, teachers):
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
    def forward(self, x):
        outputs = [t(x) for t in self.teachers]
        return sum(outputs) / len(outputs)

# Replace main with ensemble distillation
def main():
    print("Starting Teaching Ensamble training.")
    torch.manual_seed(0)
    device = get_device()
    logging.info(f'Using device: {device}')
    trainloader, testloader = get_dataset_loader()

    # Initialize student
    student = timm.create_model('ghostnetv3', width=1.0, num_classes=10)
    student.to(device)

    # Initialize teachers
    teacher_inits = [densenet161, vgg13_bn, resnet50, inception_v3]
    teachers = []
    for init in teacher_inits:
        tm = init(pretrained=True, num_classes=10)
        tm.to(device)
        tm.eval()
        teachers.append(tm)
    ensemble_teacher = EnsembleTeacher(teachers)
    ensemble_teacher.to(device)
    ensemble_teacher.eval()

    # Setup training utilities
    criterion = DistillationLoss(temperature=1.0, alpha=0.5)
    optimizer = get_optimizer(student)
    scheduler = get_scheduler(optimizer, training_length=len(trainloader))

    best_acc = 0.0
    ckpt_path = 'ensemble_ghostnetv3.pth'

    for epoch in range(1, EPOCHS + 1):
        train(student, device, trainloader, criterion, optimizer, scheduler, epoch, teacher_model=ensemble_teacher)
        acc = evaluate(student, device, testloader, nn.CrossEntropyLoss())
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), ckpt_path)
            logging.info(f'New best accuracy: {best_acc:.2f}%, model saved to {ckpt_path}')
    logging.info('Ensemble distillation complete.')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
