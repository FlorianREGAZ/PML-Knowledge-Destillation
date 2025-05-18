import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import ghostnetv3
from resnet import resnet50, resnet34, resnet18
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

    # Define distillation stages: (student_name, init_student_fn, teacher_name, init_teacher_fn)
    stages = [
        ('resnet34', lambda: resnet34(pretrained=True, device=device), 'resnet50', lambda: resnet50(pretrained=True, device=device)),
        ('resnet18', lambda: resnet18(pretrained=True, device=device), 'resnet34', lambda: resnet34(pretrained=True, device=device)),
        ('ghostnetv3', lambda: timm.create_model('ghostnetv3', width=1.0, num_classes=10), 'resnet18', lambda: resnet18(pretrained=True, device=device)),
    ]
    prev_checkpoint = None
    for student_name, init_student, teacher_name, init_teacher in stages:
        # Initialize student and teacher
        student = init_student()
        student.to(device)
        teacher = init_teacher()
        teacher.to(device)

        if prev_checkpoint:
            # load previous assistant as teacher
            teacher.load_state_dict(torch.load(prev_checkpoint, map_location=device))
        teacher.eval()

        logging.info(f"Stage: {teacher_name} -> {student_name}")
        criterion = DistillationLoss(temperature=1.0, alpha=0.5)
        optimizer = get_optimizer(student)
        scheduler = get_scheduler(optimizer)
        ema = get_ema(student)
        ema.to(device)

        best_acc = 0.0
        ckpt_path = f"assistant_{student_name}.pth"

        for epoch in range(1, EPOCHS + 1):
            train(student, device, trainloader, criterion, optimizer, ema, epoch, teacher_model=teacher)
            acc = evaluate(student, device, testloader, nn.CrossEntropyLoss(), ema)
            scheduler.step()
            if acc > best_acc:
                best_acc = acc
                torch.save(student.state_dict(), ckpt_path)
                logging.info(f'New best accuracy for {student_name}: {best_acc:.2f}%, model saved to {ckpt_path}')
        prev_checkpoint = ckpt_path

    logging.info(f'All distillation stages complete.')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
