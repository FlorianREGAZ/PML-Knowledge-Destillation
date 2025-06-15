import torch.nn as nn
import torch.nn.functional as F

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

        return alpha * ce_loss + (1 - alpha) * kd_loss