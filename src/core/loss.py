import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, T: float = 4.0, lambda_kd: float = 1.0):
        super().__init__()
        self.T = T
        self.lambda_kd = lambda_kd
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, targets, teacher_logits=None):
        loss = self.ce(student_logits, targets)
        if teacher_logits is not None:
            ps = F.log_softmax(student_logits / self.T, dim=1)
            pt = F.softmax(teacher_logits / self.T, dim=1)
            kd = (self.T * self.T) * F.kl_div(ps, pt, reduction="batchmean")
            loss = loss + self.lambda_kd * kd
        return loss
