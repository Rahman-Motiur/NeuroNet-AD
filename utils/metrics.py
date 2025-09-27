
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@torch.no_grad()
def classification_report_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    y = labels.cpu().numpy()
    acc = accuracy_score(y, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}

