import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(logits, labels):

    CLASSES = ["Ventilation", "Stimulation", "Suction"]
    NUM_LABELS = len(CLASSES)
    
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    y_pred_bin = probs > 0.5
    y_true_bin = labels_np > 0.5

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average=None, zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )
    acc_m = accuracy_score(y_true_bin, y_pred_bin)
    
    metrics = {f"{CLASSES[i]}/precision": prec[i] for i in range(NUM_LABELS)}
    metrics.update({f"{CLASSES[i]}/recall": rec[i] for i in range(NUM_LABELS)})
    metrics.update({f"{CLASSES[i]}/f1": f1[i] for i in range(NUM_LABELS)})
    metrics.update({
        "macro/precision": prec_m,
        "macro/recall": rec_m,
        "macro/f1": f1_m,
        "macro/accuracy": acc_m,
    })
    
    return metrics