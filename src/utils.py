from .models.vlm import VisionLanguageModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

def load_model(model_name: str, **kwargs) -> VisionLanguageModel:
    """
    Factory function to load a vision-language model by name.
    """
    if model_name == "LLaVANeXT":
        from .models.llavanext import LLaVANeXT
        return LLaVANeXT(**kwargs)
    elif model_name == "Qwen3VL":
        from .models.qwen import Qwen3VL
        return Qwen3VL(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    pixel_values_videos = torch.stack([item["pixel_values_videos"] for item in batch])
    video_grid_thw = torch.stack([item["video_grid_thw"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

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
