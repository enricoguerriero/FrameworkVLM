import torch

def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    collated = {
        "pixel_values_videos": torch.stack([item["pixel_values_videos"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }
    if "video_grid_thw" in batch[0]: # Qwen3VL case
        collated["video_grid_thw"] = torch.stack([item["video_grid_thw"] for item in batch])
    return collated