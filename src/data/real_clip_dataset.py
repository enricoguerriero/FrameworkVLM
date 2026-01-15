from .clip_dataset import ClipDataset
import numpy as np
import torch

class RealClipDataset(ClipDataset):
    
    def __init__(self, video_csv: str, prompt_template: list[dict], processor, num_frames: int = None):
        super().__init__(video_csv, prompt_template, processor, num_frames)

    @staticmethod
    def _build_labels(df):
        codes = df['Stimulation_Suction_Ventilations'].astype(str).str.zfill(3)
        stim = codes.str[0].astype(int)
        suct = codes.str[1].astype(int)
        vent = codes.str[2].astype(int)
        arr = np.stack([vent, stim, suct], axis=1)
        return torch.tensor(arr, dtype=torch.float32)
    
    @staticmethod
    def _get_label_counts(df):
        n = len(df)
        codes = df['Stimulation_Suction_Ventilations'].astype(str).str.zfill(3)
        vent_count = codes.str[2].astype(int).sum()
        stim_count = codes.str[0].astype(int).sum()
        suct_count = codes.str[1].astype(int).sum()
        counts = torch.tensor([vent_count, stim_count, suct_count], dtype=torch.float)
        return counts, n