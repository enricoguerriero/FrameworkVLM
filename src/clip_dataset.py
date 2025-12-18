import torch
from torch.utils.data import Dataset
import numpy as np
import av
import pandas as pd

class ClipDataset(Dataset):

    def __init__(self, video_csv: str, prompt_template: str, processor, num_frames: int):
        super().__init__()

        self.processor = processor
        self.prompt = processor.apply_chat_template(prompt_template)
        self.data = pd.read_csv(video_csv)
        self.labels = self._build_labels(self.data)        
        self.videos, self.indices = self._prepare_videos(self.data['video_path'].tolist(), num_frames)
        self.num_frames = num_frames
        if self.num_frames is None:
            self.num_frames = max([len(idxs) for idxs in self.indices])

    @staticmethod
    def _build_labels(df):
        vent = df['ventilation'].astype(int).values
        stim = df['stimulation'].astype(int).values
        suct = df['suction'].astype(int).values
        arr = np.stack([vent, stim, suct], axis=1)
        return torch.tensor(arr, dtype=torch.float32)
    
    @classmethod
    def _get_label_counts(cls, df):
        n = len(df)
        vent_count = df['ventilation'].sum()
        stim_count = df['stimulation'].sum()
        suct_count = df['suction'].sum()
        counts = torch.tensor([vent_count, stim_count, suct_count], dtype=torch.float32)
        return counts, n
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _prepare_videos(paths, num_frames):
        idxs, out_paths = [], []
        for p in paths:
            container = av.open(p)
            total = container.streams.video[0].frames
            container.close()
            if num_frames is None or num_frames >= total:
                indices = np.arange(total)
            else:
                indices = np.linspace(0, total - 1, num_frames, dtype=int)
            idxs.append(indices)
            out_paths.append(p)
        return out_paths, idxs
    
    def _read_frames_at_indices(self, filepath, indices):
        container = av.open(filepath)
        frames = []
        for i, frm in enumerate(container.decode(video=0)):
            if i > indices[-1]:
                break
            if i in indices:
                frames.append(frm.to_ndarray(format="rgb24"))
        container.close()
        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))
        return np.stack(frames)
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        video_path = self.videos[idx]
        frame_indices = self.indices[idx]
        frames = self._read_frames_at_indices(video_path, frame_indices)
        inputs = self.processor(
            videos=frames,
            text=self.prompt,
            return_tensors="pt",
        )
        item = {
            "pixel_values_videos": inputs.pixel_values_videos.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": self.labels[idx],
        }
        return item
    
    def compute_pos_weights(self):
        counts, n = self._get_label_counts(self.df)
        return torch.tensor((n-counts) / (counts + 1e-6), dtype=torch.float)

    def compute_bias(self):
        counts, n = self._get_label_counts(self.df)
        ratios = counts / (n-counts + 1e-6)
        return torch.log(ratios)