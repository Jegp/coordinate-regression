from pathlib import Path

import torch

class ShapeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        t: int = 40,
        train: bool = True,
        pose_offset: torch.Tensor = torch.Tensor([0, 0]),
        pose_delay: int = 0,
        frames_per_file: int = 128,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.files = list(Path(root).glob("*.dat"))
        split = int(0.8 * len(self.files))
        if train:
            self.files = self.files[:split]
        else:
            self.files = self.files[split:]
        self.t = t
        self.pose_delay = int(pose_delay)
        self.chunks = frames_per_file // (2 * t + self.pose_delay)
        self.pose_offset = pose_offset
        self.device = device
        assert len(self.files) > 0, f"No data files in given root '{root}'"

    def __getitem__(self, index):
        filename = self.files[index // self.chunks]
        frames, poses = torch.load(filename, map_location=self.device)
        frames = frames.to_dense()
        chunk = index % self.chunks
        start = chunk * self.t
        mid = start + self.t
        end = mid + self.t
        warmup_tensor = frames[start:mid].float()
        actual_tensor = frames[mid:end].float()
        delayed_poses = poses[mid + self.pose_delay : end + self.pose_delay]
        offset_poses = delayed_poses + self.pose_offset
        return warmup_tensor, actual_tensor, offset_poses.squeeze()

    def __len__(self):
        return len(self.files) * self.chunks
