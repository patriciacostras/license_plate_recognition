from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocessing.image_processing import read_yolo_boxes


def prepare_detector_image(image, size):
    resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255


class SinglePlateDataset(Dataset):
    """Select images with exactly one plate annotation for the first experiment."""

    def __init__(self, data_dir, split, size):
        data_dir = Path(data_dir)
        self.size = size
        self.samples = []
        self.skipped = {"empty": 0, "multiple": 0, "invalid": 0, "missing": 0}
        image_dir = data_dir / "raw" / "images" / split
        labels_dir = data_dir / "processed" / "labels" / "labels" / split
        for path in sorted(image_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            try:
                boxes = read_yolo_boxes(labels_dir / f"{path.stem}.txt", 1, 1)
            except FileNotFoundError:
                self.skipped["missing"] += 1
                continue
            except ValueError:
                self.skipped["invalid"] += 1
                continue
            if not boxes:
                self.skipped["empty"] += 1
            elif len(boxes) > 1:
                self.skipped["multiple"] += 1
            else:
                self.samples.append((path, boxes[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, (left, top, right, bottom) = self.samples[index]
        with Image.open(path) as source:
            pixels = prepare_detector_image(source, self.size)
        box = torch.tensor([(left + right) / 2, (top + bottom) / 2, right - left, bottom - top],
                           dtype=torch.float32)
        return pixels, box
