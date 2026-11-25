import csv

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from models.recognition_cnn import ALPHABET


IMAGE_SIZE = (160, 32)
CHARACTER_IDS = {character: index for index, character in enumerate(ALPHABET, 1)}


def prepare_plate(image):
    plate = ImageOps.contain(image.convert("L"), IMAGE_SIZE, Image.Resampling.BILINEAR)
    canvas = Image.new("L", IMAGE_SIZE)
    canvas.paste(plate, (0, (IMAGE_SIZE[1] - plate.height) // 2))
    return torch.from_numpy(np.array(canvas)).unsqueeze(0).float() / 255


class RecognitionDataset(Dataset):
    def __init__(self, manifest):
        self.samples = []
        with manifest.open(encoding="utf-8", newline="") as file:
            for line, row in enumerate(csv.DictReader(file), 2):
                text = row["plate_text"].strip()
                if not text:
                    continue
                if any(character not in CHARACTER_IDS for character in text):
                    raise ValueError(f"{manifest}:{line}: textul trebuie să conțină doar A–Z și 0–9")
                encoded = torch.tensor([CHARACTER_IDS[character] for character in text], dtype=torch.long)
                self.samples.append((manifest.parent / row["image"], text, encoded))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, text, encoded = self.samples[index]
        with Image.open(path) as image:
            return prepare_plate(image), text, encoded
