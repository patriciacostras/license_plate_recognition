from math import ceil, floor, isfinite
from os import link
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import torch
from PIL import Image

from config import DEVICE
from models.detection_cnn import DetectionCNN, best_boxes
from models.recognition_cnn import ALPHABET, RecognitionCNN, decode_greedy
from preprocessing.detection_dataset import prepare_detector_image
from preprocessing.recognition_dataset import IMAGE_SIZE, prepare_plate


class PlateReader:
    def __init__(self, detector_path, ocr_path, min_score, margin):
        torch.set_num_threads(min(4, torch.get_num_threads()))
        detector = torch.load(detector_path, map_location="cpu", weights_only=True)
        ocr = torch.load(ocr_path, map_location="cpu", weights_only=True)
        if tuple(ocr["image_size"]) != IMAGE_SIZE or ocr["alphabet"] != ALPHABET:
            raise ValueError("Checkpointul OCR nu corespunde dimensiunii sau alfabetului modelului.")
        self.detector = DetectionCNN().to(DEVICE)
        self.detector.load_state_dict(detector["model_state_dict"])
        self.detector.eval()
        self.detector_size = tuple(detector["image_size"])
        self.ocr = RecognitionCNN()
        self.ocr.load_state_dict(ocr["model_state_dict"])
        self.ocr.eval()
        self.min_score = min_score
        self.margin = margin

    @torch.inference_mode()
    def read(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pixels = prepare_detector_image(image, self.detector_size)
        logits = self.detector(pixels.unsqueeze(0).to(DEVICE))
        score = logits[0, 0].max().sigmoid().item()
        if score < self.min_score:
            return None
        x, y, width, height = best_boxes(logits)[0].tolist()
        half_width, half_height = width * (0.5 + self.margin), height * (0.5 + self.margin)
        left = max(0, floor((x - half_width) * image.width))
        top = max(0, floor((y - half_height) * image.height))
        right = min(image.width, ceil((x + half_width) * image.width))
        bottom = min(image.height, ceil((y + half_height) * image.height))
        if right <= left or bottom <= top:
            return None
        bounds = (left, top, right, bottom)
        plate = prepare_plate(image.crop(bounds))
        text, = decode_greedy(self.ocr(plate.unsqueeze(0)))
        return bounds, text


def process_video(source, output, reader):
    if not source.is_file():
        raise ValueError(f"Fișierul video nu există: {source}")
    if output.suffix.lower() != ".mp4" or output.exists():
        raise ValueError("Alege o cale nouă cu extensia .mp4 pentru ieșire.")
    if not output.parent.is_dir():
        raise ValueError(f"Directorul de ieșire nu există: {output.parent}")

    capture = cv2.VideoCapture(str(source))
    frames = readings = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV nu poate deschide clipul: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        ok, frame = capture.read()
        if not ok or not isfinite(fps) or fps <= 0:
            raise ValueError("Clipul trebuie să conțină cadre și o rată FPS validă.")
        height, width = frame.shape[:2]
        padded_width, padded_height = width + width % 2, height + height % 2
        with TemporaryDirectory(dir=output.parent) as directory:
            temporary = Path(directory) / "video.mp4"
            writer = cv2.VideoWriter(
                str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (padded_width, padded_height)
            )
            try:
                if not writer.isOpened():
                    raise ValueError("OpenCV nu poate crea fișierul MP4.")
                while ok:
                    result = reader.read(frame)
                    if result is not None:
                        (left, top, right, bottom), text = result
                        color = (50, 235, 80)
                        cv2.rectangle(frame, (left, top), (right - 1, bottom - 1), color, 2)
                        caption = f"OCR: {text}" if text else "OCR: text necitit"
                        position = (left, max(18, top - 6))
                    else:
                        caption, position, color = "Placuta nedetectata", (12, 24), (230, 230, 230)
                        text = ""
                    text_width = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
                    position = (max(0, min(position[0], width - text_width - 4)), position[1])
                    cv2.putText(frame, caption, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                    cv2.putText(frame, caption, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                    if width != padded_width or height != padded_height:
                        frame = cv2.copyMakeBorder(frame, 0, padded_height - height,
                                                   0, padded_width - width, cv2.BORDER_CONSTANT)
                    writer.write(frame)
                    frames += 1
                    readings += bool(text)
                    ok, frame = capture.read()
            finally:
                writer.release()
            link(temporary, output)
    finally:
        capture.release()
    return frames, readings
