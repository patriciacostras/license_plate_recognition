import argparse
from math import ceil, floor
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DATA_DIR, DEVICE
from models.detection_cnn import DetectionCNN, best_boxes
from preprocessing.detection_dataset import SinglePlateDataset
from training.train_detection import box_ious


def pixel_box(box, width, height):
    x, y, w, h = box.tolist()
    return ((x - w / 2) * width, (y - h / 2) * height,
            (x + w / 2) * width, (y + h / 2) * height)


def make_preview(dataset, samples):
    sheet = Image.new("RGB", (1200, 345 * ((len(samples) + 2) // 3) + 32), "#20242a")
    header = ImageDraw.Draw(sheet)
    header.text((12, 8), "Verde: adnotare   |   Portocaliu: predicție", fill="white")
    for tile, (index, predicted, target, iou) in enumerate(samples):
        path, _ = dataset.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail((384, 308))
        draw = ImageDraw.Draw(image)
        draw.rectangle(pixel_box(target, *image.size), outline="#28ff65", width=4)
        draw.rectangle(pixel_box(predicted, *image.size), outline="#ff9f32", width=2)
        x, y = (tile % 3) * 400 + 8, (tile // 3) * 345 + 32
        sheet.paste(image, (x, y))
        header.text((x, y + 314), f"{path.stem}  |  IoU {iou:.3g}", fill="white")
    return sheet


def padded_box(box, margin):
    return torch.cat((box[:2], box[2:] * (1 + 2 * margin)))


def contains_plate(predicted, target, margin):
    crop = pixel_box(padded_box(predicted, margin), 1, 1)
    plate = pixel_box(target, 1, 1)
    return crop[0] <= plate[0] and crop[1] <= plate[1] and crop[2] >= plate[2] and crop[3] >= plate[3]


def make_crop_preview(dataset, samples):
    sheet = Image.new("RGB", (930, 35 + len(samples) * 130), "#20242a")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Adnotare", "Detectare", "Detectare + marja de 30%")):
        draw.text((column * 310 + 10, 10), title, fill="white", font_size=16)
    for row, (index, predicted, target, iou) in enumerate(samples):
        path, _ = dataset.samples[index]
        y = 35 + row * 130
        draw.text((10, y), f"{path.stem} | IoU {iou:.3g}", fill="white")
        with Image.open(path) as source:
            image = source.convert("RGB")
        for column, box in enumerate((target, predicted, padded_box(predicted, 0.30))):
            left, top, right, bottom = pixel_box(box, *image.size)
            crop = image.crop((max(0, floor(left)), max(0, floor(top)),
                               min(image.width, ceil(right)), min(image.height, ceil(bottom))))
            sheet.paste(ImageOps.contain(crop, (290, 100)), (column * 310 + 10, y + 22))
    return sheet


@torch.inference_mode()
def benchmark(model, image):
    """Measure single-image inference, including decoding and CPU readback."""
    image = image.unsqueeze(0).to(DEVICE)
    for _ in range(10):
        best_boxes(model(image)).cpu()
    durations = []
    for _ in range(50):
        start = perf_counter()
        best_boxes(model(image)).cpu()
        durations.append(perf_counter() - start)
    return median(durations) * 1000


def main():
    parser = argparse.ArgumentParser(description="Compară predicțiile detectorului cu adnotările din val.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crops-output", type=Path, help="Salvează comparația decupajelor pentru OCR")
    parser.add_argument("--benchmark", action="store_true", help="Măsoară inferența pe o singură imagine")
    args = parser.parse_args()
    outputs = [args.output]
    if args.crops_output is not None:
        outputs.append(args.crops_output)
    resolved = {path.resolve() for path in outputs}
    if len(resolved) != len(outputs) or any(path.exists() or path.is_relative_to(DATA_DIR.resolve())
                                           for path in resolved):
        parser.error("alege fișiere de ieșire noi și distincte, în afara folderului data")

    if DEVICE.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = SinglePlateDataset(DATA_DIR, "val", tuple(checkpoint["image_size"]))
    if not dataset:
        parser.error("splitul val nu are imagini eligibile")
    model = DetectionCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    groups = [[], [], []]
    margins = (0, 0.15, 0.30)
    fully_covered = [0] * len(margins)
    outside = 0
    hits = 0
    index = 0
    with torch.inference_mode():
        for images, targets in DataLoader(dataset, batch_size=BATCH_SIZE):
            predictions = model(images.to(DEVICE)).cpu()
            for predicted, target, overlap in zip(best_boxes(predictions), targets, box_ious(predictions, targets)):
                iou = overlap.item()
                hits += iou >= 0.5
                for margin_index, margin in enumerate(margins):
                    fully_covered[margin_index] += contains_plate(predicted, target, margin)
                outside += bool(torch.any((predicted[:2] - target[:2]).abs() > target[2:] / 2))
                group = groups[0 if iou == 0 else 1 if iou < 0.5 else 2]
                group.append((index, predicted, target, iou))
                index += 1

    samples = [sample for group in groups for sample in group[:3]]
    preview = make_preview(dataset, samples)
    with args.output.open("xb") as output:
        preview.save(output, format="PNG")
    print(f"IoU >= 0.5: {hits}/{len(dataset)}")
    print(f"Fără suprapunere: {len(groups[0])}/{len(dataset)}")
    print(f"Centrul predicției în afara plăcuței adnotate: {outside}/{len(dataset)}")
    print(f"Previzualizare: {args.output}")
    for margin, count in zip(margins, fully_covered):
        print(f"Dreptunghi adnotat cuprins complet cu marjă {margin:.0%} pe fiecare latură: "
              f"{count}/{len(dataset)} ({count / len(dataset):.1%})")
    if args.crops_output is not None:
        with args.crops_output.open("xb") as output:
            make_crop_preview(dataset, samples).save(output, format="PNG")
        print(f"Planșă decupaje: {args.crops_output}")
    if args.benchmark:
        image, _ = dataset[0]
        width, height = dataset.size
        print(f"Inferență {width} × {height}, mediană din 50 de rulări: {benchmark(model, image):.2f} ms/imagine")


if __name__ == "__main__":
    main()
