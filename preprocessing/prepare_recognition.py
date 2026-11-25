import csv
import sys
from math import ceil, floor
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from preprocessing.image_processing import iter_yolo_boxes


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = DATA_DIR / "processed" / "recognition"


def export_split(split, destination):
    sources = DATA_DIR / "raw" / "images" / split
    labels = DATA_DIR / "processed" / "labels" / "labels" / split
    folder = destination / split
    folder.mkdir()
    count = 0
    with (destination / f"{split}.csv").open("x", encoding="utf-8", newline="") as manifest:
        writer = csv.writer(manifest)
        writer.writerow(("image", "plate_text"))
        for source in sorted(sources.iterdir()):
            if not source.is_file() or source.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            label = labels / f"{source.stem}.txt"
            with Image.open(source) as image:
                for line, (left, top, right, bottom) in iter_yolo_boxes(
                    label, *image.size, on_error=lambda message: print(message, file=sys.stderr)
                ):
                    bounds = (max(0, floor(left)), max(0, floor(top)),
                              min(image.width, ceil(right)), min(image.height, ceil(bottom)))
                    crop_path = folder / f"{source.stem}_{line:02d}.png"
                    with crop_path.open("xb") as crop:
                        image.crop(bounds).convert("RGB").save(crop, format="PNG")
                    writer.writerow((crop_path.relative_to(destination).as_posix(), ""))
                    count += 1
    return count


def main():
    if OUTPUT_DIR.exists():
        sys.exit(f"Exportul există deja: {OUTPUT_DIR}")
    with TemporaryDirectory(prefix=".recognition-", dir=OUTPUT_DIR.parent) as temporary:
        destination = Path(temporary) / "recognition"
        destination.mkdir()
        counts = {split: export_split(split, destination) for split in ("train", "val", "test")}
        if OUTPUT_DIR.exists():
            sys.exit(f"Exportul există deja: {OUTPUT_DIR}")
        destination.rename(OUTPUT_DIR)
    for split, count in counts.items():
        print(f"{split}: {count} decupaje")
    print(f"Export salvat în {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
