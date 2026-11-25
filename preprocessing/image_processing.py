import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def read_yolo_boxes(label_path, image_width, image_height):
    """Read normalized YOLO annotations as (left, top, right, bottom) in pixels."""
    return [box for _, box in iter_yolo_boxes(label_path, image_width, image_height)]


def iter_yolo_boxes(label_path, image_width, image_height, *, on_error=None):
    """Yield each annotation with its source line number."""
    for line_number, line in enumerate(Path(label_path).read_text().splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        try:
            class_id, *coordinates = fields
            x, y, width, height = map(float, coordinates)
            if class_id != "0":
                raise ValueError("clasa trebuie să fie 0")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError("centrul trebuie să fie în [0, 1], iar dimensiunile în (0, 1]")
        except ValueError as error:
            message = f"{label_path}:{line_number}: {error}"
            if on_error is None:
                raise ValueError(message) from error
            on_error(message)
            continue

        yield line_number, (
            (x - width / 2) * image_width,
            (y - height / 2) * image_height,
            (x + width / 2) * image_width,
            (y + height / 2) * image_height,
        )


def main():
    parser = argparse.ArgumentParser(description="Verifică vizual o adnotare YOLO.")
    parser.add_argument("split", choices=("train", "val", "test"))
    parser.add_argument("image", help="Numele imaginii fără extensie, de exemplu IMG_4475")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = Path(__file__).resolve().parents[1] / "data"
    image_dir = data / "raw" / "images" / args.split
    matches = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.stem == args.image and path.suffix.lower() in {".jpg", ".jpeg"}
    ]
    if len(matches) != 1:
        parser.exit(1, f"Imaginea {args.image} trebuie să aibă o singură potrivire în {image_dir}.\n")

    label_path = data / "processed" / "labels" / "labels" / args.split / f"{args.image}.txt"
    try:
        if args.output.resolve().is_relative_to(data.resolve()):
            raise ValueError("alege o cale de ieșire în afara folderului data")
        with Image.open(matches[0]) as source:
            image = source.convert("RGB")
        boxes = read_yolo_boxes(label_path, *image.size)
        drawing = ImageDraw.Draw(image)
        for box in boxes:
            drawing.rectangle(box, outline="#28ff65", width=max(2, image.width // 300))
        with args.output.open("xb") as output:
            image.save(output, format="PNG")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Dreptunghiuri: {len(boxes)}. Previzualizare: {args.output}")


if __name__ == "__main__":
    main()
