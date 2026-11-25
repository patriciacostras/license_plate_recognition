import argparse
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import DATA_DIR
from models.recognition_cnn import ALPHABET, RecognitionCNN, decode_beam, decode_greedy
from preprocessing.recognition_dataset import IMAGE_SIZE, RecognitionDataset


def edit_counts(target, predicted):
    """Count operations along one minimum-edit alignment."""
    rows = [[(column, 0, 0, column) for column in range(len(predicted) + 1)]]
    for index, expected in enumerate(target, 1):
        row = [(index, 0, index, 0)]
        for column, actual in enumerate(predicted, 1):
            cost, substitutions, deletions, insertions = rows[-1][column - 1]
            mismatch = expected != actual
            diagonal = (cost + mismatch, substitutions + mismatch, deletions, insertions)
            cost, substitutions, deletions, insertions = rows[-1][column]
            delete = (cost + 1, substitutions, deletions + 1, insertions)
            cost, substitutions, deletions, insertions = row[-1]
            insert = (cost + 1, substitutions, deletions, insertions + 1)
            row.append(min((diagonal, delete, insert), key=lambda entry: entry[0]))
        rows.append(row)
    return rows[-1][-1]


def prefix(text):
    return text[:next((i for i, character in enumerate(text) if character.isdigit()), len(text))]


@torch.inference_mode()
def recalibrate_batch_norm(model, dataset):
    """Refresh BatchNorm buffers from clean training inputs."""
    model.eval()
    layers = [layer for layer in model.modules() if isinstance(layer, nn.BatchNorm2d)]
    original_momenta = [layer.momentum for layer in layers]
    for layer in layers:
        layer.reset_running_stats()
        layer.train()

    seen = 0
    try:
        for images in DataLoader(dataset, batch_size=32, collate_fn=collate_images):
            seen += len(images)
            for layer in layers:
                layer.momentum = len(images) / seen
            model.features(images)
    finally:
        for layer, momentum in zip(layers, original_momenta):
            layer.momentum = momentum
        model.eval()


def collate_images(samples):
    return torch.stack([image for image, _, _ in samples])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Inspectează erorile OCR pe decupajele din val.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--details", action="store_true", help="Afișează rezultatele fiecărui decupaj")
    parser.add_argument("--beam-width", type=int, help="Păstrează mai multe secvențe candidate în decodarea CTC")
    parser.add_argument("--recalibrate-bn", action="store_true",
                        help="Actualizează statisticile BatchNorm din decupajele originale din train")
    args = parser.parse_args()
    if args.beam_width is not None and args.beam_width < 1:
        parser.error("--beam-width trebuie să fie pozitiv")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(checkpoint["image_size"]) != IMAGE_SIZE or checkpoint["alphabet"] != ALPHABET:
        parser.error("dimensiunea imaginilor sau alfabetul nu corespund modelului")
    model = RecognitionCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    manifests = DATA_DIR / "processed" / "recognition"
    if args.recalibrate_bn:
        training = RecognitionDataset(manifests / "train.csv")
        if not training:
            parser.error("train.csv nu conține decupaje transcrise pentru recalibrare")
        recalibrate_batch_norm(model, training)
        print(f"BatchNorm recalibrat pe {len(training)} decupaje de antrenare.")
    dataset = RecognitionDataset(manifests / "val.csv")
    if not dataset:
        parser.error("val.csv nu conține transcrieri")

    exact = empty = short = characters = false_tm = 0
    predicted_prefixes = Counter()
    expected_prefixes = Counter()
    operations = Counter()
    for index, (path, _, _) in enumerate(dataset.samples):
        image, target, _ = dataset[index]
        logits = model(image.unsqueeze(0))
        predictions = decode_greedy(logits) if args.beam_width is None else decode_beam(logits, args.beam_width)
        predicted, = predictions
        _, substitutions, deletions, insertions = edit_counts(target, predicted)
        operations.update({"înlocuiri": substitutions, "omisiuni": deletions, "adăugiri": insertions})
        exact += predicted == target
        empty += not predicted
        short += len(predicted) < len(target)
        characters += len(target)
        actual_prefix, target_prefix = prefix(predicted), prefix(target)
        predicted_prefixes[actual_prefix] += 1
        expected_prefixes[target_prefix] += 1
        false_tm += actual_prefix == "TM" and target_prefix != "TM"
        if args.details:
            print(f"{path.name} | așteptat {target} | prezis {predicted or '(gol)'}")

    print(f"Val | numere corecte integral: {exact}/{len(dataset)} | CER: {sum(operations.values()) / characters:.1%}")
    print(f"Predicții mai scurte decât eticheta: {short}/{len(dataset)}, dintre care goale: {empty}")
    print("Operații de editare: " + ", ".join(f"{name} {count}" for name, count in operations.items()))
    print(f"Prefix TM | așteptat: {expected_prefixes['TM']} | prezis: {predicted_prefixes['TM']}")
    print(f"Prefix TM atribuit altor județe: {false_tm}/{len(dataset) - expected_prefixes['TM']}")


if __name__ == "__main__":
    main()
