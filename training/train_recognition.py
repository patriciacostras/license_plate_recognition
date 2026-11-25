import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from config import DATA_DIR, DEVICE, LEARNING_RATE, RANDOM_SEED
from models.recognition_cnn import ALPHABET, BLANK, RecognitionCNN, decode_greedy
from preprocessing.recognition_dataset import IMAGE_SIZE, RecognitionDataset


@torch.inference_mode()
def count_matches(model, images, texts):
    model.eval()
    return sum(predicted == target for predicted, target in zip(decode_greedy(model(images)), texts))


def edit_distance(predicted, target):
    previous = list(range(len(target) + 1))
    for index, character in enumerate(predicted, 1):
        current = [index]
        for column, expected in enumerate(target, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (character != expected)))
        previous = current
    return previous[-1]


@torch.inference_mode()
def validate(model, dataset, training_texts, device):
    model.eval()
    seen = set(training_texts)
    matches = errors = characters = evaluated = excluded = 0
    for index, (_, target, _) in enumerate(dataset.samples):
        if target in seen:
            excluded += 1
            continue
        image, _, _ = dataset[index]
        predicted, = decode_greedy(model(image.unsqueeze(0).to(device)))
        matches += predicted == target
        errors += edit_distance(predicted, target)
        characters += len(target)
        evaluated += 1
    if not evaluated:
        raise ValueError("val.csv nu conține transcrieri distincte de train.csv")
    print(f"Val | numere corecte integral: {matches}/{evaluated} ({matches / evaluated:.1%}) | "
          f"CER: {errors / characters:.1%} | excluse pentru suprapunere cu train: {excluded}")


def augment_plates(images):
    count, _, height, width = images.shape
    random = torch.rand(count, 5, device=images.device)
    transform = images.new_zeros(count, 2, 3)
    transform[:, 0, 0] = transform[:, 1, 1] = 1.04 + 0.04 * random[:, 0]
    transform[:, 0, 2] = (2 * random[:, 1] - 1) * (3 / width)
    transform[:, 1, 2] = (2 * random[:, 2] - 1) * (1 / height)
    grid = F.affine_grid(transform, images.shape, align_corners=False)
    shifted = F.grid_sample(images, grid, align_corners=False)
    contrast = (0.8 + 0.4 * random[:, 3]).view(-1, 1, 1, 1)
    brightness = (0.16 * random[:, 4] - 0.08).view(-1, 1, 1, 1)
    return ((shifted - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Probă OCR cu transcrierile completate din train.csv.")
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--validate", action="store_true", help="Evaluează la final pe etichetele din val.csv")
    parser.add_argument("--balance-tm", action="store_true",
                        help="Acordă ponderi totale egale exemplelor cu și fără prefixul TM")
    parser.add_argument("--batch-size", type=int, help="Număr de imagini pe pas")
    parser.add_argument("--augment", action="store_true", help="Aplică variații ușoare decupajelor de train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("--steps și --batch-size trebuie să fie pozitive")
    if args.output.exists() or args.output.resolve().is_relative_to(DATA_DIR.resolve()):
        parser.error("alege un checkpoint nou, în afara folderului data")

    torch.manual_seed(RANDOM_SEED)
    # Keep this first experiment on CPU, including CTC and its backward pass.
    device = torch.device("cpu") if DEVICE.type == "mps" else DEVICE
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Dispozitiv pentru OCR: {device}", flush=True)
    manifest = DATA_DIR / "processed" / "recognition" / "train.csv"
    dataset = RecognitionDataset(manifest)
    if not dataset:
        parser.error("train.csv nu conține transcrieri")
    images, texts, targets = zip(*(dataset[index] for index in range(len(dataset))))
    images = torch.stack(images).to(device)
    batch_size = min(args.batch_size or len(texts), len(texts))
    print(f"Imagini pe pas: {batch_size}/{len(texts)} | Augmentare: {args.augment}", flush=True)
    target_lengths = torch.tensor([len(target) for target in targets])
    frame_count = IMAGE_SIZE[0] // 4
    for text in texts:
        repeats = sum(left == right for left, right in zip(text, text[1:]))
        if len(text) + repeats > frame_count:
            parser.error("o transcriere este prea lungă pentru ieșirea CTC")
    combined_targets = torch.cat(targets).to(device)
    input_lengths = torch.full((len(texts),), frame_count, dtype=torch.long)

    model = RecognitionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=BLANK, reduction="none" if args.balance_tm else "mean")
    if args.balance_tm:
        tm = torch.tensor([text.startswith("TM") for text in texts], device=device)
        tm_count = int(tm.sum().item())
        other_count = len(texts) - tm_count
        if not tm_count or not other_count:
            parser.error("--balance-tm necesită exemple cu și fără prefixul TM")
        sample_weights = torch.where(tm, 0.5 / tm_count, 0.5 / other_count)
        lengths_on_device = target_lengths.to(device)
        print(f"Ponderare CTC: {tm_count} cu TM, {other_count} cu alte prefixe", flush=True)
    print(f"Potriviri exacte înainte de antrenare: {count_matches(model, images, texts)}/{len(texts)}",
          flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        if batch_size < len(texts):
            indices = torch.randperm(len(texts))[:batch_size]
            batch_images = images[indices.to(device)]
            batch_targets = torch.cat([targets[index] for index in indices.tolist()]).to(device)
            batch_target_lengths = target_lengths[indices]
        else:
            batch_images, batch_targets, batch_target_lengths = images, combined_targets, target_lengths
        if args.augment:
            batch_images = augment_plates(batch_images)
        log_probabilities = model(batch_images).log_softmax(dim=-1).transpose(0, 1)
        loss = criterion(log_probabilities, batch_targets, input_lengths[:batch_size], batch_target_lengths)
        if args.balance_tm:
            if batch_size < len(texts):
                selected_weights = sample_weights[indices.to(device)] * (len(texts) / batch_size)
                loss = ((loss / batch_target_lengths.to(device)) * selected_weights).sum()
            else:
                loss = ((loss / lengths_on_device) * sample_weights).sum()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            matches = count_matches(model, images, texts)
            print(f"Pasul {step}/{args.steps} | loss {loss.item():.4f} | lot train {matches}/{len(texts)}",
                  flush=True)

    if args.validate:
        validate(model, RecognitionDataset(manifest.with_name("val.csv")), texts, device)

    with args.output.open("xb") as output:
        torch.save({
            "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "image_size": IMAGE_SIZE,
            "alphabet": ALPHABET,
            "balance_tm": args.balance_tm,
            "batch_size": batch_size,
            "augment": args.augment,
        }, output)
    print(f"Checkpoint salvat: {args.output}")


if __name__ == "__main__":
    main()
