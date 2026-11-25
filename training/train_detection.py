import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DATA_DIR, DEVICE, LEARNING_RATE, RANDOM_SEED
from models.detection_cnn import DetectionCNN, best_boxes
from preprocessing.detection_dataset import SinglePlateDataset


def detection_loss(predictions, boxes, box_loss):
    batch_size, _, rows, columns = predictions.shape
    scale = boxes.new_tensor([columns, rows])
    centers = boxes[:, :2] * scale
    grid_x = centers[:, 0].long().clamp(max=columns - 1)
    grid_y = centers[:, 1].long().clamp(max=rows - 1)
    batch = torch.arange(batch_size, device=boxes.device)
    objectness = torch.zeros_like(predictions[:, 0])
    objectness[batch, grid_y, grid_x] = 1
    logits = predictions[:, 0]
    confidence_loss = F.binary_cross_entropy_with_logits(logits, objectness, reduction="none")
    positive = objectness.bool()
    confidence_loss = confidence_loss[positive].mean() + confidence_loss[~positive].mean()

    offsets = centers - torch.stack((grid_x, grid_y), dim=1)
    targets = torch.cat((offsets, boxes[:, 2:]), dim=1)
    predicted_boxes = predictions.permute(0, 2, 3, 1)[batch, grid_y, grid_x, 1:].sigmoid()
    regression_loss = 5 * F.mse_loss(predicted_boxes, targets)
    if box_loss == "giou":
        grid = torch.stack((grid_x, grid_y), dim=1)
        predicted_centers = (predicted_boxes[:, :2] + grid) / scale
        predicted = torch.cat((predicted_centers, predicted_boxes[:, 2:]), dim=1)
        regression_loss = regression_loss + (1 - box_overlaps(predicted, boxes, generalized=True)).mean()
    return confidence_loss + regression_loss


def box_overlaps(predicted, boxes, *, generalized=False):
    predicted_min = predicted[:, :2] - predicted[:, 2:] / 2
    predicted_max = predicted[:, :2] + predicted[:, 2:] / 2
    target_min = boxes[:, :2] - boxes[:, 2:] / 2
    target_max = boxes[:, :2] + boxes[:, 2:] / 2
    intersection = (
        torch.minimum(predicted_max, target_max) - torch.maximum(predicted_min, target_min)
    ).clamp(min=0).prod(dim=1)
    union = predicted[:, 2:].prod(dim=1) + boxes[:, 2:].prod(dim=1) - intersection
    iou = intersection / union
    if generalized:
        enclosing = (
            torch.maximum(predicted_max, target_max) - torch.minimum(predicted_min, target_min)
        ).prod(dim=1)
        iou = iou - (enclosing - union) / enclosing
    return iou


def box_ious(predictions, boxes):
    return box_overlaps(best_boxes(predictions), boxes)


def train_full(model, optimizer, train, validation, epochs, box_loss):
    train_batches = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    validation_batches = DataLoader(validation, batch_size=BATCH_SIZE)
    best_score = (-1, -1)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, boxes in train_batches:
            images, boxes = images.to(DEVICE), boxes.to(DEVICE)
            optimizer.zero_grad()
            loss = detection_loss(model(images), boxes, box_loss)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)

        model.eval()
        overlaps = []
        with torch.inference_mode():
            for images, boxes in validation_batches:
                overlaps.append(box_ious(model(images.to(DEVICE)), boxes.to(DEVICE)).cpu())
        ious = torch.cat(overlaps)
        hits = int((ious >= 0.5).sum())
        score = (hits, ious.mean().item())
        print(
            f"Epoca {epoch}/{epochs} | train loss {loss_sum / len(train):.4f} | "
            f"val IoU mediu {score[1]:.3f} | val IoU >= 0.5: {hits}/{len(validation)} "
            f"({hits / len(validation):.1%})",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_weights)
    return {"epoch": best_epoch, "val_hits": best_score[0], "val_mean_iou": best_score[1]}


def train_demo(model, optimizer, dataset, steps, box_loss):
    if len(dataset) < 8:
        raise ValueError("sunt necesare cel puțin 8 imagini eligibile")
    indices = torch.randperm(len(dataset))[:8].tolist()
    images, boxes = zip(*(dataset[index] for index in indices))
    images, boxes = torch.stack(images).to(DEVICE), torch.stack(boxes).to(DEVICE)
    model.eval()
    with torch.no_grad():
        print(f"IoU mediu înainte de antrenare: {box_ious(model(images), boxes).mean().item():.3f}", flush=True)
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss = detection_loss(model(images), boxes, box_loss)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"Pasul {step}/{steps} | loss {loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        print(f"IoU mediu după antrenare: {box_ious(model(images), boxes).mean().item():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Antrenează detectorul de plăcuțe de la zero.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--steps", type=int, help="Pași de antrenare pe un lot fix de 8 imagini")
    mode.add_argument("--epochs", type=int, help="Epoci de antrenare pe întregul split eligibil")
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=(320, 192))
    parser.add_argument("--box-loss", choices=("mse", "giou"), default="mse",
                        help="giou adaugă o penalizare bazată pe suprapunerea dreptunghiurilor")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    steps = 150 if args.steps is None else args.steps
    if steps < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("--steps și --epochs trebuie să fie pozitive")
    if args.output.resolve().is_relative_to(DATA_DIR.resolve()):
        parser.error("alege o cale de ieșire în afara folderului data")
    if args.output.exists():
        parser.error(f"Fișierul există deja: {args.output}")

    torch.manual_seed(RANDOM_SEED)
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    size = tuple(args.size)
    if min(size) <= 0 or max(size) <= 16:
        parser.error("--size trebuie să producă o grilă cu cel puțin două celule")
    print(f"Rezoluție: {size[0]} × {size[1]} | Box loss: {args.box_loss}")
    train = SinglePlateDataset(DATA_DIR, "train", size)
    print(f"Train: {len(train)} imagini eligibile. Imagini omise: {train.skipped}", flush=True)
    model = DetectionCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    metadata = {}
    if args.epochs is None:
        train_demo(model, optimizer, train, steps, args.box_loss)
    else:
        validation = SinglePlateDataset(DATA_DIR, "val", size)
        print(f"Val: {len(validation)} imagini eligibile. Imagini omise: {validation.skipped}", flush=True)
        if not train or not validation:
            parser.error("spliturile train și val trebuie să aibă imagini eligibile")
        metadata = train_full(model, optimizer, train, validation, args.epochs, args.box_loss)
        print(f"Checkpoint ales din epoca {metadata['epoch']}.")

    with args.output.open("xb") as output:
        weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save({"model_state_dict": weights, "image_size": size, "box_loss": args.box_loss, **metadata}, output)
    print(f"Checkpoint salvat: {args.output}")


if __name__ == "__main__":
    main()
