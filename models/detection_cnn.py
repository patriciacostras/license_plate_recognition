import torch
from torch import nn


class DetectionCNN(nn.Module):
    """Predict objectness and a plate box at each grid cell."""

    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in (16, 24, 32, 48):
            layers.extend((
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_channels, 5, 1)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([-3.0, 0.0, 0.0, -3.0, -3.0]))

    def forward(self, images):
        return self.head(self.features(images))


def best_boxes(predictions):
    """Decode the highest-scoring cell as normalized center, width and height."""
    batch_size, _, rows, columns = predictions.shape
    batch = torch.arange(batch_size, device=predictions.device)
    best = predictions[:, 0].flatten(1).argmax(dim=1)
    grid_y, grid_x = best // columns, best % columns
    encoded = predictions.permute(0, 2, 3, 1)[batch, grid_y, grid_x, 1:].sigmoid()
    grid = torch.stack((grid_x, grid_y), dim=1)
    centers = (encoded[:, :2] + grid) / predictions.new_tensor([columns, rows])
    return torch.cat((centers, encoded[:, 2:]), dim=1)
