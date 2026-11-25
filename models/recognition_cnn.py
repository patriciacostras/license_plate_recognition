from torch import nn


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 0


class RecognitionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.sequence = nn.GRU(64 * 4, 64, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(128, len(ALPHABET) + 1)

    def forward(self, images):
        features = self.features(images)
        columns = features.permute(0, 3, 1, 2).flatten(2)
        sequence, _ = self.sequence(columns)
        return self.classifier(sequence)


def decode_greedy(logits):
    texts = []
    for sequence in logits.argmax(dim=-1).tolist():
        text = []
        previous = BLANK
        for token in sequence:
            if token != BLANK and token != previous:
                text.append(ALPHABET[token - 1])
            previous = token
        texts.append("".join(text))
    return texts
