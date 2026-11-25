from collections import defaultdict
from math import exp, inf, log1p

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


def log_add(left, right):
    larger, smaller = max(left, right), min(left, right)
    if larger == -inf:
        return larger
    return larger + log1p(exp(smaller - larger))


def decode_beam(logits, beam_width=10):
    """Rank CTC prefixes by summed probability across their alignments."""
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    texts = []
    for sequence in logits.log_softmax(dim=-1).tolist():
        beam = [((), (0.0, -inf))]
        for frame in sequence:
            candidates = defaultdict(lambda: [-inf, -inf])
            for prefix, (blank, nonblank) in beam:
                total = log_add(blank, nonblank)
                same = candidates[prefix]
                same[0] = log_add(same[0], total + frame[BLANK])
                for token, score in enumerate(frame[1:], start=1):
                    extended = candidates[prefix + (token,)]
                    if prefix and token == prefix[-1]:
                        # A repeated token extends the text only after a blank.
                        same[1] = log_add(same[1], nonblank + score)
                        extended[1] = log_add(extended[1], blank + score)
                    else:
                        extended[1] = log_add(extended[1], total + score)
            beam = sorted(candidates.items(), key=lambda entry: log_add(*entry[1]), reverse=True)[:beam_width]
        texts.append("".join(ALPHABET[token - 1] for token in beam[0][0]))
    return texts
