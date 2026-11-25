# License Plate Recognition

Proiect experimental de recunoaștere a plăcuțelor de înmatriculare. Un CNN localizează plăcuța, iar un CNN cu GRU bidirecțional și CTC transcrie decupajul. Ambele modele se antrenează de la zero.

Scopul demo-ului este să proceseze un fișier video înregistrat și să salveze un MP4 cu predicții suprapuse peste cadre.

## Instalare

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

Fotografiile, adnotările și checkpointurile se păstrează local. Scripturile de detecție citesc imaginile din `data/raw/images/{train,val,test}` și adnotările YOLO din `data/processed/labels/labels/{train,val,test}`.

## Antrenare și inferență

Pentru detector:

```bash
.venv/bin/python -m training.train_detection --epochs 5 --size 640 384 --output models/checkpoints/detector-640.pt
```

Pentru OCR, generează decupajele o singură dată cu `.venv/bin/python -m preprocessing.prepare_recognition`. Completează manual coloana `plate_text` în `data/processed/recognition/train.csv` și `val.csv`, apoi rulează:

```bash
.venv/bin/python -m training.train_recognition --steps 1800 --batch-size 32 --augment --balance-tm --validate --output models/checkpoints/ocr-augmentat.pt
```

Cu cele două checkpointuri disponibile, procesează un clip:

```bash
.venv/bin/python main.py --video clip.mov --output clip-adnotat.mp4
```

Alege un nume nou pentru fiecare fișier de ieșire. MP4-ul adnotat conține pista video.

## Rezultate experimentale

| Experiment | Rezultat pe validare |
| --- | --- |
| Detector la 320 × 192, antrenat pe CPU | 10/67 la IoU ≥ 0,5 |
| Detector la 640 × 384, antrenat pe CPU | 16/67 la IoU ≥ 0,5 |
| Detector la 640 × 384, cu GIoU | 13/67 la IoU ≥ 0,5 |
| OCR cu ponderare după prefix | 0/34 numere complete, CER 74,6% |
| OCR cu ponderare și augmentare | 0/34 numere complete, CER 73,3% |
| Același checkpoint OCR, cu CTC beam search (10 candidați) | 0/34 numere complete, CER 72,5% |
| OCR cu 231 de decupaje, ponderare și augmentare | 0/34 numere complete, CER 71,6% |
| Același checkpoint OCR cu 231 de decupaje, cu CTC beam search (10 candidați) | 0/34 numere complete, CER 70,8% |
| OCR cu 231 de decupaje și intensitatea augmentării 0,5 | 0/34 numere complete, CER 75,4% (75,0% cu beam search) |

Fluxul video a fost verificat pe un clip de probă generat dintr-o fotografie. Evaluarea pe un clip înregistrat rămâne de făcut.
