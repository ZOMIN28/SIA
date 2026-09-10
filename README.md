### 1.Dependency

Python=3.10.20 with:

```
Package            Version
matplotlib         3.10.9
numpy              1.26.4
opencv-python      4.13.0.92
pandas             2.3.3
pillow             10.2.0
scikit-learn       1.7.2
scipy              1.15.3
timm               1.0.26
tokenizers         0.20.3
torch              2.2.2+cu121
torchaudio         2.2.2+cu121
torchvision        0.17.2+cu121
tqdm               4.67.3
transformers       4.45.2
```

### 2. Reproducibility Statement

All results reported in this work can be fully reproduced using the provided code repository. The reproduction procedure is as follows:

Place the official dataset in the `dataset/` directory.

**Train the four baseline classification models** (FFT-ConvNeXt, DINOv2, CLIP, and Xception) with K-fold cross-validation by running:

```bash
python train.py
```

The trained model weights are saved under the `checkpoints/` directory.

**Train the Confusion-Guided Expert classifier** by running:

```bash
python train_bin.py
```

The trained expert model weights are also stored in `checkpoints/`.

**Obtain the K-fold ensemble logits on the test set** by running:

```bash
python infer.py
```

The resulting logits are saved in the `logits_result/` directory.

**Perform the final weighted ensemble**, activate the Confusion-Guided Expert classifier when applicable, and apply class-adaptive confidence calibration for Tencent Hunyuan by running:

```bash
python softvote.py
```

The final test set predictions are saved in the `results/` directory.