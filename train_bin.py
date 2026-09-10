import pandas as pd

import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils.utils import sequential_kfold_split, extract_classes
from data.dataloader import build_loader
from data.transforms_bin import *
from model.fftmodel import NextModelFFT
from CFG import CFG


def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(loader):
        imgs = imgs.to(CFG.device)
        labels = labels.to(CFG.device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(imgs)
            # loss = F.cross_entropy(logits, labels)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(CFG.device)
        labels = labels.to(CFG.device)

        logits = model(imgs)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        total_loss += loss.item()

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
  

    return correct / total, total_loss / len(loader)


def train_model(model, train_loader, val_loader, save_path, model_type="FFT"):

    model = model.to(CFG.device)

    # FFT
    if model_type in ["fft", "FFT"]:
        optimizer = torch.optim.AdamW([
        {"params": model.fft_branch.parameters(), "lr": 3e-4},
        {"params": model.classifier.parameters(), "lr": 3e-4},
        {"params": model.backbone.parameters(), "lr": 1e-5}, # Next: 1e-5  Res: 2e-4
        ], weight_decay=1e-4)
    

    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0
    best_loss = 10000
    patience = 10
    counter = 0

    for epoch in range(CFG.epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        
        with torch.no_grad():
            val_acc, val_loss = evaluate(model, val_loader)
            print(f"\nEpoch {epoch}: Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

            if val_loss < best_loss:
            # if val_acc > best_acc:
                best_loss = val_loss
                # best_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), save_path)
                print("Saved Best Model")
            else:
                counter += 1
                if counter >= patience:
                    print("Early Stopping")
                    break
    
    print("Best Val Loss:", best_loss)
    # print("Best Val Acc:", best_acc)
    
    return model



if __name__ == "__main__":
    df = pd.read_csv(CFG.train_csv)
    folds = sequential_kfold_split(df, n_splits=5)


    for fold_id, (train_df, val_df) in enumerate(folds):
        print(f"\n===== Fold {fold_id} =====")

        # only keep class 6/7
        train_df, val_df = extract_classes(train_df, val_df, [6,7])

        # FFT
        train_loader_fft = build_loader(train_df, train_transform_fft_bin, True, batch_size=CFG.batch_size//2)
        val_loader_fft = build_loader(val_df, test_transform_fft_bin, False)
        train_model(
            NextModelFFT(num_classes=2, fft_dim=256),
            train_loader_fft,
            val_loader_fft,
            f"{CFG.model_bin_path[:-4]}_fold{fold_id}.pth",
            f"FFT"
        )