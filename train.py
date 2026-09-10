import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import split_train_val, sequential_kfold_split, extract_classes
from data.dataloader import build_loader
from data.transforms3 import *
from model.fftmodel import  NextModelFFT
from model.clipmodel import ModelCLIP
from model.dinomodel import ModelDINO
from model.srmmodel import SRMXception
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

    # CLIP
    if model_type in ["CLIP","clip"]:
        optimizer = torch.optim.AdamW([
            {"params": model.fc.parameters(), "lr": 1e-3},
            {"params": model.vision_model.encoder.layers[-1].parameters(), "lr": 1e-5},
            {"params": model.vision_model.encoder.layers[-2].parameters(), "lr": 1e-5},
            {"params": model.vision_model.encoder.layers[-3].parameters(), "lr": 1e-5}
        ], weight_decay=1e-4)

    # Dino
    if model_type in ["DINO","dino"]:
        optimizer = torch.optim.AdamW([
            {"params": model.fc.parameters(), "lr": 1e-3},
            {"params": model.backbone.blocks[-1].parameters(), "lr": 1e-5},
            {"params": model.backbone.blocks[-2].parameters(), "lr": 1e-5},
            {"params": model.backbone.blocks[-3].parameters(), "lr": 1e-5},
        ], weight_decay=1e-4)

    # SRM
    if model_type in ["SRM", "srm"]:
        optimizer = torch.optim.AdamW([
            {"params": model.fc.parameters(), "lr": 1e-4},
            # {"params": model.srm.parameters(), "lr": 1e-4},
            {"params": model.backbone.parameters(), "lr": 1e-4},
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
    train_df, val_df = split_train_val(df)

    df = pd.read_csv(CFG.train_csv)
    folds = sequential_kfold_split(df, n_splits=5)

    for fold_id, (train_df, val_df) in enumerate(folds):
        print(f"\n===== Fold {fold_id} =====")

        # FFT
        train_loader_fft = build_loader(train_df, train_transform_fft, True)
        val_loader_fft = build_loader(val_df, test_transform_fft, False)
        train_model(
            NextModelFFT(fft_dim=256),
            train_loader_fft,
            val_loader_fft,
            f"{CFG.model_fft_path[:-4]}_fold{fold_id}.pth",
            f"FFT"
        )


        # CLIP
        train_loader_clip = build_loader(train_df, train_transform_clip, True)
        val_loader_clip = build_loader(val_df, test_transform_clip, False)
        train_model(
            ModelCLIP(),
            train_loader_clip,
            val_loader_clip,
            f"{CFG.model_clip_path[:-4]}_fold{fold_id}.pth",
            f"CLIP"
        )

        # DINO
        train_loader_dino = build_loader(train_df, train_transform_dino, True)
        val_loader_dino = build_loader(val_df, test_transform_dino, False)
        train_model(
            ModelDINO(),
            train_loader_dino,
            val_loader_dino,
            f"{CFG.model_dino_path[:-4]}_fold{fold_id}.pth",
            f"DINO"
        )

        # xception
        train_loader_srm = build_loader(train_df, train_transform_srm, True)
        val_loader_srm = build_loader(val_df, test_transform_srm, False)
        train_model(
            SRMXception(),
            train_loader_srm,
            val_loader_srm,
            f"{CFG.model_srm_path[:-4]}_fold{fold_id}.pth",
            f"srm"
        )