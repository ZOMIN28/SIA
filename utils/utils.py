from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torchvision.transforms as T

def split_train_val(df, val_ratio=0.2):
    return train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["y"],
        random_state=42
    )


def sequential_kfold_split(df, n_splits=5):
    fold_size = len(df) // n_splits
    folds = []

    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else len(df)

        val_df = df.iloc[start:end]
        train_df = pd.concat([df.iloc[:start], df.iloc[end:]])

        folds.append((train_df, val_df))

    return folds


def make_submission(test_df, preds, path="submission.csv", if_save=True):

    df = pd.DataFrame({
        "ID": test_df["ID"],
        "TARGET": preds
    })

    if if_save:
        df.to_csv(path, index=False)
        print("Saved:", path)

    return df


def build_test_views(img):
    w, h = img.size

    crops = []

    # center
    crops.append(T.functional.center_crop(img, min(w,h) // 2))

    # 4 corners
    crops.append(img.crop((0, 0, w//2, h//2)))          # top-left
    crops.append(img.crop((w//2, 0, w, h//2)))         # top-right
    crops.append(img.crop((0, h//2, w//2, h)))         # bottom-left
    crops.append(img.crop((w//2, h//2, w, h)))         # bottom-right

    return crops


def extract_classes(train_df, val_df, classes=[6, 7]):
    train_df = train_df[
    train_df["y"].isin(classes)
    ].copy()

    val_df = val_df[
        val_df["y"].isin(classes)
    ].copy()

    label_map = {label: idx for idx, label in enumerate(classes)}

    train_df["y"] = train_df["y"].map(label_map)

    val_df["y"] = val_df["y"].map(label_map)

    print(train_df["y"].value_counts())
    print(val_df["y"].value_counts())

    return train_df, val_df
