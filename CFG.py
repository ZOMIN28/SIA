import torch

class CFG:
    train_csv = "dataset/Data/Data/training.csv"
    test_csv = "dataset/Data/Data/test.csv"
    batch_size = 16
    num_workers = 4

    epochs = 50
    lr = 3e-4
    num_classes = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    basepath = "checkpoints/"

    model_fft_path = basepath + "best_fft.pth"
    model_clip_path = basepath + "best_clip.pth"
    model_dino_path = basepath + "best_dino.pth"
    model_srm_path = basepath + "best_srm.pth"

    model_bin_path = basepath + "bin/best_bin67.pth"

    result_path = f"result/submission.csv"