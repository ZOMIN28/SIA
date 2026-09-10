import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch

from utils.utils import make_submission
from data.transforms3 import *
from model.fftmodel import NextModelFFT
from model.clipmodel import ModelCLIP
from model.dinomodel import ModelDINO
from model.srmmodel import SRMXception
from data.transforms_bin import *
import CFG

@torch.no_grad()
def ensemble_predict(test_df, fold_id, best_w=[0.2, 0.3, 0.4, 0.1], margin_th=0.2, bin=False):

    # load models
    m1 = NextModelFFT(fft_dim=256).to(CFG.device)
    m1.load_state_dict(torch.load(CFG.model_fft_path[:-4]+fold_id+".pth"))
    m1.eval()

    m2 = ModelCLIP().to(CFG.device)
    m2.load_state_dict(torch.load(CFG.model_clip_path[:-4]+fold_id+".pth"))
    m2.eval()

    m3 = ModelDINO().to(CFG.device)
    m3.load_state_dict(torch.load(CFG.model_dino_path[:-4]+fold_id+".pth"))
    m3.eval()

    m4 = SRMXception(10).to(CFG.device)
    m4.load_state_dict(torch.load(CFG.model_srm_path[:-4]+fold_id+".pth"))
    m4.eval()


    # =====================================================
    # Predict
    # =====================================================
    preds = []
    results = []
    count = 0


    for i in tqdm(range(len(test_df))):
        base_path = "dataset/Data/"
        img = Image.open(base_path+test_df.iloc[i]["path"]).convert("RGB")

        img1 = test_transform_fft(img).unsqueeze(0).to(CFG.device)   # FFT
        img2 = test_transform_clip(img).unsqueeze(0).to(CFG.device)      # CLIP
        img3 = test_transform_dino(img).unsqueeze(0).to(CFG.device)   # DINO
        img4 = test_transform_srm(img).unsqueeze(0).to(CFG.device)   # xception


        logits = (
            best_w[0] * m1(img1) +
            best_w[1] * m2(img2) +
            best_w[2] * m3(img3) +
            best_w[3] * m4(img4)
        ).squeeze(0)


        pred = logits.argmax().item()

        logits = logits.cpu().numpy()

        row = {
            "ID": test_df.iloc[i]["ID"]
        }

        for j in range(len(logits)):
            row[f"logit_{j}"] = logits[j]

        results.append(row)
            
        preds.append(pred)


    return preds, results


if __name__ == "__main__":
    with torch.no_grad():
        test_df = pd.read_csv(CFG.test_csv)
        
        fold_id_list = ["_fold0", "_fold1", "_fold2", "_fold3", "_fold4"]

        
        for fold_id in fold_id_list:

            preds, results = ensemble_predict(test_df=test_df, fold_id=fold_id, bin=False)

            df = make_submission(test_df, preds, CFG.result_path[:-4]+fold_id+".csv", if_save=True)

            df_logits = pd.DataFrame(results)
            save_path = CFG.result_path[:-4]+fold_id+".csv"
            df_logits.to_csv(f"logits_{save_path}", index=False)
            print("Saved logits csv.")
        