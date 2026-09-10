import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy.special import softmax

import torch
import torch.nn.functional as F
from data.transforms_bin import *
from model.fftmodel import NextModelFFT
from CFG import CFG

BIN = True

NUM_CLASSES = 10

result_path = CFG.result_path


logit_files = [
    "logits_result/submission_fold0.csv",
    "logits_result/submission_fold1.csv",
    "logits_result/submission_fold2.csv",
    "logits_result/submission_fold3.csv",
    "logits_result/submission_fold4.csv",
]

# 不确定阈值
margin_th_76 = 0.50
margin_th_67 = 0.50

margin_th_9 = 0.2                                                                                            


weights = np.array([
    1.0,
    0.5,
    0.5,
    0.5,
    1.0,
], dtype=np.float32)

weights = weights / weights.sum()

test_df = pd.read_csv(CFG.test_csv)

dfs = []

for file in logit_files:

    df = pd.read_csv(file)

    df = df.sort_values("ID").reset_index(drop=True)

    dfs.append(df)

# 检查ID一致
base_ids = dfs[0]["ID"]

for df in dfs:
    assert all(df["ID"] == base_ids)


if BIN:
    m_bin_67_list = []
    for fold_id in range(5):
        m_bin_67 = NextModelFFT(2).to(CFG.device)
        m_bin_67.load_state_dict(torch.load(f"{CFG.model_bin_path[:-4]}_fold{fold_id}.pth"))
        m_bin_67.eval()
        m_bin_67_list.append(m_bin_67)

    m_bin_49_list = []

results = []

trigger_count = 0

count = 0

for i in tqdm(range(len(test_df))):

    sample_id = test_df.iloc[i]["ID"]

    # =====================================================
    # 1. soft vote
    # =====================================================

    probs_all = []

    preds = []

    for df in dfs:

        logits = df[
            [f"logit_{j}" for j in range(NUM_CLASSES)]
        ].iloc[i].values

        prob = softmax(logits)

        pred = prob.argmax()

        preds.append(pred)

        probs_all.append(prob)
    
    probs_all = np.array(probs_all)

    # mean_prob = probs_all.mean(axis=0)
    mean_prob = np.average(
        probs_all,
        axis=0,
        weights=weights
    )

    mean_prob = torch.tensor(mean_prob.copy())
    pred = mean_prob.argmax().item()

    # =====================================================
    # 2. margin
    # =====================================================

    if BIN:
        probs = mean_prob

        top2_index = torch.topk(probs, 2).indices
        top1_cls = top2_index[0].item()
        top2_cls = top2_index[1].item()
        top2 = torch.topk(probs, 2)
        margin = (top2.values[0] - top2.values[1]).item()
        
        if (top2_cls == 9) and margin < margin_th_9:
            pred = 9

        elif (top1_cls == 7) and (top2_cls == 6) and margin < margin_th_76:
            base_path = "dataset/Data/"
            img = Image.open(base_path+test_df.iloc[i]["path"]).convert("RGB")

            img_bin = test_transform_fft_bin(img).unsqueeze(0).to(CFG.device)

            logit_bin = 0
            for m_bin in m_bin_67_list:
                logit_bin += m_bin(img_bin) * 0.2
            logit_bin = logit_bin.squeeze(0)
                
            probs_bin = F.softmax(logit_bin, dim=0)
            pred_bin = logit_bin.argmax().item()

            new_margin = np.abs((probs_bin[0]-probs_bin[1]).item())

            if new_margin > 0.5:
                if pred_bin == 0:
                    pred_new = 6
                else:
                    pred_new = 7
            
                if pred != pred_new:
                    count += 1
                    pred = pred_new
        
        elif (top1_cls == 6) and (top2_cls == 7) and margin < margin_th_67:
            base_path = "dataset/Data/"
            img = Image.open(base_path+test_df.iloc[i]["path"]).convert("RGB")

            img_bin = test_transform_fft_bin(img).unsqueeze(0).to(CFG.device)
            
            logit_bin = 0
            for m_bin in m_bin_67_list:
                logit_bin += m_bin(img_bin) * 0.2
            logit_bin = logit_bin.squeeze(0)

            probs_bin = F.softmax(logit_bin, dim=0)
            pred_bin = logit_bin.argmax().item()

            new_margin = np.abs((probs_bin[0]-probs_bin[1]).item())

            if new_margin > 0.5:
                if pred_bin == 0:
                    pred_new = 6
                else:
                    pred_new = 7
            
                if pred != pred_new:
                    count += 1
                    pred = pred_new
    


    results.append({
        "ID": sample_id,
        "TARGET": pred
    })


submission = pd.DataFrame(results)

submission.to_csv(
    result_path,
    index=False
)