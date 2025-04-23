import json
import os
import shutil
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Solving the problem of not being able to import your own packages under linux
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")

from src.clusters import paperClusterByDis
from src.dataset import GCN_ContrastiveDataSetSR
from src.utils import evaluate, parseJson

warnings.filterwarnings("ignore")


def save_embeddings(
    paper_ids: List[str],
    s_emb: torch.Tensor,
    r_emb: torch.Tensor,
    prediction: torch.Tensor,
    label: torch.Tensor,
    save_path: str,
    append: bool = False,
    is_complete: bool = True,
):
    """
    Save semantic, structural, fused embeddings and labels into a JSONL file.

    Args:
        paper_ids (List[str]): Original paper ID strings
        s_emb (Tensor): Semantic embeddings, shape [N, D]
        r_emb (Tensor): Structural embeddings, shape [N, D]
        prediction (Tensor): Fused embeddings, shape [N, D]
        label (Tensor): Ground truth labels, shape [N]
        save_path (str): Path to JSONL file
        append (bool): If True, append to file; otherwise, create new
        is_complete (bool): Whether the output is complete or not
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    s_emb_np = s_emb.detach().cpu().numpy()
    r_emb_np = r_emb.detach().cpu().numpy()
    pred_np = prediction.detach().cpu().numpy()
    label_np = label.detach().cpu().numpy()

    base_path, ext = os.path.splitext(save_path)
    incomplete_path = f"{base_path}_incomplete{ext}"

    mode = "a" if append else "w"
    with open(incomplete_path, mode, encoding="utf-8") as f:
        for i in range(len(label_np)):
            row = {
                "paper_id": paper_ids[i],
                "label": int(label_np[i]),
                "semantic_emb": s_emb_np[i].tolist(),
                "structural_emb": r_emb_np[i].tolist(),
                "fused_emb": pred_np[i].tolist(),
            }
            f.write(json.dumps(row) + "\n")

    if is_complete and os.path.exists(incomplete_path):
        shutil.move(incomplete_path, save_path)
        print(f"✅ Saved complete embeddings to {save_path}")
        return

    if not append:
        print(
            f"💾 {'Create' if not append else 'Update'} incomplete embeddings in {save_path}"
        )


best_modelPath = "arcc_model/modelrun_semTrue_relFalse_dp0.5_hid100_ep150_bs1_lr0.0001_tepc0.07_tepr0.07_tepf0.07_lfc1.0_lfs1.0_lff1.0_gcnL1_seed2021_metric-cosine_low0.45_high0.95.pkt"

model_name = best_modelPath.split("/")[-1].split(".pkt")[0]

model = torch.load(best_modelPath, weights_only=False).cuda()
model.eval()

all_pid_to_idx = parseJson("ARCC-AND/data/Aminer-18/processed/all_pid_to_idx.json")

valid_adj_matrix = parseJson("ARCC-AND/data/Aminer-18/processed/test_adj_rule.json")
valid_data_df = pd.read_csv("ARCC-AND/data/Aminer-18/processed/test_paper_label.csv")
valid_name_list = valid_data_df["name"].unique().tolist()
valid_data = GCN_ContrastiveDataSetSR(valid_name_list, valid_data_df)
valid_params = {"batch_size": 1, "shuffle": False, "drop_last": False}
valid_generator = DataLoader(valid_data, **valid_params)

accs = []
losses = []
counter = 0
with torch.no_grad():
    test_pres_perepoch = []
    test_recs_perepoch = []
    test_f1s_perepoch = []
    for iter, sample_list in enumerate(valid_generator):
        name = str(sample_list[2][0])

        label_list = []
        pid_index_list = []
        for paper_id, la in zip(sample_list[0], sample_list[1]):
            pid_index_list.append(all_pid_to_idx[paper_id[0]])
            label_list.append(la.item())
        pid_index_tensor = torch.tensor(pid_index_list).cuda()
        label = torch.tensor(label_list).cuda()
        adj_matrix_tensor = torch.tensor(valid_adj_matrix[name]).cuda()

        s_emb, r_emb, prediction, _ = model(
            pid_index_tensor=pid_index_tensor,
            adj_matrix_tensor=adj_matrix_tensor,
        )

        paper_ids = [pid[0] for pid in sample_list[0]]
        save_embeddings(
            paper_ids=paper_ids,
            s_emb=s_emb,
            r_emb=r_emb,
            prediction=prediction,
            label=label,
            save_path="./embeddings/Aminer-18/aminer18_embeddings_test.jsonl",
            append=iter != 0,
            is_complete=iter == len(valid_generator) - 1,
        )

        # 得到accuracy：预测正确的样本个数与总样本数的比值，需要对比预测的类别标签，与真实值得类别标签
        ########## computer_ACC####################
        sim_matrix = F.cosine_similarity(
            prediction.unsqueeze(1), prediction.unsqueeze(0), dim=2
        ).detach()
        pred_matrix = torch.where(sim_matrix > 0.5, 1, 0).detach()
        label_matrix = torch.where(
            label.unsqueeze(1) == label.unsqueeze(1).T, 1, 0
        ).detach()
        acc_t = torch.sum(torch.where(label_matrix == pred_matrix, 1, 0)).item() / (
            label_matrix.shape[0] * label_matrix.shape[1]
        )
        accs.append(acc_t)
        counter += 1
        ###### computer F1 #######
        dis = 1 - sim_matrix
        cluster_num = len(set(label_list))
        papers = list(range(len(label_list)))
        paper_name_dict = {}
        for paperid in papers:
            label_value = int(label[paperid])
            if label_value not in paper_name_dict.keys():
                paper_name_dict[label_value] = []
                paper_name_dict[label_value].append(paperid)
            else:
                paper_name_dict[label_value].append(paperid)
        result = paperClusterByDis(dis.cpu(), papers, cluster_num, method="AG")
        # 评估指标
        precision, recall, f1 = evaluate(result, paper_name_dict)
        test_pres_perepoch.append(precision)
        test_recs_perepoch.append(recall)
        test_f1s_perepoch.append(f1)
    precision = np.mean(test_pres_perepoch)
    recall = np.mean(test_recs_perepoch)
    f1 = np.mean(test_f1s_perepoch)
    print({"precision": precision, "recall": recall, "f1": f1})
