import os
import sys
import json
import warnings
import shutil
from typing import List

import torch
import torch.nn.functional as F
import datetime
import pandas as pd
import numpy as np
import codecs

from torch.utils.data import DataLoader

# Solving the problem of not being able to import your own packages under linux
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")

from src.new_models import NB_AREByS_N2VEmbGCN_SCL
from src.dataset import GCN_ContrastiveDataSetSR
from src.contrastive_loss import SupConLoss
from src.utils import parseJson, saveJson, evaluate, parse_configion
from src.util_training import setup_seed, mkdir, draw_acc_loss_curve
from src.clusters import paperClusterByDis

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


def mark_file(file_path: str, mark="best"):
    """
    Create a copy of the given file with '_best' added before the extension.
    If such a file already exists, it will be replaced.

    Args:
        file_path (str): Path to the original file
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    base, ext = os.path.splitext(file_path)
    mark_path = f"{base}_{mark}{ext}"

    try:
        shutil.move(file_path, mark_path)
        print(f"🏆 Saved {mark} version as: {mark_path}")
    except Exception as e:
        print(f"⚠️ Failed to save {mark} version: {e}")


def load_embeddings(save_path: str, max_lines: int = None):
    """
    Load embeddings from a JSONL file.

    Args:
        save_path (str): Path to JSONL file
        max_lines (int, optional): If specified, only load this many lines

    Returns:
        List[dict]: Loaded embedding entries
    """
    data = []
    with open(save_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            data.append(json.loads(line.strip()))
    return data


# step1: parse config and add parse
config = parse_configion()
setup_seed(config["seed"])
# Solve the problem of absolute and relative paths for different clients
BasePath = os.path.abspath(os.path.dirname(__file__))
# Current file name
curFileName = os.path.basename(__file__).split(".")[0]
###########################################################################


# step2: Construct the output file, defining the directory where the output to be generated by the model will be located.
save_base_folder = "{}/{}/{}".format(BasePath, config["save_path"], curFileName)
save_bestmodels = "{}/train_bestmodels".format(save_base_folder)
save_train_result = "{}/train_result".format(save_base_folder)
save_train_logs = "{}/train_logs".format(save_base_folder)
save_test_output = "{}/test_output".format(save_base_folder)
save_test_result = "{}/test_result".format(save_base_folder)

# step3: get file path

# test raw data
test_raw_data_path = "{}/{}/{}".format(
    BasePath, config["raw_path"], config["test_raw_data"]
)

# df
# all_pid2name_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['all_pid2name'])
train_df_path = "{}/{}/{}".format(
    BasePath, config["processed_path"], config["train_df"]
)
valid_df_path = "{}/{}/{}".format(
    BasePath, config["processed_path"], config["valid_df"]
)
test_df_path = "{}/{}/{}".format(BasePath, config["processed_path"], config["test_df"])

# adj
train_adj_rule_path = "{}/{}/{}".format(
    BasePath, config["processed_path"], config["train_adj_rule"]
)
valid_adj_rule_path = "{}/{}/{}".format(
    BasePath, config["processed_path"], config["valid_adj_rule"]
)
test_adj_rule_path = "{}/{}/{}".format(
    BasePath, config["processed_path"], config["test_adj_rule"]
)


# embedding layer
all_pid_list_path = "{}/{}/{}".format(
    BasePath, config["processed_path"], config["all_pid"]
)
all_pid_to_idx_path = "{}/{}/{}".format(
    BasePath, config["processed_path"], config["all_pid_to_idx"]
)
sem_emb_vector_bert_path = "{}/{}/{}".format(
    BasePath, config["processed_path"], config["sem_emb_vector_bert"]
)
# sem_emb_vector_w2v_path =  "{}/{}/{}".format(BasePath,config['processed_path'],config['sem_emb_vector_w2v'])
rel_emb_vector_rule_path = "{}/{}/{}".format(
    BasePath, config["processed_path"], config["rel_emb_vector_rule"]
)


# ACC_SIM
ACC_SIM = config["acc_sim"]


def train():
    # get parameter
    run_model = config["run_model"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    hidden = config["hidden"]
    dropout = config["dropout"]
    lr = config["learning_rate"]
    temperature_content = config["temperature_content"]
    temperature_structure = config["temperature_structure"]
    temperature_fusion = config["temperature_fusion"]
    lossfac_content = config["lossfac_content"]
    lossfac_structure = 1.0
    lossfac_fusion = config["lossfac_fusion"]
    gcnLayer = config["gcnLayer"]
    seed = config["seed"]
    low_sim_threshold = config["low_sim_threshold"]
    high_sim_threshold = config["high_sim_threshold"]
    metric_type = config["metric_type"]
    rel_freeze = bool(config["rel_freeze"])
    sem_freeze = bool(config["sem_freeze"])

    save_file_name = "model{}_sem{}_rel{}_dp{}_hid{}_ep{}_bs{}_lr{}_tepc{}_tepr{}_tepf{}_lfc{}_lfs{}_lff{}_gcnL{}_seed{}_metric-{}_low{}_high{}".format(
        run_model,
        sem_freeze,
        rel_freeze,
        dropout,
        hidden,
        epochs,
        batch_size,
        lr,
        temperature_content,
        temperature_structure,
        temperature_fusion,
        lossfac_content,
        lossfac_structure,
        lossfac_fusion,
        gcnLayer,
        seed,
        metric_type,
        low_sim_threshold,
        high_sim_threshold,
    )
    logpath = "{}/{}.txt".format(save_train_logs, save_file_name)

    global log
    if run_model == "debug":
        log = sys.stdout
    else:
        log = open(logpath, "w", encoding="utf-8")

    # df
    # all_pid2name = parseJson(all_pid2name_path)
    train_data_df = pd.read_csv(train_df_path)
    valid_data_df = pd.read_csv(valid_df_path)
    # name list
    train_name_list = train_data_df["name"].unique().tolist()
    valid_name_list = valid_data_df["name"].unique().tolist()

    # adj
    train_adj_matrix = parseJson(train_adj_rule_path)
    valid_adj_matrix = parseJson(valid_adj_rule_path)

    # embedding layer data
    all_pid_to_idx = parseJson(all_pid_to_idx_path)
    all_sem_emb_vector_list = parseJson(sem_emb_vector_bert_path)
    all_rel_emb_vector_list = parseJson(rel_emb_vector_rule_path)

    all_sem_emb_vector = torch.tensor(all_sem_emb_vector_list).cuda()
    all_rel_emb_vector = torch.tensor(all_rel_emb_vector_list).cuda()

    # dataset dataloader
    train_data = GCN_ContrastiveDataSetSR(train_name_list, train_data_df)
    training_params = {"batch_size": batch_size, "shuffle": True, "drop_last": False}
    training_generator = DataLoader(train_data, **training_params)

    valid_data = GCN_ContrastiveDataSetSR(valid_name_list, valid_data_df)
    valid_params = {"batch_size": batch_size, "shuffle": False, "drop_last": False}
    valid_generator = DataLoader(valid_data, **valid_params)

    #
    num_iter_per_epoch = len(training_generator)

    # model structure
    model = NB_AREByS_N2VEmbGCN_SCL(
        sem_freeze=sem_freeze,
        rel_freeze=rel_freeze,
        sem_emb_vector=all_sem_emb_vector,
        rel_emb_vector=all_rel_emb_vector,
        hidden=hidden,
        dropout=dropout,
        gcn_layer=gcnLayer,
        low_sim_threshold=low_sim_threshold,
        high_sim_threshold=high_sim_threshold,
        metric_type=metric_type,
    ).cuda()
    # loss function
    criterion_content = SupConLoss(temperature=temperature_content)
    criterion_structure = SupConLoss(temperature=temperature_structure)
    criterion_fusion = SupConLoss(temperature=temperature_fusion)
    # optimizer selector
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # var data
    train_step = 0
    valid_step = 0
    best_acc = 0
    best_loss = 100
    best_f1 = 0
    train_loss = []
    train_accs = []
    test_loss = []
    test_accs = []
    test_pres = []
    test_recs = []
    test_f1s = []
    best_modelPath = ""
    res_file_path = ""
    train_pres_perepoch = []
    train_recs_perepoch = []
    train_f1s_perepoch = []

    for ep in range(epochs):
        if run_model == "debug":
            if ep > 2:
                break
        begin1 = datetime.datetime.now()

        print("##########{}/{}:Train Model#############".format(ep, epochs), file=log)
        print("##########{}/{}:Train Model#############".format(ep, epochs))
        train_pres_perbs = []
        train_recs_perbs = []
        train_f1s_perbs = []
        train_loss_tmp = []
        train_acc_tmp = []
        model.train()
        begin2 = datetime.datetime.now()
        for iter, sample_list in enumerate(training_generator):
            train_step += 1
            name = str(sample_list[2][0])

            # name = all_pid2name[sample_list[0][0][0]]
            if run_model == "debug":
                if iter > 2:
                    break
            label_list = []
            pid_index_list = []
            for paper_id, la in zip(sample_list[0], sample_list[1]):
                pid_index_list.append(all_pid_to_idx[paper_id[0]])
                label_list.append(la.item())
            pid_index_tensor = torch.tensor(pid_index_list).cuda()
            label = torch.tensor(label_list).cuda()
            adj_matrix_tensor = torch.tensor(
                train_adj_matrix[name], requires_grad=True
            ).cuda()

            # 梯度清零
            optimizer.zero_grad()
            # 执行模型，得到批预测结果
            s_emb, r_emb, prediction, _ = model(
                pid_index_tensor=pid_index_tensor, adj_matrix_tensor=adj_matrix_tensor
            )

            # Save embeddings
            if ep == epochs - 1 or ep % 10 == 0:
                paper_ids = [pid[0] for pid in sample_list[0]]
                save_embeddings(
                    paper_ids=paper_ids,
                    s_emb=s_emb,
                    r_emb=r_emb,
                    prediction=prediction,
                    label=label,
                    save_path="./embeddings/Aminer-18/aminer18_embeddings_train.jsonl",
                    append=iter != 0,
                    is_complete=iter == len(training_generator) - 1,
                )

            # 根据pre与label值的距离，计算loss，loss是标量
            # loss 是数值，是通过loss function计算得到的数值，含义是预测数值与真实数值的差距；
            loss1 = criterion_content(s_emb, label)
            loss2 = criterion_structure(r_emb, label)
            loss3 = criterion_fusion(prediction, label)
            loss = (
                lossfac_content * loss1
                + lossfac_structure * loss2
                + lossfac_fusion * loss3
            )
            # 误差反向传播，计算梯度
            loss.backward()
            # 根据梯度，更新参数
            optimizer.step()

            # 得到accuracy：预测正确的样本个数与总样本数的比值，需要对比预测的类别标签，与真实值得类别标签
            ########## computer_ACC####################
            sim_matrix = F.cosine_similarity(
                prediction.unsqueeze(1), prediction.unsqueeze(0), dim=2
            ).detach()
            pred_matrix = torch.where(sim_matrix > ACC_SIM, 1, 0).detach()
            label_matrix = torch.where(
                label.unsqueeze(1) == label.unsqueeze(1).T, 1, 0
            ).detach()
            acc_t = torch.sum(torch.where(label_matrix == pred_matrix, 1, 0)).item() / (
                label_matrix.shape[0] * label_matrix.shape[1]
            )
            # train_acc_iter = np.mean(acc_t)
            train_acc_tmp.append(acc_t)
            train_loss_tmp.append(loss.item())
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
            # tp, fp, fn, tn = evaluate_fourMetrics(result, name_papers[name])
            train_pres_perbs.append(precision)
            train_recs_perbs.append(recall)
            train_f1s_perbs.append(f1)

            # 打印指标日志
            end2 = datetime.datetime.now()
            if iter == len(training_generator) - 1:
                print(
                    "Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {:.4f}, Accuracy: {},Usingtime:{}".format(
                        ep + 1,
                        epochs,
                        iter + 1,
                        num_iter_per_epoch,
                        optimizer.param_groups[0]["lr"],
                        loss,
                        acc_t,
                        end2 - begin2,
                    ),
                    file=log,
                )
                print(
                    "Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {:.4f}, Accuracy: {},Usingtime:{}".format(
                        ep + 1,
                        epochs,
                        iter + 1,
                        num_iter_per_epoch,
                        optimizer.param_groups[0]["lr"],
                        loss,
                        acc_t,
                        end2 - begin2,
                    )
                )

            begin2 = datetime.datetime.now()

        train_loss.append(np.mean(train_loss_tmp))
        train_accs.append(np.mean(train_acc_tmp))
        train_pres_perepoch.append(np.mean(train_pres_perbs))
        train_recs_perepoch.append(np.mean(train_recs_perbs))
        train_f1s_perepoch.append(np.mean(train_f1s_perbs))

        print("##########{}/{}: Eval Model#############".format(ep, epochs))
        print("##########{}/{}: Eval Model#############".format(ep, epochs), file=log)
        # Verify the performance of the epoch
        model.eval()
        accs = []
        losses = []
        counter = 0
        with torch.no_grad():
            test_pres_perepoch = []
            test_recs_perepoch = []
            test_f1s_perepoch = []
            for iter, sample_list in enumerate(valid_generator):
                valid_step += 1
                name = str(sample_list[2][0])

                # name = all_pid2name[sample_list[0][0][0]]

                if run_model == "debug":
                    if iter > 2:
                        break

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

                # Save embeddings
                if ep == epochs - 1 or ep % 10 == 0:
                    paper_ids = [pid[0] for pid in sample_list[0]]
                    save_embeddings(
                        paper_ids=paper_ids,
                        s_emb=s_emb,
                        r_emb=r_emb,
                        prediction=prediction,
                        label=label,
                        save_path="./embeddings/Aminer-18/aminer18_embeddings_validation.jsonl",
                        append=iter != 0,
                        is_complete=iter == len(valid_generator) - 1,
                    )

                loss1 = criterion_content(s_emb, label)
                loss2 = criterion_structure(r_emb, label)
                loss3 = criterion_fusion(prediction, label)
                loss = (
                    lossfac_content * loss1
                    + lossfac_structure * loss2
                    + lossfac_fusion * loss3
                )

                # 得到accuracy：预测正确的样本个数与总样本数的比值，需要对比预测的类别标签，与真实值得类别标签
                ########## computer_ACC####################
                sim_matrix = F.cosine_similarity(
                    prediction.unsqueeze(1), prediction.unsqueeze(0), dim=2
                ).detach()
                pred_matrix = torch.where(sim_matrix > ACC_SIM, 1, 0).detach()
                label_matrix = torch.where(
                    label.unsqueeze(1) == label.unsqueeze(1).T, 1, 0
                ).detach()
                acc_t = torch.sum(
                    torch.where(label_matrix == pred_matrix, 1, 0)
                ).item() / (label_matrix.shape[0] * label_matrix.shape[1])
                accs.append(acc_t)
                losses.append(loss.item())
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

            # 计算平均loss, 总loss / 总数据个数
            te_loss = sum(losses) / counter
            # 记录该批次的evel数据集的loss
            test_loss.append(te_loss)
            te_acc = sum(accs) / counter
            test_accs.append(te_acc)
            test_pres.append(np.mean(test_pres_perepoch))
            test_recs.append(np.mean(test_recs_perepoch))
            test_f1s.append(np.mean(test_f1s_perepoch))
            te_f1 = np.mean(test_f1s_perepoch)
            if te_f1 > best_f1:
                best_f1 = te_f1
                # best_loss = te_loss
                best_model_epoch = ep
                # todo 添加参数
                best_modelPath = "{}/{}.pkt".format(save_bestmodels, save_file_name)
                torch.save(model, best_modelPath)

                if ep == epochs - 1 or ep % 10 == 0:
                    mark_file(
                        file_path="./embeddings/Aminer-18/aminer18_embeddings_train.jsonl"
                    )
                    mark_file(
                        file_path="./embeddings/Aminer-18/aminer18_embeddings_validation.jsonl"
                    )

            # 打印eval的指标日志
            print(
                "Eval ==> Epoch: {}/{}, EvalLoss: {:.4f}, EvalAccuracy: {}".format(
                    ep, epochs, te_loss, te_acc
                ),
                file=log,
            )
            print(
                "Eval ==> Epoch: {}/{}, EvalLoss: {:.4f}, EvalAccuracy: {}".format(
                    ep, epochs, te_loss, te_acc
                )
            )
        # 最好的epoch数据记录下来
        result = {
            "model_name": curFileName,
            "run_model": run_model,
            "best_model_path": best_modelPath,
            "best_model_epoch": best_model_epoch,
            "log_path": logpath,
            "parameters": {
                "hidden": hidden,
                "epoch": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "dropout": dropout,
                "temperature_content": temperature_content,
                "temperature_structure": temperature_structure,
                "temperature_fusion": temperature_fusion,
                "lossfac_content": lossfac_content,
                "lossfac_structure": lossfac_structure,
                "lossfac_fusion": lossfac_fusion,
                "gcnLayer": gcnLayer,
                "seed": seed,
                "metric_type": metric_type,
                "low_sim_threshold": low_sim_threshold,
                "high_sim_threshold": high_sim_threshold,
                "rel_freeze": rel_freeze,
                "sem_freeze": sem_freeze,
            },
            "best_eval_acc": best_acc,
            "best_eval_loss": best_loss,
            "best_eval_f1": best_f1,
            "train_accs": train_accs,
            "train_loss": train_loss,
            "train_pres": train_pres_perepoch,
            "train_recs": train_recs_perepoch,
            "train_f1s": train_f1s_perepoch,
            "eval_accs": test_accs,
            "eval_loss": test_loss,
            "eval_pres": test_pres,
            "eval_recs": test_recs,
            "eval_f1s": test_f1s,
        }
        res_file_path = "{}/{}.json".format(save_train_result, save_file_name)
        saveJson(res_file_path, result)
        end1 = datetime.datetime.now()
        print("one epoch using time:", end1 - begin1, file=log)
        print("one epoch using time:", end1 - begin1)
    print("################ TrainFunction Finish! ##########################", file=log)
    print("################ TrainFunction Finish! ##########################")

    return res_file_path


def model_test(res_file_path):
    train_res_file = parseJson(res_file_path)
    best_modelPath = train_res_file["best_model_path"]
    global log
    model_name = best_modelPath.split("/")[-1].split(".pkt")[0]
    print("################# Test Start ###################", file=log)
    print("################# Test Start ###################")
    # test数据集
    name_papers = parseJson(test_raw_data_path)
    test_adj_matrix = parseJson(test_adj_rule_path)
    all_pid_to_idx = parseJson(all_pid_to_idx_path)

    model = torch.load(best_modelPath).cuda()
    model.eval()

    result_file_word2vec_local1 = "{}/{}.xls".format(save_test_output, model_name)
    file_word2vec_local1 = codecs.open(result_file_word2vec_local1, "w")
    file_word2vec_local1.write("index\tname\tprecision\trecall\tf1\n")

    sigmoid_score1 = [0, 0, 0]
    cnt = 0
    train_acc_tmp = []
    # 按name迭代预测
    for index, name in enumerate(name_papers.keys()):
        # try:
        cnt += 1
        papers = []
        # 获得该name的所有论文
        label_list = []
        label_counter = 1

        for talentid in name_papers[name]:
            papers.extend(name_papers[name][talentid])
            label_list.extend([label_counter] * len(name_papers[name][talentid]))
            label_counter += 1

        pid_index_list = []
        for paper_id in papers:
            pid_index_list.append(all_pid_to_idx[paper_id])

        pid_index_tensor = torch.tensor(pid_index_list).cuda()
        adj_matrix_tensor = torch.tensor(
            test_adj_matrix[name], requires_grad=True
        ).cuda()
        label = torch.tensor(label_list).cuda()

        paper_num = len(papers)
        print(index, name, paper_num, file=log)
        print(index, name, paper_num)

        s_emb, r_emb, prediction, refine_adj_matrix_tensor = model(
            pid_index_tensor=pid_index_tensor, adj_matrix_tensor=adj_matrix_tensor
        )

        # Save embeddings
        save_embeddings(
            paper_ids=papers,
            s_emb=s_emb,
            r_emb=r_emb,
            prediction=prediction,
            label=label,
            save_path="./embeddings/Aminer-18//aminer18_embeddings_test.jsonl",
            append=index != 0,
        )

        pred = prediction.cpu().detach()

        #####################################
        sim_matrix = F.cosine_similarity(pred.unsqueeze(1), pred.unsqueeze(0), dim=2)
        pred_matrix = torch.where(sim_matrix > ACC_SIM, 1, 0)
        label_matrix = torch.where(
            label.unsqueeze(1) == label.unsqueeze(1).T, 1, 0
        ).detach()
        label_matrix = label_matrix.cpu().detach()
        pred_matrix = pred_matrix.cpu().detach()
        # train_acc_iter = np.mean(acc_t)
        train_metrics_t = torch.sum(
            torch.where(label_matrix == pred_matrix, 1, 0)
        ).item() / (label_matrix.shape[0] * label_matrix.shape[1])
        train_acc_tmp.append(train_metrics_t)
        pred_sim = sim_matrix
        dis = 1 - pred_sim
        # papers 所有的论文，name_papers[name] 所有的talentid数量，即为聚类数量
        result = paperClusterByDis(dis, papers, len(name_papers[name]), method="AG")
        # 评估指标
        precision, recall, f1 = evaluate(result, name_papers[name])
        sigmoid_score1[0] += precision
        sigmoid_score1[1] += recall
        sigmoid_score1[2] += f1

        file_word2vec_local1.write(
            "%s\t %s\t %4.2f%%\t %4.2f%%\t %4.2f%%\n"
            % (index, name, precision * 100, recall * 100, f1 * 100)
        )
        # for()
        print("pred距离聚类结果：", precision, recall, f1, file=log)
        print("pred距离聚类结果：", precision, recall, f1)

    sigmoid_score1[0] /= cnt
    sigmoid_score1[1] /= cnt
    sigmoid_score1[2] /= cnt

    file_word2vec_local1.write(
        "0\t average\t %4.2f%%\t %4.2f%%\t %4.2f%%\n"
        % (sigmoid_score1[0] * 100, sigmoid_score1[1] * 100, sigmoid_score1[2] * 100)
    )
    train_res_file["test_metrics"] = "0\t average\t %4.2f%%\t %4.2f%%\t %4.2f%%\n" % (
        sigmoid_score1[0] * 100,
        sigmoid_score1[1] * 100,
        sigmoid_score1[2] * 100,
    )
    train_res_file["test_acc"] = np.mean(train_acc_tmp)
    test_res_file_path = "{}/{}.json".format(save_test_result, model_name)
    saveJson(test_res_file_path, train_res_file)
    print("################ TestFunction Finish! ##########################", file=log)
    print("################ TestFunction Finish! ##########################")
    return test_res_file_path


if __name__ == "__main__":
    # mkdir output file
    mkdir(save_base_folder)
    mkdir(save_bestmodels)
    mkdir(save_train_result)
    mkdir(save_train_logs)
    mkdir(save_test_output)
    mkdir(save_test_result)

    # train and test and draw_pic
    res_file_path = train()
    test_res_file_path = model_test(res_file_path)
    draw_acc_loss_curve(test_res_file_path, save_test_result)
