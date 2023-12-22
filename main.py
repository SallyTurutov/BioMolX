import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import random

from sklearn.metrics import roc_auc_score

import pandas as pd
from meta_model import Meta_model

import os
import shutil


def main(dataset, input_model_file, gnn_type, add_similarity, add_selfsupervise, add_masking, add_weight, m_support):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="graphsage")
    parser.add_argument('--dataset', type=str, help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=42, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')

    parser.add_argument('--k_query', type=int, default=32, help="size of query datasets")
    parser.add_argument('--meta_lr', type=float, default=0.001)
    parser.add_argument('--update_lr', type=float, default=0.4)
    parser.add_argument('--add_similarity', type=bool, default=False)
    parser.add_argument('--add_selfsupervise', type=bool, default=False)
    parser.add_argument('--add_weight', type=float, default=0.1)

    args = parser.parse_args()

    args.dataset = dataset
    args.input_model_file = input_model_file
    args.gnn_type = gnn_type
    args.add_similarity = add_similarity
    args.add_selfsupervise = add_selfsupervise
    args.add_masking = add_masking
    args.add_weight = add_weight
    args.m_support = m_support

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    epochs = 1000
    organism_labels_dict = {
        "human": 0,
        "mouse": 1,
        "rat": 2,
        "chinese hamster": 3,
        "guinea pig": 4,
        "rabbit": 5,
        "pig": 6,
        "bovine": 7

    }

    if args.dataset == "brain":
        #  0 - human, 1 - mouse, 2 - rat, 3 - pig, 4 - guinea pig, 5 - bovine, 6 - rabbit
        all_tasks = [2, 1, 4, 6, 3, 5, 0]  # Target: Human
        organisms = ["rat", "mouse", "guinea pig", "rabbit", "pig", "bovine", "human"]

        # all_tasks = [5, 3, 0, 6, 4, 1, 2]  # Target: Rat
        # organisms = ["bovine", "pig", "human", "rabbit", "guinea pig", "mouse", "rat"]

        # all_tasks = [5, 3, 4, 6, 2, 0, 1]  # Target: Mouse
        # organisms = ["bovine", "pig", "guinea pig", "rabbit", "rat", "human", "mouse"]

        # all_tasks = [5, 3, 6, 1, 2, 0, 4]  # Target: Guinea Pig
        # organisms = ["bovine", "pig", "rabbit", "mouse", "rat", "human", "guinea pig"]

        # all_tasks = [5, 3, 4, 0, 2, 1, 6]  # Target: Rabbit
        # organisms = ["bovine", "pig", "guinea pig", "human", "rat", "human", "rabbit"]

        # all_tasks = [0, 4, 6, 5, 2, 1, 3]  # Target: Pig
        # organisms = ["human", "guinea pig", "rabbit", "bovine", "rat", "mouse", "pig"]

        # all_tasks = [4, 6, 3, 2, 1, 0, 5]  # Target: Bovine
        # organisms = ["guinea pig", "rabbit", "pig", "rat", "mouse", "human", "bovine"]

    elif args.dataset == "breast":
        #  0 - human
        all_tasks = [0]  # Target: Human
        organisms = ["human"]

    elif args.dataset == "cervix":
        #  0 - human
        all_tasks = [0]  # Target: Human
        organisms = ["human"]

    elif args.dataset == "intestinal":
        #  0 - human, 1 - guinea pig
        all_tasks = [1, 0]  # Target: Human
        organisms = ["guinea pig", "human"]

        # all_tasks = [0, 1]  # Target: Guinea Pig
        # organisms = ["human", "guinea pig"]

    elif args.dataset == "kidney":
        #  0 - human, 1 - rat, 2 - pig
        all_tasks = [1, 0]  # Target: Human
        organisms = ["rat", "human"]

        # all_tasks = [0, 1]  # Target: Rat
        # organisms = ["human", "rat"]

    elif args.dataset == "liver":
        #  0 - human, 1 - rat
        all_tasks = [1, 0]  # Target: Human
        organisms = ["rat", "human"]

        # all_tasks = [0, 1]  # Target: Rat
        # organisms = ["human", "rat"]

    elif args.dataset == "lung":
        #  0 - human, 1 - guinea pig
        all_tasks = [1, 0]  # Target: Human
        organisms = ["guinea pig", "human"]

        # all_tasks = [0, 1]  # Target: Guinea Pig
        # organisms = ["human", "guinea pig"]

    elif args.dataset == "ovary":
        #  0 - human, 1 - chinese hamster
        all_tasks = [1, 0]  # Target: Human
        organisms = ["chinese hamster", "human"]

        # all_tasks = [0, 1]  # Target: Chinese Hamster
        # organisms = ["human", "chinese hamster"]

    elif args.dataset == "prostate":
        #  0 - human, 1 - rat
        all_tasks = [1, 0]  # Target: Human
        organisms = ["rat", "human"]

        # all_tasks = [0, 1]  # Target: Rat
        # organisms = ["human", "rat"]

    elif args.dataset == "skin":
        #  0 - human
        all_tasks = [0]  # Target: Human
        organisms = ["human"]

    else:
        raise ValueError("Invalid dataset name: " + str(args.dataset))

    args.num_tasks = len(all_tasks)
    model = Meta_model(args).to(device)

    best_accs = []
    best_acc_target_task = 0
    best_epoch_target_task = 0

    target_task = 0  # Human
    target_idx = all_tasks.index(target_task)
    print(f"all_tasks: {all_tasks}, target_task: {target_task}, target_idx: {target_idx}.")
    print(f"Training for task: {args.dataset}")
    label_by_organism_list = [organism_labels_dict[organism] for organism in organisms]

    model_path = "result/model" + args.dataset + ".pth"
    out_path = "result/" + args.dataset + ".txt"

    for epoch in tqdm(range(1, epochs + 1)):

        accs = model.train_and_eval(all_tasks, label_by_organism_list, epoch)

        # Check if the ACC for the last task in all_tasks has improved
        if accs[target_idx] > best_acc_target_task:
            best_acc_target_task = accs[target_idx]
            best_epoch_target_task = epoch
            model.save_checkpoint(model_path)

        if best_accs != []:
            for acc_num in range(len(best_accs)):
                if best_accs[acc_num] < accs[acc_num]:
                    best_accs[acc_num] = accs[acc_num]
        else:
            best_accs = accs

        fw = open(out_path, "a")
        fw.write("epoch: " + str(epoch) + "\t")
        fw.write("test: " + "\t")
        for i in accs:
            fw.write(str(i) + "\t")

        fw.write("best: " + "\t")
        for i in best_accs:
            fw.write(str(i) + "\t")
        fw.write("\n")
        fw.close()

    # Update the model's parameters to the parameters at the best epoch for the last task
    fw = open(out_path, "a")
    fw.write('Updating model parameters from epoch: ' + str(best_epoch_target_task) + ' with AUC: ' + str(
        best_acc_target_task) + "\n" + "\n")
    fw.close()

    model = Meta_model(args).to(device)  # trying this to init optimizer and other params
    model = model.load_checkpoint(model_path)
    acc, specificity, sensitivity = model.goal_testing(all_tasks[target_idx], label_by_organism_list[target_idx])
    fw = open(out_path, "a")
    fw.write('Final Results: ' + "\n" + 'AUC: ' + str(acc) + "\n" + 'Specificity: ' + str(
        specificity) + "\n" + 'Sensitivity: ' + str(sensitivity) + "\n" + "\n")
    fw.close()

    # # Prediction Example:
    # model_path = "result/Human/05 - kidney_model.pth"
    # out_path = "result/Human/05 - kidney.txt"
    # model = Meta_model(args).to(device)  # trying this to init optimizer and other params
    # model = model.load_checkpoint(model_path)
    # acc, specificity, sensitivity = model.goal_testing(all_tasks[target_idx], label_by_organism_list[target_idx])
    # fw = open(out_path, "a")
    # fw.write('Final Results: ' + "\n" + 'AUC: ' + str(acc) + "\n" + 'Specificity: ' + str(
    #     specificity) + "\n" + 'Sensitivity: ' + str(sensitivity) + "\n" + "\n")
    # fw.close()


if __name__ == "__main__":
    # dataset, pretrained_model, graph_model, taskaware_attention, edge_pred, atom_pred, weight, #support
    main("brain", "model_gin/supervised_contextpred.pth", "gin", True, True, True, 0.1, 5)
    # brain / breast / cervix / intestinal / kidney / liver / lung / ovary / prostate / skin
