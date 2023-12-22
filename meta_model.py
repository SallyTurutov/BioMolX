import copy
import torch
import random
import torch.nn as nn
from samples import sample_datasets
from model import GNN, GNN_graphpred
import torch.nn.functional as F
from loader import MoleculeDataset
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)


class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x


class Meta_model(nn.Module):
    def __init__(self, args):
        super(Meta_model, self).__init__()

        self.dataset = args.dataset
        self.k_query = args.k_query
        self.gnn_type = args.gnn_type

        self.emb_dim = args.emb_dim

        self.device = args.device

        self.add_similarity = args.add_similarity
        self.add_selfsupervise = args.add_selfsupervise
        self.add_masking = args.add_masking
        self.add_weight = args.add_weight

        self.batch_size = args.batch_size

        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr

        self.criterion = nn.BCEWithLogitsLoss()

        self.graph_model = GNN_graphpred(args.num_layer, args.emb_dim, 1, JK=args.JK, drop_ratio=args.dropout_ratio,
                                         graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        if not args.input_model_file == "":
            self.graph_model.from_pretrained(args.input_model_file)

        if self.add_selfsupervise:
            self.self_criterion = nn.BCEWithLogitsLoss()

        if self.add_masking:
            self.masking_criterion = nn.CrossEntropyLoss()
            self.masking_linear = nn.Linear(self.emb_dim, 119)

        if self.add_similarity:
            self.Attention = attention(self.emb_dim)

        model_param_group = []
        model_param_group.append({"params": self.graph_model.gnn.parameters()})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": self.graph_model.pool.parameters(), "lr": args.lr * args.lr_scale})
        model_param_group.append(
            {"params": self.graph_model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})

        if self.add_masking:
            model_param_group.append({"params": self.masking_linear.parameters()})

        if self.add_similarity:
            model_param_group.append({"params": self.Attention.parameters()})

        self.optimizer = optim.Adam(model_param_group, lr=args.meta_lr, weight_decay=args.decay)

    def update_params(self, loss, update_lr):
        grads = torch.autograd.grad(loss, self.graph_model.parameters())
        return parameters_to_vector(grads), parameters_to_vector(self.graph_model.parameters()) - parameters_to_vector(
            grads) * update_lr

    def build_negative_edges(self, batch):
        font_list = batch.edge_index[0, ::2].tolist()
        back_list = batch.edge_index[1, ::2].tolist()

        all_edge = {}
        for count, front_e in enumerate(font_list):
            if front_e not in all_edge:
                all_edge[front_e] = [back_list[count]]
            else:
                all_edge[front_e].append(back_list[count])

        negative_edges = []
        for num in range(batch.x.size()[0]):
            if num in all_edge:
                for num_back in range(num, batch.x.size()[0]):
                    if num_back not in all_edge[num] and num != num_back:
                        negative_edges.append((num, num_back))
            else:
                for num_back in range(num, batch.x.size()[0]):
                    if num != num_back:
                        negative_edges.append((num, num_back))

        negative_edge_index = torch.tensor(np.array(random.sample(negative_edges, len(font_list))).T, dtype=torch.long)

        return negative_edge_index

    def set_parameters(self, new_params):
        vector_to_parameters(new_params, self.graph_model.parameters())

    def get_parameters(self):
        return parameters_to_vector(self.graph_model.parameters())

    def eval(self, all_tasks, label_by_organism_list):
        self.graph_model.eval()

        accs = []
        for task, organism_label in zip(all_tasks, label_by_organism_list):
            dataset = MoleculeDataset("datasets/" + self.dataset + "/new/" + str(task + 1),
                                      dataset=self.dataset)
            _, query_dataset = sample_datasets(dataset, self.dataset, task, self.k_query)
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

            device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")

            y_true = []
            y_scores = []
            for step, batch in enumerate(query_loader):
                batch = batch.to(device)

                pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                                  organism_label)
                pred = torch.sigmoid(pred)
                pred = torch.where(pred > 0.5, torch.ones_like(pred), pred)
                pred = torch.where(pred <= 0.5, torch.zeros_like(pred), pred)
                y_scores.append(pred)
                y_true.append(batch.y.view(pred.shape))

            y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
            y_scores = torch.cat(y_scores, dim=0).cpu().detach().numpy()

            roc_list = []
            try:
                roc_list.append(roc_auc_score(y_true, y_scores))
                acc = sum(roc_list) / len(roc_list)
            except Exception as e:
                print('For task ' + str(task) + " got exception: " + str(e))
                acc = 0
            accs.append(acc)

        return accs

    def train_and_eval(self, all_tasks, label_by_organism_list, epoch):
        self.graph_model.train()

        for task, organism_label in zip(all_tasks, label_by_organism_list):
            dataset = MoleculeDataset("datasets/" + self.dataset + "/new/" + str(task + 1),
                                      dataset=self.dataset)
            support_dataset, _ = sample_datasets(dataset, self.dataset, task, self.k_query, epoch)
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
            device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")

            for step, batch in enumerate(support_loader):
                batch = batch.to(device)

                pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                                  organism_label)
                y = batch.y.view(pred.shape).to(torch.float64)

                loss = torch.sum(self.criterion(pred.double(), y)) / pred.size()[0]

                if self.add_selfsupervise:
                    try:
                        positive_score = torch.sum(
                            node_emb[batch.edge_index[0, ::2]] * node_emb[batch.edge_index[1, ::2]], dim=1)

                        negative_edge_index = self.build_negative_edges(batch)
                        negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]],
                                                   dim=1)

                        self_loss = torch.sum(
                            self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(
                                negative_score, torch.zeros_like(negative_score))) / negative_edge_index[0].size()[0]

                        loss += (self.add_weight * self_loss)

                    except:
                        pass

                if self.add_masking:
                    try:
                        mask_num = random.sample(range(0, node_emb.size()[0]), self.batch_size)
                        pred_emb = self.masking_linear(node_emb[mask_num])
                        loss += (self.add_weight * self.masking_criterion(pred_emb.double(), batch.x[mask_num, 0]))
                    except:
                        pass

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.eval(all_tasks, label_by_organism_list)

    def goal_testing(self, goal_task, organism_label):
        print('Goal Testing - task: ' + str(goal_task))
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        self.graph_model.eval()

        dataset = MoleculeDataset("datasets/" + self.dataset + "/new/" + str(goal_task + 1),
                                  dataset=self.dataset)
        _, query_dataset = sample_datasets(dataset, self.dataset, goal_task, self.k_query)
        query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

        y_true = []
        y_scores = []
        for step, batch in enumerate(query_loader):
            batch = batch.to(device)

            pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, organism_label)
            pred = torch.sigmoid(pred)
            pred = torch.where(pred > 0.5, torch.ones_like(pred), pred)
            pred = torch.where(pred <= 0.5, torch.zeros_like(pred), pred)
            y_scores.append(pred)
            y_true.append(batch.y.view(pred.shape))

        y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
        y_scores = torch.cat(y_scores, dim=0).cpu().detach().numpy()

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(y_true, (y_scores > 0.5).astype(int))

        # Extract True Positives, True Negatives, False Positives, and False Negatives
        tn, fp, fn, tp = conf_matrix.ravel()

        # Calculate sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn)

        # Calculate specificity (True Negative Rate)
        specificity = tn / (tn + fp)

        roc_list = []
        roc_list.append(roc_auc_score(y_true, y_scores))
        acc = sum(roc_list) / len(roc_list)

        return acc, specificity, sensitivity

    def get_prediction(self, organism_label):
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        self.graph_model.eval()

        dataset = MoleculeDataset("datasets/test/new/1", dataset=self.dataset)
        query_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

        pred_scores = []
        for step, batch in enumerate(query_loader):
            batch = batch.to(device)

            pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, organism_label)
            pred = torch.sigmoid(pred)
            pred_list = pred.cpu().detach().numpy().tolist()
            flat_list = [item for sublist in pred_list for item in sublist]
            pred_scores = pred_scores + flat_list

        return pred_scores

    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return self

    def run_baseline1(self, all_tasks, label_by_organism_list, epoch):
        self.graph_model.train()

        # Create a list to hold all the data samples
        all_samples = []

        # Iterate through dataloaders and collect samples
        for task, organism_label in zip(all_tasks, label_by_organism_list):
            dataset = MoleculeDataset("datasets/" + self.dataset + "/new/" + str(task + 1),
                                      dataset=self.dataset)
            support_dataset, _ = sample_datasets(dataset, self.dataset, task, self.k_query, epoch)
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

            organism_all_samples = []
            for batch in support_loader:
                # Convert the batch into a list of tensors or desired format
                samples_list = [batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.y, organism_label]
                organism_all_samples.append(samples_list)
            all_samples.extend(organism_all_samples)

        #############################
        ####### Baseline 1 ##########
        #############################

        random.shuffle(all_samples)

        #############################
        #############################

        # Iterate through the shuffled list and process each sample
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")

        for sample in all_samples:
            # Extract individual components from the list
            x, edge_index, edge_attr, batch, y, organism_label = sample

            x, edge_index, edge_attr, batch, y = x.to(device), edge_index.to(device), edge_attr.to(device), batch.to(
                device), y.to(device)

            pred, node_emb = self.graph_model(x, edge_index, edge_attr, batch, organism_label)
            y = y.view(pred.shape).to(torch.float64)

            loss = torch.sum(self.criterion(pred.double(), y)) / pred.size()[0]

            if self.add_selfsupervise:
                try:
                    positive_score = torch.sum(
                        node_emb[edge_index[0, ::2]] * node_emb[edge_index[1, ::2]], dim=1)

                    negative_edge_index = self.build_negative_edges(batch)
                    negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]],
                                               dim=1)

                    self_loss = torch.sum(
                        self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(
                            negative_score, torch.zeros_like(negative_score))) / negative_edge_index[0].size()[0]

                    loss += (self.add_weight * self_loss)

                except:
                    pass

            if self.add_masking:
                try:
                    mask_num = random.sample(range(0, node_emb.size()[0]), self.batch_size)
                    pred_emb = self.masking_linear(node_emb[mask_num])
                    loss += (self.add_weight * self.masking_criterion(pred_emb.double(), x[mask_num, 0]))
                except:
                    pass

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self.eval(all_tasks, label_by_organism_list)

    def run_baseline2(self, all_tasks, label_by_organism_list, epoch):
        self.graph_model.train()

        # Create a list to hold all the data samples
        all_organism_lists = []

        # Iterate through dataloaders and collect samples
        for task, organism_label in zip(all_tasks, label_by_organism_list):
            dataset = MoleculeDataset("datasets/" + self.dataset + "/new/" + str(task + 1),
                                      dataset=self.dataset)
            support_dataset, _ = sample_datasets(dataset, self.dataset, task, self.k_query, epoch)
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

            organism_all_samples = []
            for batch in support_loader:
                # Convert the batch into a list of tensors or desired format
                samples_list = [batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.y, organism_label]
                organism_all_samples.append(samples_list)
            all_organism_lists.append(organism_all_samples)

        #############################
        ####### Baseline 2 ##########
        #############################

        result = []
        list_of_lists = all_organism_lists
        total_items = sum(len(sublist) for sublist in list_of_lists)
        weights = [1.0 / (i + 1) for i in range(len(list_of_lists))]

        while len(result) < total_items:
            prob_distribution = [w / sum(weights) for w in weights]
            sublist_index = random.choices(range(len(list_of_lists)), prob_distribution)[0]
            while len(list_of_lists[sublist_index]) == 0:
                sublist_index = random.choices(range(len(list_of_lists)), prob_distribution)[0]
            item = list_of_lists[sublist_index].pop()
            result.append(item)
        all_samples = result

        #############################
        #############################

        # Iterate through the shuffled list and process each sample
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")

        for sample in all_samples:
            # Extract individual components from the list
            x, edge_index, edge_attr, batch, y, organism_label = sample

            x, edge_index, edge_attr, batch, y = x.to(device), edge_index.to(device), edge_attr.to(device), batch.to(
                device), y.to(device)

            pred, node_emb = self.graph_model(x, edge_index, edge_attr, batch, organism_label)
            y = y.view(pred.shape).to(torch.float64)

            loss = torch.sum(self.criterion(pred.double(), y)) / pred.size()[0]

            if self.add_selfsupervise:
                try:
                    positive_score = torch.sum(
                        node_emb[edge_index[0, ::2]] * node_emb[edge_index[1, ::2]], dim=1)

                    negative_edge_index = self.build_negative_edges(batch)
                    negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]],
                                               dim=1)

                    self_loss = torch.sum(
                        self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(
                            negative_score, torch.zeros_like(negative_score))) / negative_edge_index[0].size()[0]

                    loss += (self.add_weight * self_loss)

                except:
                    pass

            if self.add_masking:
                try:
                    mask_num = random.sample(range(0, node_emb.size()[0]), self.batch_size)
                    pred_emb = self.masking_linear(node_emb[mask_num])
                    loss += (self.add_weight * self.masking_criterion(pred_emb.double(), x[mask_num, 0]))
                except:
                    pass

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self.eval(all_tasks, label_by_organism_list)
