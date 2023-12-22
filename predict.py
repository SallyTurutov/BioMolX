from meta_model import Meta_model
import pandas as pd
import argparse
import torch


parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=5,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=2000,
                    help='number of epochs to train (default: 100)')
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
parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')

parser.add_argument('--num_tasks', type=int, default=12, help="# of tasks")
parser.add_argument('--n_way', type=int, default=2, help="n_way of dataset")
parser.add_argument('--m_support', type=int, default=5, help="size of the support dataset")
parser.add_argument('--k_query', type=int, default=32, help="size of querry datasets")
parser.add_argument('--meta_lr', type=float, default=0.001)
parser.add_argument('--update_lr', type=float, default=0.4)
parser.add_argument('--add_similarity', type=bool, default=False)
parser.add_argument('--add_selfsupervise', type=bool, default=False)
parser.add_argument('--add_weight', type=float, default=0.1)

args = parser.parse_args()
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.runseed)

args.input_model_file = "model_gin/supervised_contextpred.pth"
args.gnn_type = "gin"
args.add_similarity = True
args.add_selfsupervise = True
args.add_masking = True
args.add_weight = 0.1
args.m_support = 5

args.dataset = "brain"
checkpoint = 'result/modelbrain.pth'
test_file_path = 'Original_datasets/test/raw/test.csv'

test_data = pd.read_csv(test_file_path)

model = Meta_model(args).to(device)
model = model.load_checkpoint(checkpoint)
print('Loading Checkpoint from: ' + str(checkpoint))
prediction = model.get_prediction()

results_df = pd.DataFrame({'smiles': test_data['smiles'], 'predicted_label': prediction})

print(results_df)