import pandas as pd
import torch
from rdkit.Chem.Draw import SimilarityMaps
import random
import numpy as np
import os
import time
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from torch_geometric.data import DataLoader

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import pandas as pd

import sys
import os


# 添加当前脚本目录的父目录到sys.path
sys.path.append('/home/*/project/MTGL-ADMET/')
sys.path.append('/home/*/project/MTGL-ADMET/Experiments')
import logging
import argparse
import datetime

from Experiments.model import *
from Experiments.paremeters import *
from Data.data_prepare import *
from Experiments.AutomaticWeightedLoss import *
from Experiments.utils import *

import argparse
import datetime
import logging

import sys
import os
# 配置日志记录
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 生成随机数
random_number = random.randint(1000, 9999)
# 合并时间戳和随机数，生成唯一的日志文件名
log_filename = f"./Log/{timestamp}_{random_number}.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)




# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Run MTGL_ADMET with specified task list.')
    parser.add_argument('--select_task_list', type=str, nargs='+', default=[], help='List of selected tasks')
    parser.add_argument('--use_uncertainty', type=str2bool, default=True, help='Whether to use uncertainty weighting')
    parser.add_argument('--primary_task_weight', type=float, default=10, help='primary uncertainty weighting coefficient')
    parser.add_argument('--use_gib', type=str2bool, default=True, help='Whether to use gib module')
    parser.add_argument('--use_grl', type=str2bool, default=False, help='Whether to use GradientReversalLayer module')
    parser.add_argument('--use_primary_centered_gate', type=str2bool, default=True, help='Whether to use primary centered gate module')
    parser.add_argument('--device', type=int, default=4, help='which gpu to use if any (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    # 在 parse_args() 中添加新的超参数
    parser.add_argument('--in_feats', type=int, default=40, help='Input feature dimension')
    parser.add_argument('--hidden_feats', type=int, default=64, help='Hidden feature dimension')
    parser.add_argument('--conv2_out_dim', type=int, default=128, help='Output dimension of the second convolutional layer')
    parser.add_argument('--gnn_out_feats', type=int, default=64, help='GNN output feature dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--classifier_hidden_feats', type=int, default=128, help='Hidden feature dimension of the classifier')
    # 在 parse_args() 中添加新的超参数
    parser.add_argument('--times', type=int, default=1, help='Number of times to run the experiment')
    parser.add_argument('--beta', type=float, default=0.0001, help='KL loss coefficient for GIB module')
    parser.add_argument('--model_save_path', type=str, default=None, help='model_save_path')

    return parser.parse_args()

# 直接将解析结果赋值给 args
args = parse_args()
# 现在你可以通过 args.xxx 来访问所有解析的参数
args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
args.atom_data_field = 'atom'
args.classification_metric_name = 'roc_auc'
args.regression_metric_name = 'r2'
args.mode = 'higher'
args.task_name = 'admet'
args.data_name = 'admet'
args.bin_path = './Data/admet.bin'
args.group_path = './Data/admet_group.csv'
args.select_task_list = args.select_task_list or ['CYP2C9', 'CYP2D6', 'ESOL', 'logD']

# PATH = './Model_Save/new/admet_early_stop_20240815_131938_4294.pth'
# args.select_task_list = ['IGC50','CYP2D6']
# args.gnn_out_feats = 128
# args.dropout= 0.2
# args.beta= 1e-1 
PATH = './Model_Save/new/admet_early_stop_20240815_131611_9747.pth'
PATH = args.model_save_path

# args.select_task_list = ['Caco-2 permeability', 'OB' ,'ESOL']
# selected task, generate select task index, task class, and classification_num
# args.select_task_list'] = ['Respiratory toxicity','CYP2C9', 'CYP2D6', 'Caco-2 permeability','PPB']  # change
args.select_task_index = []
args.classification_num = 0
args.regression_num = 0

args.all_task_list = ['HIA','OB','p-gp inhibitor','p-gp substrates',	'BBB',
                             'Respiratory toxicity','Hepatotoxicity', 'half-life', 'CL',
                         'Cardiotoxicity1','Cardiotoxicity10', 'Cardiotoxicity30', 'Cardiotoxicity5',
                            'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4',
                            'Acute oral toxicity (LD50)','IGC50','ESOL','logD',	'Caco-2 permeability','PPB']  # change

# select_task_list 是初始给定的多任务列表， select_task_name是实际执行时的任务列表，select_task_index是实际执行的任务列表的index
args.select_task_name = []
# generate select task index
for index, task in enumerate(args.all_task_list):
    if task in args.select_task_list:
        args.select_task_index.append(index)
        args.select_task_name.append(task)
logging.info(f"args.select_task_list: {args.select_task_list}")
logging.info(f"args.select_task_index: {args.select_task_index}")
logging.info(f"args.select_task_name: {args.select_task_name}")

PRIMARY_TASK = args.select_task_list[0]
PRIMARY_TASK_INDEX = args.select_task_name.index(PRIMARY_TASK)
logging.info(f"Primary Task: {PRIMARY_TASK}, Index: {PRIMARY_TASK_INDEX}")

# generate classification_num
for task in args.select_task_list:
    if task in ("Caco-2 permeability","PPB","Acute oral toxicity (LD50)","IGC50","ESOL","logD"):
        args.regression_num = args.regression_num + 1
    else:
        args.classification_num = args.classification_num + 1

# generate classification_num
if args.classification_num != 0 and args.regression_num != 0:
    args.task_class = 'classification_regression'
if args.classification_num != 0 and args.regression_num == 0:
    args.task_class = 'classification'
if args.classification_num == 0 and args.regression_num != 0:
    args.task_class = 'regression'
logging.info(f'Classification task: {args.classification_num}, Regression Task: {args.regression_num}')

result_pd = pd.DataFrame(columns=args.select_task_list+['group'] + args.select_task_list+['group']
                         + args.select_task_list+['group'])
all_times_train_result = []
all_times_val_result = []
all_times_test_result = []

logging.info("Hyperparameters:")
print_args(args)

set_random_seed(3407)
one_time_train_result = []  
one_time_val_result = []
one_time_test_result = []
train_set, val_set, test_set, task_number = load_graph_from_csv_bin_for_splited(
    bin_path=args.bin_path,
    group_path=args.group_path,
    select_task_index=args.select_task_index
)
logging.info(f'Task number: {task_number}')
logging.info("Molecule graph generation is complete !")
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,shuffle=True,collate_fn=collate_molgraphs)
val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size,shuffle=True, collate_fn=collate_molgraphs)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,collate_fn=collate_molgraphs)
pos_weight_np = pos_weight(train_set, classification_num=args.classification_num).to(args.device)
loss_criterion_cs = [
    torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_np[i])
    for i in range(args.classification_num)
]
loss_criterion_r = torch.nn.MSELoss(reduction='none')
args.num_tasks_in_use = args.regression_num + args.classification_num

IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 600,600
torch.set_num_threads(2)
# df = pd.read_csv(f'data/raw_data/ZhangDDI_train.csv', sep=",")

def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print("Loading dataset...")
start = time.time()
# Load dataset
# train_set = torch.load("./data/processed/{}_train.pt".format(args.dataset))

print("Dataset Loaded! ({:.4f} sec)".format(time.time() - start))
model = MTGL_ADMET(PRIMARY_TASK_INDEX, 
                    in_feats=args.in_feats, 
                    hidden_feats=args.hidden_feats,
                    conv2_out_dim=args.conv2_out_dim, 
                    gnn_out_feats=args.gnn_out_feats,
                    dropout=args.dropout, 
                    classifier_hidden_feats=args.classifier_hidden_feats,
                    device=args.device, n_tasks=task_number,num_gates=args.num_tasks_in_use-1,
                    use_primary_centered_gate = args.use_primary_centered_gate, 
                    ).to(args.device)

model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu'))['model_state_dict'])
# train_loader = DataLoader(train_set, batch_size = 1, shuffle=False)
train_loader = DataLoader(dataset=train_set, batch_size=1,shuffle=False,collate_fn=collate_molgraphs)
def calculate_similarity(vector1, vector2):
    """Calculates cosine similarity between two vectors."""
    vector1 = vector1.flatten()  # Flatten to 1D array
    vector2 = vector2.flatten()  # Flatten to 1D array
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

# Get current timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
i=0
z=0
ls = [193]
import matplotlib.pyplot as plt
for batch_id, batch_data in enumerate(train_loader):
    smiles, bg, labels, mask = batch_data
    smile = smiles[0]
    mask = mask.float().to(args.device)
    labels.float().to(args.device)
    bg = bg.to(args.device)  # Move bg to the correct device
    # print(labels.shape)
    atom_feats = bg.ndata.pop(args.atom_data_field).float().to(args.device)
    lambda_pos_list = model.get_subgraph_list(bg,atom_feats)
    mol = Chem.MolFromSmiles(smile)
    solu_atom_num=mol.GetNumAtoms()
    mol.RemoveAllConformers()

    print(smile)
    fig, axs = plt.subplots(1, len(lambda_pos_list), figsize=(5 * len(lambda_pos_list), 5))
    # Save atom weights for the current task
    folder_path = f"Result/explatiable/foler2/{timestamp}/{i}/"
    weights_folder = f"{folder_path}weights/"
    os.makedirs(weights_folder, exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)
    similarity_data = []
     # Save SMILES to a file
    with open(f"{folder_path}smiles.txt", 'w') as f:
        f.write(smile)
    # print(lambda_pos_list)
    for j in range(len(lambda_pos_list)):
        current_task_atom_weighted_list = lambda_pos_list[j].cpu().detach().numpy()
        current_task_index =  args.select_task_index[j]
        current_task_name = args.all_task_list[current_task_index]
        
        # Save atom weights for the current task
        np.save(weights_folder + current_task_name + '_weights.npy', current_task_atom_weighted_list)
         # Create a DataFrame and save to CSV
        weights_df = pd.DataFrame(current_task_atom_weighted_list, columns=['Atom_Weight'])
        weights_df.to_csv(weights_folder + current_task_name + '_weights.csv', index=False)

            # Calculate similarity between the current task and others
        for k in range(j + 1, len(lambda_pos_list)):
            next_task_atom_weighted_list = lambda_pos_list[k].cpu().detach().numpy()
            similarity = calculate_similarity(current_task_atom_weighted_list, next_task_atom_weighted_list)
            print(f"Similarity between {current_task_name} and {args.all_task_list[args.select_task_index[k]]}: {similarity:.4f}")
            similarity_data.append({
                            'Task1': current_task_name,
                            'Task2': args.all_task_list[args.select_task_index[k]],
                            'Similarity': similarity
                        })
            print(f"Similarity between {current_task_name} and {args.all_task_list[args.select_task_index[k]]}: {similarity:.4f}")

        ax = axs[j] if len(lambda_pos_list) > 1 else axs  # Handle case with single task
        solvent_fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, current_task_atom_weighted_list, colorMap='RdBu', alpha=0.05, size=(200, 200))
        solvent_fig.savefig(folder_path + current_task_name + '.png', bbox_inches='tight', dpi=600)
        ax.imshow(solvent_fig.canvas.buffer_rgba())
        ax.set_title(current_task_name)
        ax.axis('off')
        
    combined_fig_path = folder_path + 'combined_task_visualization.png'
    fig.savefig(combined_fig_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    
    # Save similarity data to CSV
    similarity_df = pd.DataFrame(similarity_data)
    similarity_df.to_csv(f"{folder_path}similarity_data.csv", index=False)
    
    i=i+1
        
        
