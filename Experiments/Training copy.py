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
from Experiments.utils import str2bool

import argparse

# 配置日志记录
def setup_logging():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./Log/{timestamp}.log"
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
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    return parser.parse_args()


# 初始化模型和超参数
def initialize_args(args_parser):
    args = {
        'use_uncertainty': args_parser.use_uncertainty,
        'primary_task_weight': args_parser.primary_task_weight,
        'use_gib': args_parser.use_gib,
        'use_primary_centered_gate': args_parser.use_primary_centered_gate,
        'use_grl': args_parser.use_grl,
        'lr': args_parser.lr,
        'device': f"cuda:{args_parser.device}" if torch.cuda.is_available() else "cpu",
        'atom_data_field': 'atom',
        'classification_metric_name': 'roc_auc',
        'regression_metric_name': 'r2',
        'num_epochs': args_parser.num_epochs,
        'patience': args_parser.patience,
        'mode': 'higher',
        'task_name': 'admet',
        'data_name': 'admet',
        'times': 1,
        'bin_path': './Data/admet.bin',
        'group_path': './Data/admet_group.csv',
        'select_task_list': args_parser.select_task_list or ['CYP2C9', 'CYP2D6', 'ESOL', 'logD'],
        'beta': 0.0001
    }
    return args

# 解析命令行参数
parser = argparse.ArgumentParser(description='Run MTGL_ADMET with specified task list.')
parser.add_argument('--select_task_list', type=str, nargs='+', default=[], help='List of selected tasks')
parser.add_argument('--use_uncertainty', type=str2bool, default=True, help='Whether to use uncertainty weighting')
parser.add_argument('--primary_task_weight', type=float, default=10, help='primary uncertainty weighting coefficient')
parser.add_argument('--use_gib', type=str2bool, default=True, help='Whether to use gib module')
parser.add_argument('--use_grl', type=str2bool, default=False, help='Whether to use GradientReversalLayer module')
parser.add_argument('--use_primary_centered_gate', type=str2bool, default=True, help='Whether to use primary centered gate module')
parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
parser.add_argument('--lr', type=float, default=1e-3, help='lr')
args_parser = parser.parse_args()

# fix parameters of model
args = {}
args['use_uncertainty'] = args_parser.use_uncertainty
args['primary_task_weight'] = args_parser.primary_task_weight
args['use_gib'] = args_parser.use_gib
args['use_primary_centered_gate'] = args_parser.use_primary_centered_gate
args['use_grl'] = args_parser.use_grl
args['lr'] = args_parser.lr
logging.info(f'use gib module: {args["use_gib"]}')
logging.info(f'use uncertainty: {args["use_uncertainty"]}')
logging.info(f'use primary centered gate: {args["use_primary_centered_gate"]}')
logging.info(f'use grl: {args["use_grl"]}')
    
adversarial_weight = 0.1  # 对抗损失的权重系数，可以通过实验调整
   
args['device'] = f"cuda:{args_parser.device}" if torch.cuda.is_available() else "cpu"

args['atom_data_field'] = 'atom'
args['classification_metric_name'] = 'roc_auc'
args['regression_metric_name'] = 'r2'
# model parameter
args['num_epochs'] = 300
args['patience'] = 50
args['mode'] = 'higher'

# task name (model name)
args['task_name'] = 'admet'
args['data_name'] = 'admet'
args['times'] = 1

# selected task, generate select task index, task class, and classification_num
# args['select_task_list'] = ['Respiratory toxicity','CYP2C9', 'CYP2D6', 'Caco-2 permeability','PPB']  # change

args['bin_path'] = './Data/' + args['data_name'] + '.bin'
args['group_path'] = './Data/' + args['data_name'] + '_group.csv'

args['select_task_index'] = []
args['classification_num'] = 0
args['regression_num'] = 0

args['all_task_list'] = ['HIA','OB','p-gp inhibitor','p-gp substrates',	'BBB',
                             'Respiratory toxicity','Hepatotoxicity', 'half-life', 'CL',
                         'Cardiotoxicity1','Cardiotoxicity10', 'Cardiotoxicity30', 'Cardiotoxicity5',
                            'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4',
                            'Acute oral toxicity (LD50)','IGC50','ESOL','logD',	'Caco-2 permeability','PPB']  # change

logging.info(f'args_parser.select_task_list: {args_parser.select_task_list}')
if not args_parser.select_task_list:
    args_parser.select_task_list = ['CYP2C9', 'CYP2D6', 'ESOL', 'logD']

args['select_task_list'] = args_parser.select_task_list
logging.info(f"args['select_task_list']: {args['select_task_list']}")   



args['beta'] = 0.0001

#### for debug
# args['select_task_list'] = ['CYP2D6', 'CYP2C9', 'PPB', 'ESOL'] 
# args['select_task_list'] = ['CYP2C9','CYP2D6', 'ESOL','PPB'] 
# args['select_task_list'] = ['CYP2C9','CYP2D6', 'ESOL','logD'] 
# args['select_task_list'] = ['CYP2D6', 'Caco-2 permeability','PPB'] 

# args['num_epochs'] = 2
# args['patience'] = 1
# args['select_task_list'] = ['HIA','OB'] 
# args['bin_path'] = './Data/' + args['data_name'] + '_demo.bin'
# args['group_path'] = './Data/' + args['data_name'] + '_group_demo.csv'
#### for debug


# select_task_list 是初始给定的多任务列表， select_task_name是实际执行时的任务列表，select_task_index是实际执行的任务列表的index
args['select_task_name'] = []
# generate select task index
for index, task in enumerate(args['all_task_list']):
    if task in args['select_task_list']:
        args['select_task_index'].append(index)
        args['select_task_name'].append(task)
logging.info(f"args['select_task_index']: {args['select_task_index']}")
logging.info(f"args['select_task_name']: {args['select_task_name']}")

PRIMARY_TASK = args['select_task_list'][0]
PRIMARY_TASK_INDEX = args['select_task_name'].index(PRIMARY_TASK)
logging.info(f"Primary Task: {PRIMARY_TASK}, Index: {PRIMARY_TASK_INDEX}")

# generate classification_num
for task in args['select_task_list']:
    if task in ("Caco-2 permeability","PPB","Acute oral toxicity (LD50)","IGC50","ESOL","logD"):
        args['regression_num'] = args['regression_num'] + 1
    else:
        args['classification_num'] = args['classification_num'] + 1

# generate classification_num
if args['classification_num'] != 0 and args['regression_num'] != 0:
    args['task_class'] = 'classification_regression'
if args['classification_num'] != 0 and args['regression_num'] == 0:
    args['task_class'] = 'classification'
if args['classification_num'] == 0 and args['regression_num'] != 0:
    args['task_class'] = 'regression'
logging.info(f'Classification task: {args["classification_num"]}, Regression Task: {args["regression_num"]}')


result_pd = pd.DataFrame(columns=args['select_task_list']+['group'] + args['select_task_list']+['group']
                         + args['select_task_list']+['group'])
all_times_train_result = []
all_times_val_result = []
all_times_test_result = []
for time_id in range(args['times']):
    set_random_seed(3407)
    one_time_train_result = []  
    one_time_val_result = []
    one_time_test_result = []
    logging.info(f'{args["task_name"]}, {time_id+1}/{args["times"]} time')
    train_set, val_set, test_set, task_number = load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        select_task_index=args['select_task_index']
    )
    logging.info(f'Task number: {task_number}')
    logging.info("Molecule graph generation is complete !")
    train_loader = DataLoader(dataset=train_set, batch_size=128,shuffle=True,collate_fn=collate_molgraphs)
    val_loader = DataLoader(dataset=val_set, batch_size=128,shuffle=True, collate_fn=collate_molgraphs)
    test_loader = DataLoader(dataset=test_set, batch_size=128,collate_fn=collate_molgraphs)
    pos_weight_np = pos_weight(train_set, classification_num=args['classification_num']).to(args['device'])
    loss_criterion_cs = [
        torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_np[i])
        for i in range(args['classification_num'])
    ]
    loss_criterion_r = torch.nn.MSELoss(reduction='none')
    args['num_tasks_in_use'] = args['regression_num'] + args['classification_num']
    model = MTGL_ADMET(PRIMARY_TASK_INDEX, in_feats=40, hidden_feats=64,gnn_out_feats_0=128, gnn_out_feats=64,dropout=0.5,classifier_hidden_feats=128,
                       device=args['device'], n_tasks=task_number,num_gates=args['num_tasks_in_use']-1,
                       use_primary_centered_gate = args['use_primary_centered_gate'], 
                       ) 

    awl = AutomaticWeightedLoss(args['num_tasks_in_use'],args['use_uncertainty'],PRIMARY_TASK_INDEX,args['primary_task_weight'])	# we h    
    logging.info(f"Uncertainty weight params values: {[param.data.cpu().numpy() for param in awl.parameters()]}")
    optimizer = Adam([
                {'params': model.parameters(), "lr":0.001, "weight_decay":10**-5},
                {'params': awl.parameters(),  'weight_decay': 0}	
            ])
    
    stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'], mode=args['mode'])
    model.to(args['device'])
    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch_heterogeneous(args, epoch, model, train_loader, loss_criterion_cs, loss_criterion_r, optimizer, awl, adversarial_weight)

        # Validation and early stop
        validation_result = run_an_eval_epoch_heterogeneous(args, model, val_loader)
        
        # Assuming the primary task's validation result is the first element
        if len(validation_result) == 1:
            # If there's only one task, use its validation result directly
            weighted_val_score = validation_result[0]
        else:
            primary_task_weight = 0.7
            other_tasks_weight = (1 - primary_task_weight) / (len(validation_result) - 1)
            weighted_val_score = primary_task_weight * validation_result[0] + other_tasks_weight * sum(validation_result[1:])
        early_stop = stopper.step(weighted_val_score, model)
        logging.info(f'epoch {epoch + 1}/{args["num_epochs"]}, weighted validation {weighted_val_score:.4f}, best weighted validation {stopper.best_score:.4f} validation result: {validation_result}')
        # val_score = np.mean(validation_result)
        # Use primary task's validation result for early stopping
        # early_stop = stopper.step(val_score, model)
        
        # primary_task_val_score = validation_result[0]
        # early_stop = stopper.step(primary_task_val_score, model)
        # logging.info(f'epoch {epoch + 1}/{args["num_epochs"]},primary validation {primary_task_val_score:.4f},primary best validation {stopper.best_score:.4f} validation result: {validation_result}')
        if early_stop:
            break
    stopper.load_checkpoint(model)
    test_score = run_an_eval_epoch_heterogeneous(args, model, test_loader)
    train_score = run_an_eval_epoch_heterogeneous(args, model, train_loader)
    val_score = run_an_eval_epoch_heterogeneous(args, model, val_loader)
    # deal result
    result = train_score + ['training'] + val_score + ['valid'] + test_score + ['test']
    result_pd.loc[time_id] = result
    logging.info(f'********************************{args["task_name"]}, {time_id+1}_times_result*******************************')
    logging.info(args['select_task_name'])
    logging.info(f"training_result: {train_score}")
    logging.info(f"val_result: {val_score}")
    logging.info(f"test_result: {test_score}")
    logging.info(f"use uncertainty loss: {args['use_uncertainty']}")
    logging.info(f"Uncertainty weight params values: {[param.data.cpu().numpy() for param in awl.parameters()]}")
    logging.info(f'use gib module: {args["use_gib"]}')
    logging.info(f'use primary centered gate: {args["use_primary_centered_gate"]}')

# 生成文件名
filename = f"./Result/folder1/{PRIMARY_TASK}_gib_{args['use_gib']}_uncert_{args['use_uncertainty']}_gate_{args['use_primary_centered_gate'] }_result_{timestamp}.csv"
result_pd.to_csv(filename, index=None)
logging.info(f"Results saved to {filename}")