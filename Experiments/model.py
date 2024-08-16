import torch
import torch.nn.functional as F
import dgl
import numpy as np
import random
from dgl.readout import sum_nodes
from dgl.nn.pytorch.conv import GraphConv
from torch import nn
import pandas as pd
from functools import reduce
import torch.nn.init as init
from torch.nn import MultiheadAttention
class ResGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=F.relu, residual=True, batchnorm=True):
        super(ResGCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv_layer = GraphConv(in_feats, out_feats, bias=True, activation=activation, allow_zero_in_degree=True)
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, node_feats):
        
        new_feats = self.graph_conv_layer(bg, node_feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        del res_feats
        torch.cuda.empty_cache()
        return new_feats


class NewResGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=F.relu, residual=True, batchnorm=True, 
                 dynamic_residual=False, use_attention=False, feature_dropout=0.0):
        super(NewResGCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv_layer = GraphConv(in_feats, out_feats, bias=True, activation=activation, allow_zero_in_degree=True)
        self.residual = residual
        self.dynamic_residual = dynamic_residual
        self.use_attention = use_attention
        self.feature_dropout = feature_dropout

        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

        if use_attention:
            self.attention_layer = nn.Linear(out_feats, 1)

    def forward(self, bg, node_feats):
        new_feats = self.graph_conv_layer(bg, node_feats)
        
        # Apply dynamic residual connection
        if self.residual:
            if self.dynamic_residual:
                res_weights = torch.sigmoid(self.res_connection(node_feats))
                res_feats = self.activation(self.res_connection(node_feats)) * res_weights
            else:
                res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        
        # Apply feature dropout
        if self.feature_dropout > 0.0:
            new_feats = F.dropout(new_feats, p=self.feature_dropout, training=self.training)
        
        # Apply attention mechanism
        if self.use_attention:
            attention_weights = torch.sigmoid(self.attention_layer(new_feats))
            new_feats = new_feats * attention_weights
        
        # Apply batch normalization
        if self.bn:
            new_feats = self.bn_layer(new_feats)

        torch.cuda.empty_cache()
        return new_feats

from dgl.nn.pytorch import SAGEConv
class GraphSAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=F.relu, residual=True, batchnorm=True):
        super(GraphSAGELayer, self).__init__()
        self.activation = activation
        self.sage_conv = SAGEConv(in_feats, out_feats, 'mean')
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, node_feats):
        new_feats = self.sage_conv(bg, node_feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats



class WeightAndSum(nn.Module):
    def __init__(self, in_feats, task_num=1, attention=True, return_weight=False):
        super(WeightAndSum, self).__init__()
        self.attention = attention
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_weight=return_weight
        self.atom_weighting_specific = nn.ModuleList([self.atom_weight(self.in_feats) for _ in range(self.task_num)])
        self.shared_weighting = self.atom_weight(self.in_feats)
    def forward(self, bg, feats):
        feat_list = []
        atom_list = []
        # cal specific feats
        for i in range(self.task_num):
            with bg.local_scope():
                bg.ndata['h'] = feats
                weight = self.atom_weighting_specific[i](feats)
                bg.ndata['w'] = weight
                specific_feats_sum = sum_nodes(bg, 'h', 'w')
                atom_list.append(bg.ndata['w'])
            feat_list.append(specific_feats_sum)

        # cal shared feats
        with bg.local_scope():
            bg.ndata['h'] = feats
            bg.ndata['w'] = self.shared_weighting(feats)
            shared_feats_sum = sum_nodes(bg, 'h', 'w')
        # feat_list.append(shared_feats_sum)
        if self.attention:
            if self.return_weight:
                return feat_list, atom_list
            else:
                return feat_list
        else:
            return shared_feats_sum

    def atom_weight(self, in_feats):
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
            )
        
from torch_geometric.nn import Set2Set
from torch_scatter import scatter_mean, scatter_add, scatter_std

class GraphInformationBottleneckModule(nn.Module):
    def __init__(self,
                device,
                node_input_dim=64,
                node_hidden_dim=64,
                num_step_set2set = 2):
        super(GraphInformationBottleneckModule, self).__init__()
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.device = device
        self.compressor = nn.Sequential(
            nn.Linear(self.node_input_dim, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 1)
            )

        self.predictor = nn.Sequential(
            nn.Linear(2*self.node_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.num_step_set2set = num_step_set2set
        self.set2set = Set2Set(self.node_hidden_dim, self.num_step_set2set)
        
    def compress(self, features):
        p = self.compressor(features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()

        return gate_inputs, p
    
    def forward(self, features, bg):
        # 计算 lambda_pos 和 lambda_neg
        lambda_pos, p = self.compress(features)
        lambda_pos = lambda_pos.reshape(-1, 1)
        lambda_neg = 1 - lambda_pos

        # 获取 preserve_rate
        preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

        # 克隆并分离 features
        static_feature = features.clone().detach()
        
        # 获取批次索引
        batch_num_nodes = bg.batch_num_nodes()
        batch_index = torch.cat([torch.full((num,), i, dtype=torch.long) for i, num in enumerate(batch_num_nodes)]).to(features.device)
        
        # 调试输出以确保索引和特征的长度匹配
        # print(f"features shape: {features.shape}")
        # print(f"static_feature shape: {static_feature.shape}")
        # print(f"batch_index shape: {batch_index.shape}")
        
        # 计算均值和标准差
        node_feature_mean = scatter_mean(static_feature, batch_index, dim=0)[batch_index]
        node_feature_std = scatter_std(static_feature, batch_index, dim=0)[batch_index]

        # 生成噪声特征
        noisy_node_feature_mean = lambda_pos * features + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std
        noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std

        # 使用 set2set 方法处理噪声特征
        noisy_subgraphs = self.set2set(noisy_node_feature, batch_index)

        # 计算 KL 损失
        epsilon = 1e-7
        KL_tensor = 0.5 * scatter_add(((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim=1), batch_index).reshape(-1, 1) + \
                    scatter_add((((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2), batch_index, dim=0)
        KL_Loss = torch.mean(KL_tensor)
        # print("KL loss:",KL_Loss)
        # Prediction Y
        final_features = noisy_subgraphs
        # print("final_features shape:",final_features.shape)
        predictions = self.predictor(final_features)
        # print("predictions shape:",predictions.shape)
        return predictions, KL_Loss, preserve_rate, lambda_pos
    
    
    
from GRL import *
class MTGL_ADMET(nn.Module):
    # def __init__(self,prim_index, in_feats,hidden_feats,device,gnn_out_feats=64,n_tasks=None,  return_weight=False,
    #              classifier_hidden_feats=128, dropout=0.,num_gates=4,use_primary_centered_gate=False, lambda_=0.5, num_heads=4):
    def __init__(self, prim_index, in_feats, hidden_feats, device, gnn_out_feats=64,conv2_out_dim=128, n_tasks=None, return_weight=False,
                 classifier_hidden_feats=128, dropout=0.5, num_gates=4, use_primary_centered_gate=False,num_heads=4,):
        super(MTGL_ADMET, self).__init__()
        self.prim_index = prim_index
        self.use_primary_centered_gate = use_primary_centered_gate
        self.device = device
        # Number of Auxiliary tasks
        self.num_gates = num_gates
        self.task_num = n_tasks
        self.return_weight = return_weight 
        self.weighted_sum_readout = WeightAndSum(gnn_out_feats, self.task_num, return_weight=self.return_weight)
        
        # # Two-layer ResGCN
        self.conv1 = ResGCNLayer(in_feats, hidden_feats)
        self.conv2 = ResGCNLayer(hidden_feats, conv2_out_dim)
        self.conv3 = ResGCNLayer(conv2_out_dim, gnn_out_feats)
        
        # self.conv1 = NewResGCNLayer(in_feats, hidden_feats, dynamic_residual=True, use_attention=True, feature_dropout=0.2)
        # self.conv2 = NewResGCNLayer(hidden_feats, conv2_out_dim, dynamic_residual=True, use_attention=True, feature_dropout=0.2)
        # self.conv3 = NewResGCNLayer(conv2_out_dim, gnn_out_feats, dynamic_residual=True, use_attention=True, feature_dropout=0.2)

  
        self.gates = nn.ModuleList()
        for i in range(self.task_num):
                if i == self.prim_index:
                    self.gates.append(None)
                else:
                    self.gates.append(nn.Linear(gnn_out_feats, 2))
                    
        self.gates_git = nn.ModuleList()
        for i in range(self.task_num):
            if i == self.prim_index:
                self.gates_git.append(None)
            else:
                self.gates_git.append(nn.Linear(gnn_out_feats, 2))

        self.fc_in_feats = gnn_out_feats 
        for i in range(self.task_num):
            self.fine_f = nn.ModuleList([self.fc_layer(dropout,gnn_out_feats, gnn_out_feats) for _ in range(self.task_num)])

        self.fc_layers1 = nn.ModuleList([self.fc_layer(dropout,self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.output_layer1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)])
        
        self.fc_layers_git1 = nn.ModuleList([self.fc_layer(dropout,self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers_git2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.output_layer_git1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)])
        
        # self.graph_information_bottleneck_module = GraphInformationBottleneckModule(self.device,gnn_out_feats,gnn_out_feats)
        self.graph_information_bottleneck_module = nn.ModuleList(
            [self.gib_layer(device,gnn_out_feats) for _ in range(self.task_num)])
        
        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                # torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
                    
    def forward(self, bg, node_feats):
        bg = bg.to(node_feats.device)
        node_feats = self.conv1(bg, node_feats)
        node_feats = self.conv2(bg, node_feats)
        node_feats = self.conv3(bg, node_feats)
        # gib 
        lambda_pos_list = []
        gib_feats_list, KL_loss_all, preserve_rate_all = [], [], []
        for i in range(self.task_num):
            gib_feat, KL_loss, preserve_rate,lambda_pos = self.graph_information_bottleneck_module[i](node_feats, bg)
            lambda_pos_list.append(lambda_pos)
            gib_feats_list.append(gib_feat)
            KL_loss_all.append(KL_loss)
            preserve_rate_all.append(preserve_rate)
            
        gib_feats_list = torch.cat(gib_feats_list, dim=1)
        gib_prediction_all = gib_feats_list
        preserve_rate_all = torch.stack(preserve_rate_all).mean()
        # gib  

        # 
        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)
        gating_combine = self.compute_gating(bg,node_feats,feats_list,git=False)
        # 
        
        Pri_centered_feats_list = []
        Pri_centered_git_feats_list = []
        for i in range(self.task_num):
            if i == self.prim_index and self.use_primary_centered_gate == True:
                Pri_centered_feats_list.append(gating_combine)
            else:
                Pri_centered_feats_list.append(feats_list[i])

        prediction_all = []
        # Multi-task predictor
        for i in range(self.task_num):
            mol_feats = Pri_centered_feats_list[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            predict = self.output_layer1[i](h2)
            prediction_all.append(predict)
        prediction_all = torch.cat(prediction_all, dim=1)
            
        return prediction_all, gib_prediction_all, KL_loss_all, preserve_rate

    def get_subgraph_list(self, bg, node_feats):
        bg = bg.to(node_feats.device)
        node_feats = self.conv1(bg, node_feats)
        node_feats = self.conv2(bg, node_feats)
        node_feats = self.conv3(bg, node_feats)
        # gib 
        lambda_pos_list = []
        gib_feats_list, KL_loss_all, preserve_rate_all = [], [], []
        for i in range(self.task_num):
            gib_feat, KL_loss, preserve_rate,lambda_pos = self.graph_information_bottleneck_module[i](node_feats, bg)
            lambda_pos_list.append(lambda_pos)
            gib_feats_list.append(gib_feat)
            KL_loss_all.append(KL_loss)
            preserve_rate_all.append(preserve_rate)
            
        return lambda_pos_list
    def gib_layer(self,device,gnn_out_feats):
        return GraphInformationBottleneckModule(device,gnn_out_feats,gnn_out_feats)

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
            )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
            )
    def atom_weight(self, in_feats):
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
            )

    def compute_gating(self, bg, node_feats, feats_list, git=False):
        # gate input
        combine = []
        bg.ndata['h'] = node_feats
        hg = dgl.mean_nodes(bg, 'h')

        prim = feats_list[self.prim_index]
        prim_u = torch.unsqueeze(prim, dim=1)
        
        if self.task_num == 1:
            return prim
        
        for i in range(self.task_num):
            if i == self.prim_index:
                continue

            auxi = feats_list[i]
            auxi_u = torch.unsqueeze(auxi, dim=1)
            # print(f"auxi shape: {auxi.shape}, prim shape: {prim.shape}")
            # print(f"auxi_u shape: {auxi_u.shape}, prim_u shape: {prim_u.shape}")
            
            gating_f = torch.cat((auxi_u, prim_u), dim=1)
            if git == True:
                gate = self.gates_git[i](hg)
            else:
                gate = self.gates[i](hg)
            gate = F.softmax(gate, dim=-1)
            gate = torch.unsqueeze(gate, dim=-1)
            # print(f"hg shape: {hg.shape}")
            # print(f"gating_f shape: {gating_f.shape}, gate shape: {gate.shape}")
            
            gating_r = torch.sum(gating_f * gate, dim=1)
            combine.append(gating_r)    
            
        gating_combine = sum(combine)
        return gating_combine

