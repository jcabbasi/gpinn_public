

from cmath import tanh
import torch
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
import os
import numpy as np
import time
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from MGN_phys import *
from MGN_utils import *  
from copy import deepcopy
from time import gmtime, strftime, localtime
from gatv2_conv_PyG import *




class MeshGraphNet(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge,geometric_dim, hidden_dim, output_dim, args, index_list, emb=False):
        super(MeshGraphNet, self).__init__()


        self.file_version=' 7.0.0'
        self.index_list=index_list
        self.model_log='\n ##  Log : Training History ## \n\n\n'
        self.model_log += '** Model is initialized at: '+ strftime("%a, %d %b %Y %H:%M:%S", gmtime())
        self.args=args
        self.train_stat=dict( epoch=[], loss=[], loss_test=[], best_test_loss=[])
        self.model_name=args.model_name        
        self.num_layers = args.num_layers
        self.model_char_name=char_logger(args)     
        self.model_log  += self.model_char_name  + '\n\n'
        
        self.model_stat=dict(   epoch=[], loss=[], loss_data_train=[],  loss_phys_train=[],loss_dPINN_train=[], loss_test=[],
                                best_test_loss=[], epoch_time=[], called_for_training=0, minimum_loss_achieved=9999, epoch_cum=0)        
                
        # # INJECTING PHYSICS
        self.phys=GraphPhysics(model_char=self.model_char_name)
                
                
                
        self.node_encoder = node_encoder_builder( input_dim_node + geometric_dim , args )
        self.edge_encoder = edge_encoder_builder( input_dim_edge                 , args )
        self.pos_encoder  = pos_encoder_builder(geometric_dim,args)

        
        
        self.processor = nn.ModuleList()
        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(args))



        self.decoder = node_decoder_builder(output_dim,args)


    def build_processor_model(self):
        return ProcessorLayer

    def  forward_euler(self,data,time=None,x_dir=None,y_dir=None):
        
        
        il=data.index_list
        dx=self.forward(data,time,x_dir,y_dir)
        x= dx * data.x[ : , [ il['n_dt'] , il['n_dt'] ] ] + data.x[ : , [ il['n_p'] , il['n_sw'] ] ]
        
        if self.args.physical_correction:
            x = self.phys.physical_correction(x,data.index_list,data.s_ir)
                   

        return x , dx
        
        
    def forward(self,data_inp,time=None,x_dir=None,y_dir=None):
        
        # # # # # # # # # #     PRE-PROCESSING    # # # # # # # # # # 
        
        # data_inp.x=torch.tensor(data_inp.x.detach())
        # data_inp.y=torch.tensor(data_inp.y.detach())
        # data_inp.edge_attr=torch.tensor(data_inp.edge_attr.detach())
        # data_inp.pos=torch.tensor(data_inp.pos.detach())
        
        data=deepcopy(data_inp)
        x_chng=data.x.clone()
        pos_chng=data.pos.clone()
        
        if time is not None:
            x_chng[:, data.index_list['n_time']]=time*1
        if x_dir is not None:
            pos_chng[:, data.index_list['ng_x']]=x_dir*1  
        if y_dir is not None:       
            pos_chng[:, data.index_list['ng_y']]=y_dir*1 
        data.x=x_chng
        data.pos=pos_chng
    

        # # # # # # # # # #     N O R M A L I Z A T I O N    # # # # # # # # # # 
        if self.args.live_normalization:
            data=self.scaler.normalize(data,attr='inputs')


        # # # # # # # # # #     E N C O D I N G    # # # # # # # # # # 
        x, edge_index, edge_attr, pos = data.x, data.edge_index, data.edge_attr, data.pos
        pos = self.pos_encoder(pos) 
        x = torch.cat( [ x , pos ] , dim = 1 )
        
        x = self.node_encoder(x) 
        
        edge_attr = self.edge_encoder(edge_attr.float()) 


        # # # # # # # # # #     P R O C E S S I N G    # # # # # # # # # # 
        for i in range(self.num_layers):
            x , edge_attr = self.processor[i](x,edge_index,edge_attr)

        # # # # # # # # # #     POST-PROCESSING    # # # # # # # # # # 
        x = self.decoder(x) ; 
        
        x =  self.scaler.denormalize_vec(x,attr='targ',cols=['targ_dpdt','targ_dswdt'])


        
        return x



    def loss(self, y_pred_abs,y_delta_pred, data,time,x_dir,y_dir,physics_included=False, pinn_case=True):

        # # # # # # # # # #     PRE-PROCESSING    # # # # # # # # # # 
        w_p=self.args.loss_parameter_weight[0]
        w_s=self.args.loss_parameter_weight[1]
        if self.args.partial_differential_loss:
            pd_ids=[data[0].index_list['targ_dpdt'],data[0].index_list['targ_dswdt']]
            y_pred=y_delta_pred
            y_true =  data.y[:,pd_ids]  
            
        else:
            pd_ids=[data.index_list['targ_p'],data.index_list['targ_sw']]
            y_pred=y_pred_abs
            y_true =  data.y[:,pd_ids]   
            
         
         
         
        # # # # # # # # # #     PHYSICAL LOSS FUNCTION    # # # # # # # # # #                    
        loss_phys=torch.tensor(0.0)
        loss_dPINN=torch.tensor(0.0)        
        if physics_included and pinn_case:
            loss_phys, loss_dPINN=self.phys.phys_loss(y_pred_abs,data,time,x_dir,y_dir,data.index_list,data.n_relperm,data.s_ir,data.krmax, self.args)
               
 
 
         # # # # # # # # # #     DATA-BASED LOSS FUNCTION    # # # # # # # # # #                    
               
        if self.args.normalized_evaluation_loss:
            y_pred=  self.scaler.normalize_vec(y_pred+0,attr='targ')
            y_true=  self.scaler.normalize_vec(y_true+0,attr='targ')
             
        error_p=torch.sum((y_true[:,0]-y_pred[:,0])**2,axis=0)
        error_sw=torch.sum((y_true[:,1]-y_pred[:,1])**2,axis=0)
        loss_p=torch.sqrt(torch.mean(error_p)) / 1e6 # rescaling values by dividing to 10bars
        loss_sw=torch.sqrt(torch.mean(error_sw))
        loss_data= (loss_p * w_p + loss_sw * w_s) * 2 / (w_p + w_s)   * 1e7  # multiplied by 1e7 to rescale the values close to 1
        
        
         # # # # # # # # # #     GATHERING LOSSESS    # # # # # # # # # #                    
        loss=  ( self.args.loss_weight[0] * loss_data   +   self.args.loss_weight[1] * loss_phys +  self.args.loss_weight[2] * loss_dPINN) / sum(self.args.loss_weight)
        
        return loss, loss_data, loss_phys, loss_dPINN



class ProcessorLayer(MessagePassing):
    def __init__(self,args,  **kwargs):
        super(ProcessorLayer, self).__init__(  **kwargs )
        node_hidden_dims = args.hidden_dim
        edge_hidden_dims = args.hidden_dim_edges
        self.args=args
        edge_aggrgate_hidden_dim=2*node_hidden_dims+edge_hidden_dims
        node_aggrgate_hidden_dim=1*node_hidden_dims+edge_hidden_dims
        
        if self.args.encode_edges:
            self.edge_mlp = Sequential(Linear( edge_aggrgate_hidden_dim , edge_hidden_dims),
                                    ReLU(),
                                    Linear( edge_hidden_dims, edge_hidden_dims),
                                    LayerNorm(edge_hidden_dims))

        self.node_mlp = Sequential(Linear( node_aggrgate_hidden_dim , node_hidden_dims),
                                   ReLU(),
                                   Linear( node_hidden_dims, node_hidden_dims),
                                   nn.Tanh(),                                   
                                   LayerNorm(node_hidden_dims))

        if self.args.node_strategy=='mlp':
            self.node_processor=   Sequential(Linear( node_aggrgate_hidden_dim , node_hidden_dims),
                                   ReLU(),
                                   Linear( node_hidden_dims, node_hidden_dims),
                                   nn.Tanh(),                                   
                                   LayerNorm(node_hidden_dims))
        
        elif self.args.node_strategy=='gatv2':
            self.node_processor=GATv2Conv(node_aggrgate_hidden_dim,node_hidden_dims,args.attention_heads,concat=False)



        self.reset_parameters()

    def reset_parameters(self):
        if self.args.encode_edges:
            self.edge_mlp[0].reset_parameters()
            self.edge_mlp[2].reset_parameters()
        if self.args.node_strategy=='mlp':   
            self.node_processor[0].reset_parameters()
            self.node_processor[2].reset_parameters()




    def forward(self, x, edge_index, edge_attr, size = None):

        if self.args.fourier_transform:
            x=torch.fft.fft(x).real
            # x=torch.fft.fft(x)

        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = size) 
        updated_nodes = torch.cat([x,out] , dim=1)           

        if self.args.node_strategy=='mlp':
            updated_nodes = x + self.node_processor(updated_nodes)   
        
        elif self.args.node_strategy=='gatv2':
            updated_nodes = x + self.node_processor(updated_nodes, edge_index)   

                
        if self.args.fourier_transform:
            updated_nodes=torch.fft.ifft(updated_nodes).real
        
        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        updated_edges=edge_attr
        if self.args.encode_edges:        
            updated_edges=torch.cat([x_i, x_j, edge_attr], dim = 1) 
            updated_edges=self.edge_mlp(updated_edges)+edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size = None):

        node_dim = 0
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'sum')

        return out, updated_edges





class node_encoder_builder(torch.nn.Module):
    
    def __init__(self,input_dim_node,args):
        super().__init__()
        hidden_dim=args.hidden_dim        
        self.fc1 = nn.Linear(input_dim_node, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.tan_1= nn.Tanh()
        self.fc3 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        # x= self.tan_1(x)
        x = self.fc3(x)
        return x


class pos_encoder_builder(torch.nn.Module):
    
    def __init__(self,input_dim_node,args):
        super().__init__()
        hidden_dim=args.hidden_dim        
        self.fc1 = nn.Linear(input_dim_node, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim_node)
        self.fc3 = nn.LayerNorm(input_dim_node)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x



class edge_encoder_builder(torch.nn.Module):
    
    def __init__(self,input_dim_edge,args):
        super().__init__()
        hidden_dim_edges=args.hidden_dim_edges
        
        self.fc1 = nn.Linear(input_dim_edge, hidden_dim_edges)
        self.fc2 = nn.Linear(hidden_dim_edges, hidden_dim_edges)
        self.end = nn.LayerNorm(hidden_dim_edges)
        self.tan_1= nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        x= self.tan_1(x)        
        x = self.end(x)
        return x



class node_decoder_builder(torch.nn.Module):
    
    def __init__(self,output_dim, args):
        super().__init__()
        hidden_dim=args.hidden_dim        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

