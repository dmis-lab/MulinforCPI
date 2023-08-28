from typing import Dict, List, Union, Callable

import dgl
import torch
import numpy as np
from functools import partial

from torch import nn
import torch.nn.functional as F

from commons.mol_encoder import AtomEncoder, BondEncoder
from models.base_layers import MLP
import seaborn as sns

EPS = 1e-5


def aggregate_mean(h, **kwargs):
    return torch.mean(h, dim=-2)


def aggregate_max(h, **kwargs):
    return torch.max(h, dim=-2)[0]


def aggregate_min(h, **kwargs):
    return torch.min(h, dim=-2)[0]


def aggregate_std(h, **kwargs):
    return torch.sqrt(aggregate_var(h) + EPS)


def aggregate_var(h, **kwargs):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_moment(h, n=3, **kwargs):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=-2, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=-2)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1.0 / n)
    return rooted_h_n


def aggregate_sum(h, **kwargs):
    return torch.sum(h, dim=-2)


# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x N x Din) as output


def scale_identity(h, D=None, avg_d=None):
    return h


def scale_amplification(h, D, avg_d):
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in the training set
    return h * (np.log(D + 1) / avg_d["log"])


def scale_attenuation(h, D, avg_d):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    return h * (avg_d["log"] / np.log(D + 1))


PNA_AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
    "moment3": partial(aggregate_moment, n=3),
    "moment4": partial(aggregate_moment, n=4),
    "moment5": partial(aggregate_moment, n=5),
}

PNA_SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}


class MultiheadAttention(nn.Module):
    ""
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Define query, key, and value linear transformations
        self.q_linear = nn.Linear(input_dim, embed_dim, bias= False)
        self.k_linear = nn.Linear(input_dim, embed_dim, bias= False)
        self.v_linear = nn.Linear(input_dim, embed_dim, bias= False)

        # Define multi-head linear transformation
        self.multihead_linear = nn.Linear(embed_dim, embed_dim, bias= False)

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs_Q,inputs_K,inputs_V, mask=None):
        # Apply query, key, and value linear transformations
        q = self.q_linear(inputs_Q)  # shape: (batch_size, seq_len, embed_dim)
        k = self.k_linear(inputs_K)  # shape: (batch_size, seq_len, embed_dim)
        v = self.v_linear(inputs_V)  # shape: (batch_size, seq_len, embed_dim)

        # Split the embeddings into num_heads pieces
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        k = k.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        v = v.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)

        # Compute the attention scores and apply the mask if provided
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_dim // self.num_heads, dtype=torch.float32))  # shape: (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply the attention scores to the value embeddings and concatenate the results
        attention = torch.softmax(scores, dim=-1)
        x = torch.matmul(self.dropout(attention), v)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # shape: (batch_size, seq_len, embed_dim)

        # Apply the multi-head linear transformation
        x = self.multihead_linear(x)  # shape: (batch_size, seq_len, embed_dim)

        return x

class MultiheadAttentioninter(nn.Module):
    ""
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttentioninter, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Define query, key, and value linear transformations
        self.q_linear = nn.Linear(input_dim, embed_dim, bias= False)
        self.k_linear = nn.Linear(20, 4, bias= False)
        self.v_linear = nn.Linear(20, 4, bias= False)

        # Define multi-head linear transformation
        self.multihead_linear = nn.Linear(embed_dim, embed_dim, bias= False)

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs_Q,inputs_K,inputs_V, mask=None):
        # Apply query, key, and value linear transformations
        q = self.q_linear(inputs_Q)  # shape: (batch_size, seq_len, embed_dim)
        k = self.k_linear(inputs_K)  # shape: (batch_size, seq_len, embed_dim)
        v = self.v_linear(inputs_V)  # shape: (batch_size, seq_len, embed_dim)

        # Split the embeddings into num_heads pieces
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        k = k.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        v = v.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)

        # Compute the attention scores and apply the mask if provided
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_dim // self.num_heads, dtype=torch.float32))  # shape: (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply the attention scores to the value embeddings and concatenate the results
        attention = torch.softmax(scores, dim=-1)
        x = torch.matmul(self.dropout(attention), v)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # shape: (batch_size, seq_len, embed_dim)

        # Apply the multi-head linear transformation
        x = self.multihead_linear(x)  # shape: (batch_size, seq_len, embed_dim)

        return x
class PNAcpi(nn.Module):
    """
    3Dinforcpi structure:
    capable of generating 3D geometric information from a graph, we proceed to the fine-tune phase.
    During this phase, PNA is utilized to extract information from compounds, in combination with Morgan fingerprints information.
    While the protein information is extracted from output of ESM-fold using multiple CNNs.
    """

    def __init__(self,
                 hidden_dim,
                 target_dim,
                 aggregators: List[str],
                 scalers: List[str],
                 readout_aggregators: List[str],
                 params_1d : List[str],
                 params_res_1d : List[str],
                 params_2d: List[str],
                 readout_batchnorm: bool = True,
                 readout_hidden_dim=None,
                 readout_layers: int = 2,
                 residual: bool = True,
                 pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "elu",
                 last_activation: Union[Callable, str] = "none",
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False,
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 batch_norm_momentum = 0.1,
                 last_activation_readout: Union[Callable, str] = "relu",
                 **kwargs):
        super(PNAcpi, self).__init__()
        self.node_gnn = PNAGNN(hidden_dim=hidden_dim, aggregators=aggregators,
                               scalers=scalers, residual=residual, pairwise_distances=pairwise_distances,
                               activation=activation, last_activation=last_activation, mid_batch_norm=mid_batch_norm,
                               last_batch_norm=last_batch_norm, propagation_depth=propagation_depth, dropout=dropout,
                               posttrans_layers=posttrans_layers, pretrans_layers=pretrans_layers,
                               batch_norm_momentum=batch_norm_momentum
                               )
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.protein_net1 = nn.Sequential(*[nn.Linear(params_res_1d[0],20), nn.ReLU(), nn.Dropout(dropout)])
        self.dist_net = nn.Sequential(         
            nn.Conv2d(
                in_channels = params_2d[0],              
                out_channels= params_2d[1],            
                kernel_size = params_2d[2],              
                stride = params_2d[2],                   
                padding = params_2d[3],                  
            ),
            nn.BatchNorm2d(params_2d[1]),                            
            nn.ReLU(),
            nn.Conv2d(
                in_channels=params_2d[1],              
                out_channels=1,            
                kernel_size=params_2d[2],              
                stride=params_2d[2],                   
                padding=params_2d[3],                  
            ),
            nn.BatchNorm2d(1),                             
            nn.ReLU()
        )
        self.embedding_xt = nn.Embedding(2, 4)
        self.conv_in = nn.Sequential(*[nn.Conv1d(in_channels= params_1d[0], out_channels=params_1d[1], kernel_size=1),
            nn.BatchNorm1d(params_1d[1]),nn.ReLU()])
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=params_res_1d[0],
                                    out_channels=params_res_1d[0]*2,
                                    kernel_size=params_res_1d[1],
                                    padding= params_res_1d[1]//2) for _ in range(params_res_1d[2])]) 
        self.norm = nn.LayerNorm(params_1d[1])

        self.multihead_att_comp = MultiheadAttention(input_dim = 4, embed_dim = 4, num_heads = 1)
        self.multihead_att = MultiheadAttention(input_dim = 20, embed_dim = 20, num_heads = 1)

        self.out_prot_dist = nn.Sequential(*[nn.Linear(100,32), nn.ReLU(), nn.Dropout(dropout )])
        self.output_proteinnet = MLP(in_dim=400, hidden_size=readout_hidden_dim,
                                  mid_batch_norm=readout_batchnorm, out_dim=128,
                                  layers=readout_layers, batch_norm_momentum=batch_norm_momentum, last_activation = last_activation_readout)
        self.output_f = MLP(in_dim=256, hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum, last_activation = last_activation_readout)
        self.output_comp = MLP(in_dim=2048*4, hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=128,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum, last_activation = last_activation_readout)
    
    def fourier_encode_dist(self, x, num_encodings=1, include_self=True):
        # reimplementation of high-frequency variation transformation
        x1 = x.unsqueeze(-1)
        device, dtype, orig_x = x1.device, x1.dtype, x1
        scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
        x2 = x1 / scales
        dummy_tensor = torch.zeros((x2.size(0),x2.size(1),x2.size(2), num_encodings*2), device=device, dtype=dtype)
        dummy_tensor[:, :, :,0::2] = x2.sin()
        dummy_tensor[:, :, :,1::2] = x2.cos()
        processed_out = torch.cat((orig_x, dummy_tensor), dim=-1) if include_self else dummy_tensor
        
        return processed_out

    def normalization(self,vector_present,threshold=0.1):
        '''
        The goal of normalization is to bring all the features to a similar scale, 
        which can improve the performance and convergence of certain machine learning algorithms. 
        One common method of normalization is scaling features to have a minimum value of 0 and a maximum value of 1, also known as min-max normalization.
        '''
        vector_present_clone = vector_present.clone()
        num = vector_present_clone - vector_present_clone.min(1,keepdim = True)[0]
        de = vector_present_clone.max(1,keepdim = True)[0] - vector_present_clone.min(1,keepdim = True)[0]

        return num / de

    def forward(self, graph: dgl.DGLGraph, protein_feature: torch.Tensor, protein_dist_matrix: torch.Tensor, morgan_fingers: torch.Tensor):
        # PNA model for extracting graph information
        self.node_gnn(graph)
        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        graph_feat = readout.view(readout.size(0),200,4)#batch 200 4

        # Learnable lookup table for extracting MFs information 
        morgan_out = self.embedding_xt(morgan_fingers.cuda()) # batch 2048 4
        
        # Final compound representation
        comp_re = self.multihead_att_comp(morgan_out,graph_feat,graph_feat) # batch 2048 4
        
        # put the Distance map to to higher dimensional by Fourier feature mapping function
        dist_transform = self.fourier_encode_dist(protein_dist_matrix).transpose(1,3)
        
        # 2DCNN for extracting distance information of protein
        protein_dist = self.dist_net(dist_transform)
        protein_dist =  protein_dist.squeeze(1) # batch 20 20

        # 1DCNN for extracting atomic level information of protein
        input_nn = self.conv_in(protein_feature) # batch 500 65 

        conv_input = input_nn.permute(0, 2, 1) # batch 65 500
        for i, conv in enumerate(self.convs):
            conved = self.norm(conv(conv_input))
            conved = F.glu(conved, dim=1)
            conv_input = conved + conv_input

        out_put = conv_input.permute(0, 2, 1) # batch 500 65
        prot_batch_emb_1hot = self.protein_net1(out_put) # batch 500 20

        # Final protein representation
        prot_batch_att = self.multihead_att(protein_dist,prot_batch_emb_1hot,prot_batch_emb_1hot) # batch 20 20

        # Faltten the features of C and P
        prot_batch_emb_out = self.output_proteinnet(torch.flatten(prot_batch_att, start_dim = 1)) # batch 400
        comp_batch_emb_out = self.output_comp(torch.flatten(comp_re, start_dim = 1))

        interaction = torch.cat([torch.nan_to_num(self.normalization(comp_batch_emb_out)),
            torch.nan_to_num(self.normalization(prot_batch_emb_out))],dim =-1)

        if True in torch.isnan(interaction):
            import pdb
            pdb.set_trace()

        return self.output_f(interaction)


class PNAGNN(nn.Module):
    def __init__(self, hidden_dim, aggregators: List[str], scalers: List[str],
                 residual: bool = True, pairwise_distances: bool = False, activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none", mid_batch_norm: bool = False,
                 last_batch_norm: bool = False, batch_norm_momentum=0.1, propagation_depth: int = 5,
                 dropout: float = 0.0, posttrans_layers: int = 1, pretrans_layers: int = 1, **kwargs):
        super(PNAGNN, self).__init__()

        self.mp_layers = nn.ModuleList()

        for _ in range(propagation_depth):
            self.mp_layers.append(
                PNALayer(in_dim=hidden_dim, out_dim=int(hidden_dim), in_dim_edges=hidden_dim, aggregators=aggregators,
                         scalers=scalers, pairwise_distances=pairwise_distances, residual=residual, dropout=dropout,
                         activation=activation, last_activation=last_activation, mid_batch_norm=mid_batch_norm,
                         last_batch_norm=last_batch_norm, avg_d={"log": 1.0}, posttrans_layers=posttrans_layers,
                         pretrans_layers=pretrans_layers, batch_norm_momentum=batch_norm_momentum
                         ),

            )
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(self, graph: dgl.DGLGraph):
        graph.ndata['feat'] = self.atom_encoder(graph.ndata['feat'])
        graph.edata['feat'] = self.bond_encoder(graph.edata['feat'])

        for mp_layer in self.mp_layers:
            mp_layer(graph)


class PNALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, in_dim_edges: int, aggregators: List[str], scalers: List[str],
                 activation: Union[Callable, str] = "relu", last_activation: Union[Callable, str] = "none",
                 dropout: float = 0.0, residual: bool = True, pairwise_distances: bool = False,
                 mid_batch_norm: bool = False, last_batch_norm: bool = False, batch_norm_momentum=0.1,
                 avg_d: Dict[str, float] = {"log": 1.0}, posttrans_layers: int = 2, pretrans_layers: int = 1, ):
        super(PNALayer, self).__init__()
        self.aggregators = [PNA_AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [PNA_SCALERS[scale] for scale in scalers]
        self.edge_features = in_dim_edges > 0
        self.activation = activation
        self.avg_d = avg_d
        self.pairwise_distances = pairwise_distances
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        self.pretrans = MLP(
            in_dim=(2 * in_dim + in_dim_edges + 1) if self.pairwise_distances else (2 * in_dim + in_dim_edges),
            hidden_size=in_dim, out_dim=in_dim, mid_batch_norm=mid_batch_norm, last_batch_norm=last_batch_norm,
            layers=pretrans_layers, mid_activation=activation, dropout=dropout, last_activation=last_activation,
            batch_norm_momentum=batch_norm_momentum

        )
        self.posttrans = MLP(in_dim=(len(self.aggregators) * len(self.scalers) + 1) * in_dim, hidden_size=out_dim,
                             out_dim=out_dim, layers=posttrans_layers, mid_activation=activation,
                             last_activation=last_activation, dropout=dropout, mid_batch_norm=mid_batch_norm,
                             last_batch_norm=last_batch_norm, batch_norm_momentum=batch_norm_momentum
                             )

    def forward(self, g):
        h = g.ndata['feat']
        h_in = h
        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['feat']], dim=-1)
        # post-transformation
        h = self.posttrans(h)
        if self.residual:
            h = h + h_in

        g.ndata['feat'] = h

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        The message function to generate messages along the edges.
        """
        return {"e": edges.data["e"]}

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        r"""
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.
        """
        h_in = nodes.data['feat']
        h = nodes.mailbox["e"]
        D = h.shape[-2]
        h_to_cat = [aggr(h=h, h_in=h_in) for aggr in self.aggregators]
        h = torch.cat(h_to_cat, dim=-1)

        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=-1)

        return {'feat': h}

    def pretrans_edges(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        Return a mapping to the concatenation of the features from
        the source node, the destination node, and the edge between them (if applicable).
        """
        if self.edge_features and self.pairwise_distances:
            squared_distance = torch.sum((edges.src['x'] - edges.dst['x']) ** 2, dim=-1)[:, None]
            z2 = torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], squared_distance], dim=-1)
        elif not self.edge_features and self.pairwise_distances:
            squared_distance = torch.sum((edges.src['x'] - edges.dst['x']) ** 2, dim=-1)[:, None]
            z2 = torch.cat([edges.src['feat'], edges.dst['feat'], squared_distance], dim=-1)
        elif self.edge_features and not self.pairwise_distances:
            z2 = torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=-1)
        else:
            z2 = torch.cat([edges.src['feat'], edges.dst['feat']], dim=-1)
        return {"e": self.pretrans(z2)}