import torch
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn as nn
from torch_geometric.nn import GINConv, SAGEConv, GCNConv


class MLP_classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, args.num_classes)

    def clip_parameters(self):
        for p in self.lin.parameters():
            p.data.clamp_(-self.args.clip_c, self.args.clip_c)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None):
        h = self.lin(h)

        return h


class MLP_encoder(torch.nn.Module):
    def __init__(self, args):
        super(MLP_encoder, self).__init__()
        self.args = args

        self.lin = Linear(args.num_features, args.hidden)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index=None, mask_node=None):
        h = self.lin(x)

        return h   
    

class GCN_encoder(nn.Module):
    def __init__(self, args):
        super(GCN_encoder, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2 = GCNConv(args.hidden, args.hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.transition(x)
        h = self.conv2(x, edge_index, edge_weight)
        return h
    
# class GCN_encoder(nn.Module):
#     def __init__(self, args):
#         super(GCN_encoder, self).__init__()
#         self.conv = GCNConv(args.num_features, args.hidden)

#     def reset_parameters(self):
#         self.conv.reset_parameters()

#     def forward(self, x, edge_index, edge_weight=None):
#         h = self.conv(x, edge_index, edge_weight)
#         return h


class GIN_encoder(nn.Module):
    def __init__(self, args):
        super(GIN_encoder, self).__init__()

        self.args = args

        self.mlp = nn.Sequential(
            nn.Linear(args.num_features, args.hidden),
            # nn.ReLU(),
            # nn.BatchNorm1d(args.hidden),
            # nn.Linear(args.hidden, args.hidden),
        )

        self.conv = GINConv(self.mlp)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, adj_norm_sp=None):
        h = self.conv(x, edge_index)
        return h


class SAGE_encoder(nn.Module):
    def __init__(self, args):
        super(SAGE_encoder, self).__init__()

        self.args = args

        self.conv1 = SAGEConv(args.num_features, args.hidden, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2 = SAGEConv(args.hidden, args.hidden, normalize=True)
        self.conv2.aggr = 'mean'

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.transition(x)
        h = self.conv2(x, edge_index, edge_weight)
        return h
