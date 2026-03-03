# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GraphConvolution(nn.Module):
    """
    GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, text, adj):
        batch_size, max_len, feat_dim = text.shape
        text = text.unsqueeze(dim=2)
        text = text.repeat(1, 1, max_len, 1)
        adj = adj.unsqueeze(-1).repeat(1, 1, 1, feat_dim)
        hidden = torch.matmul(text.float(), self.weight.float())

        output = hidden * adj.float()
        output = torch.sum(output, dim=1)

        if self.bias is not None:
            output = output + self.bias

        return F.relu(output.type_as(text))

