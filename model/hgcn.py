"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import model.manifolds as manifolds
import model.layers.hyp_layers as hyp_layers

class HGCN(nn.Module):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, args):
        super(HGCN, self).__init__()
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, args.manifold)()
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)

    def forward(self, x, adj):

        batch_size, max_len, feat_dim = x.shape

        x = x.unsqueeze(dim=2).repeat(1,1,max_len,1)
        x = (x.float() * x.float().transpose(1, 2))

        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        output = (x_hyp, adj)
        for i, hgcn_layer_module in enumerate(self.layers):
            output = hgcn_layer_module(output)

        output, _ = output
        output = self.manifold.proj_tan0(self.manifold.logmap0(output, c=self.c), c=self.c)

        return output

