import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from torch.cuda.amp import autocast
from model.hgcn import HGCN
from transformers import AutoModel
import model.manifolds as manifolds

from model.self_optimiza import nt_xent

import random
from model.gcn import GraphConvolution

class our_model(nn.Module):
    def __init__(self, config):
        # super(gcn, self).__init__(config)
        # self.bert = BertModel(config)
        super().__init__()
        self.config = config
        self.c = config.c
        self.encoder = AutoModel.from_pretrained(config.model_name_or_path, config=config)
        # self.p_hgcn = HGCN(config)
        # self.n_hgcn = HGCN(config)
        self.hgcn = HGCN(config)

        self.num_pos_embedding = nn.Embedding(config.num_pos, config.hidden_size, padding_idx=0)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_fnt = nn.CrossEntropyLoss()
        self.manifold = getattr(manifolds, config.manifold)()
        self.decoder_line = nn.Linear(config.hidden_size, config.hidden_size, config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        self.fc2 = torch.nn.Linear(config.num_labels, config.hidden_size)
        self.NT_Xent = nt_xent.NT_Xent(self.config.train_batch_size, 0.5, 1)
        self.p = config.p

    def max_pooling(self, sequence, e_mask):
        entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2)
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def extract_entity(self, sequence, e_mask):
        return self.max_pooling(sequence, e_mask)

    def get_logits(self, hidden, e1_mask, e2_mask, labels=None):

        hidden = torch.sum(hidden, 1)
        e1_h = self.extract_entity(hidden, e1_mask)
        e2_h = self.extract_entity(hidden, e2_mask)

        pooled_output = torch.cat([e1_h, e2_h], dim=-1)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss, pooled_output
        else:
            return logits, pooled_output

    @autocast()
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e1_mask=None, e2_mask=None,
                dep_adj_matrix=None, dep_type_matrix=None, valid_ids=None, stanford_pos=None):

        sequence_output = self.encoder(input_ids, attention_mask=attention_mask)[0]

        hgcn_output = self.dropout(sequence_output)

        # p_hgcn_output = self.p_hgcn(hgcn_output, dep_adj_matrix)
        p_hgcn_output = self.hgcn(hgcn_output, dep_adj_matrix)
        p_hgcn_logits, p_hgcn_entity = self.get_logits(p_hgcn_output, e1_mask, e2_mask, labels)

        if labels is not None:
            # n_hgcn_output = self.n_hgcn(hgcn_output, dep_type_matrix)
            n_hgcn_output = self.hgcn(hgcn_output, dep_type_matrix)
            n_hgcn_logits, n_hgcn_entity = self.get_logits(n_hgcn_output, e1_mask, e2_mask, labels)

            l1 = self.NT_Xent(p_hgcn_entity, n_hgcn_entity)
            l2 = self.NT_Xent(n_hgcn_entity, p_hgcn_entity)

            ret = (l1 + l2) * 0.5
            ret = ret.mean()

            return (self.p*p_hgcn_logits + (1-self.p)*n_hgcn_logits) + 0.5*ret
        else:
            return p_hgcn_logits
