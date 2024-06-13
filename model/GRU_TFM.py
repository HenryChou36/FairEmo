#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

class TFM(nn.Module):
    def __init__(self, layers, hidden_size, head):
        super(TFM, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.head = head
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.hidden_size, nhead=self.head, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.layers, norm=nn.LayerNorm(self.hidden_size))
    def forward(self, encoder_outputs, pad_mask=None):
        if pad_mask is None:
            out = self.transformer_encoder.forward(encoder_outputs)
        else:
            out = self.transformer_encoder.forward(encoder_outputs, src_key_padding_mask=pad_mask)
        return out

class Bi_GRU_dynamic(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(Bi_GRU_dynamic, self).__init__()
        self.feat_dim = feat_dim
        self.gru_hidden_dim = hidden_dim
        # self.hidden_layers_num = hidden_layers_num
        # self.max_length = max_length
        
        self.gru_model = nn.GRU(self.feat_dim, self.gru_hidden_dim, 1, bidirectional=True)
    
    def forward(self, input_seqs, input_lens, hidden=None):
        input_seqs = input_seqs.transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lens)
        outputs, hidden = self.gru_model(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)  #[T,B,E]
        outputs = outputs[:, :, :self.gru_hidden_dim] \
                + outputs[:, :, self.gru_hidden_dim:]   
        outputs = outputs.transpose(0, 1)
        hidden = hidden.transpose(0, 1)
        
        return outputs, hidden, input_lens

class GRU_dynamic(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(GRU_dynamic, self).__init__()
        self.feat_dim = feat_dim
        self.gru_hidden_dim = hidden_dim
        
        self.gru_model = nn.GRU(self.feat_dim, self.gru_hidden_dim, 1, bidirectional=False)
    
    def forward(self, input_seqs, input_lens, hidden=None):
        input_lens = input_lens.cpu()
        input_seqs = input_seqs.transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lens)
        outputs, hidden = self.gru_model(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)  #[T,B,E]      
        outputs = outputs.transpose(0, 1)
        hidden = hidden.transpose(0, 1)
        
        return outputs, hidden, input_lens

class GRU_TFM_mlt_reg(nn.Module):
    def __init__(self, feat_dim, hidden_dim, hidden_layers_num, tfm_head, max_length, num_task = 3, dropout_r=0.0):
        super(GRU_TFM_mlt_reg, self).__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers_num = hidden_layers_num
        self.tfm_head = tfm_head
        self.max_length = max_length
        self.num_task = num_task

        # input fc
        network = [nn.Linear(self.feat_dim, self.hidden_dim)]
        self.input_layer = nn.Sequential(*network)
        
        # bi_GRU_encoder
        self.gru = GRU_dynamic(self.hidden_dim, self.hidden_dim)
        
        # tfm_encoder
        self.tfm = TFM(self.hidden_layers_num, self.hidden_dim, self.tfm_head)
        
        # drop_out
        self.drop_out = nn.Dropout(p=dropout_r)
        
        # output fc
        self.regs = torch.nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), 
                                                        nn.Linear(self.hidden_dim, 1)) 
                                                        for _ in range(self.num_task)])
        
    def forward(self, input_seqs, input_lens):
        # input fc (B, S, H)
        batch_size = len(input_seqs)
        input_fc_op = torch.zeros((batch_size, input_lens.max() ,self.hidden_dim)).to(input_seqs.device)
        input_mask = torch.ones((batch_size, input_lens.max()), dtype = torch.bool).to(input_seqs.device)
        
        for i in range(len(input_lens)):
            input_fc_op[i, :input_lens[i], :] = self.input_layer.forward(input_seqs[i,:input_lens[i],:])
            input_mask[i, :input_lens[i]] = False
        
        # gru encode
        gru_op, _, _ = self.gru.forward(input_fc_op, input_lens)
        gru_op = gru_op.transpose(0, 1)
        # tfn_emcoder (S, B, H)
        tfm_output = self.tfm.forward(gru_op, input_mask)
        # # output fc
        tfm_output = tfm_output.transpose(0, 1)
        # (B, S, H)
        tfm_output = self.drop_out(tfm_output)
        reg_output = []
        output_mask = torch.arange(input_lens.max()).unsqueeze(0).expand(batch_size, -1).to(input_seqs.device)
        output_mask = output_mask < input_lens.unsqueeze(dim=1)
        output_mask = output_mask.unsqueeze(2).expand(-1, -1, self.hidden_dim)
        reg_output = torch.zeros(batch_size, self.num_task).to(input_seqs.device)
        for i in range(self.num_task):
            reg_output[:, i] = self.regs[i].forward(torch.sum(tfm_output * output_mask.float(), dim=1, keepdim=False) / 
                                                    input_lens.unsqueeze(dim=1).float()).squeeze()

        return tfm_output, reg_output
