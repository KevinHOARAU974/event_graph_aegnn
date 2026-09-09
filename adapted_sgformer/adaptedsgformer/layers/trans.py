import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from torch import Tensor

# Adapted self Attention layer of SGFormer
class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, input : Tensor, batch : Tensor, output_attn=False):
        # feature transformation

        #B : Batch size
        #Nmax : number of nodes of the largest graph in the batch
        #H : Number of Heads
        #I : Input size
        #M : Output size

        # Groupe by graph in order to have global attention by graph in batch
        x, mask_dense = to_dense_batch(input, batch) #[B, Nmax, I]
        
        # batch_size = len(batch.unique())
        batch_size = x.size(0)

        qs = self.Wq(x).reshape(batch_size, -1, self.num_heads, self.out_channels) 
        ks = self.Wk(x).reshape(batch_size, -1, self.num_heads, self.out_channels)

        if self.use_weight:
            vs = self.Wv(x).reshape(batch_size, -1, self.num_heads, self.out_channels)
        else:
            vs = x.reshape(batch_size, -1, 1, self.out_channels)

        #Set to zeros padding elements in order that they not contribute to attention
        qs[~mask_dense] = 0.0 
        ks[~mask_dense] = 0.0
        vs[~mask_dense] = 0.0

        # normalize input
        # qs = qs / torch.norm(qs, p=2)  # [B, Nmax, H, M]
        # ks = ks / torch.norm(ks, p=2)  # [B, Nmax, H, M]

        qs = F.normalize(qs, p=2, dim=-1, eps=1e-6)
        ks = F.normalize(ks, p=2, dim=-1, eps=1e-6)

        N = mask_dense.sum(dim=1) # [B] Number of nodes in each graph 

        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)  # [B, Nmax, H, D]
        attention_num += N.view(batch_size, 1, 1, 1) * vs

        # denominator
        all_ones = torch.ones([ks.shape[1]]).to(ks.device)
        ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [B, Nmax, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [B, Nmax, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N.view(batch_size, 1, 1, 1)
        attn_output = attention_num / attention_normalizer  # [B,Nmax, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("bnhm,blhm->bnlh", qs, ks).mean(dim=-1)  # [Nmax, Nmax]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [Nmax,1]
            attention = attention / normalizer
        
        attn_output = attn_output[mask_dense]

        final_output = attn_output.mean(dim=1)

        if output_attn:
            return final_output, attention
        else:
            return final_output



class TransLayerMultiHead(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()

        assert out_channels % num_heads == 0
        self.head_dim = out_channels // num_heads


        self.Wk = nn.Linear(in_channels, out_channels)
        self.Wq = nn.Linear(in_channels, out_channels)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

        self.Wo = nn.Linear(out_channels, out_channels)

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, batch: Batch, output_attn=False):
        # feature transformation

        #B : Batch size
        #Nmax : number of nodes of the largest graph in the batch
        #H : Number of Heads
        #I : Input size
        #M : Output size

        # Groupe by graph in order to have global attention by graph in batch
        x, mask_dense = to_dense_batch(batch.x, batch.batch) #[B, Nmax, I]
        
        # batch_size = len(batch.unique())
        batch_size = x.size(0)

        qs = self.Wq(x).reshape(batch_size, -1, self.num_heads, self.head_dim) 
        ks = self.Wk(x).reshape(batch_size, -1, self.num_heads, self.head_dim)

        if self.use_weight:
            vs = self.Wv(x).reshape(batch_size, -1, self.num_heads, self.head_dim)
        else:
            vs = x.reshape(batch_size, -1, 1, self.out_channels)

        #Set to zeros padding elements in order that they not contribute to attention
        qs[~mask_dense] = 0.0 
        ks[~mask_dense] = 0.0
        vs[~mask_dense] = 0.0

        # normalize input
        # qs = qs / torch.norm(qs, p=2)  # [B, Nmax, H, M/H]
        # ks = ks / torch.norm(ks, p=2)  # [B, Nmax, H, M/H]

        qs = F.normalize(qs, p=2, dim=-1, eps=1e-6)
        ks = F.normalize(ks, p=2, dim=-1, eps=1e-6)

        N = mask_dense.sum(dim=1) # [B] Number of nodes in each graph 

        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)  # [B, Nmax, H, D]
        attention_num += N.view(batch_size, 1, 1, 1) * vs

        # denominator
        all_ones = torch.ones([ks.shape[1]]).to(ks.device)
        ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [B, Nmax, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [B, Nmax, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N.view(batch_size, 1, 1, 1)
        attn_output = attention_num / attention_normalizer  # [B,Nmax, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("bnhm,blhm->bnlh", qs, ks).mean(dim=-1)  # [Nmax, Nmax]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [Nmax,1]
            attention = attention / normalizer
        
        attn_output = attn_output[mask_dense]

        attn_output = attn_output.reshape(attn_output.size(0), self.out_channels)
        final_output = self.Wo(attn_output)
            

        if output_attn:
            return final_output, attention
        else:
            return final_output


class SoftmaxTrans(nn.Module):

    def __init__(self, in_channels,
                     out_channels,
                     num_heads):
        
        super().__init__()

        assert out_channels % num_heads == 0
        self.head_dim = out_channels // num_heads

        self.Wk = nn.Linear(in_channels, out_channels)
        self.Wq = nn.Linear(in_channels, out_channels)
        self.Wv = nn.Linear(in_channels, out_channels)

        self.out_channels = out_channels
        self.num_heads = num_heads

        self.scale = self.head_dim ** -0.5

        self.Wo = nn.Linear(out_channels, out_channels)

    def forward(self, batch : Batch):

        #B : Batch size
        #Nmax : number of nodes of the largest graph in the batch
        #H : Number of Heads
        #I : Input size
        #M : Output size

        # Groupe by graph in order to have global attention by graph in batch
        x, mask_dense = to_dense_batch(batch.x, batch.batch) #[B, Nmax, I]
        
        # batch_size = len(batch.unique())
        batch_size = x.size(0)

        qs = self.Wq(x).reshape(batch_size, -1, self.num_heads, self.head_dim) 
        ks = self.Wk(x).reshape(batch_size, -1, self.num_heads, self.head_dim)

        vs = self.Wv(x).reshape(batch_size, -1, self.num_heads, self.head_dim)

        #QK^T
        attn = torch.einsum("bnhm, blhm -> bhnl", qs, ks)
        attn *= self.scale

        ##Set padding values to -inf before softmax
        attn = attn.masked_fill(~mask_dense[:, None, None, :],float("-inf"))

        attn = F.softmax(attn, dim=-1)

        #AV

        out = torch.einsum("bhnj, bjhd -> bnhd", attn, vs)
        out = out[mask_dense]
        out = out.reshape(out.size(0), self.out_channels)

        out = self.Wo(out)

        return out

class BiasSoftmaxTrans(nn.Module):

    def __init__(self, in_channels,
                     out_channels,
                     num_heads):
        
        super().__init__()

        assert out_channels % num_heads == 0
        self.head_dim = out_channels // num_heads
        self.num_heads = num_heads
        self.out_channels = out_channels

        self.Wk = nn.Linear(in_channels, out_channels)
        self.Wq = nn.Linear(in_channels, out_channels)
        self.Wv = nn.Linear(in_channels, out_channels)

        self.Wa = nn.Linear(3, self.num_heads) #Multiplicative bias
        self.Wb = nn.Linear(3, self.num_heads) #Additive attention bias
        self.Wc = nn.Linear(3, self.out_channels) #Additive values bias

        self.pa = nn.Parameter(torch.tensor([1.0, 1.0, 1.0])) #Learnable parameter for far events
        self.pb = nn.Parameter(torch.tensor([1.0, 1.0, 1.0])) #Learnable parameter for far events
        self.pc = nn.Parameter(torch.tensor([1.0, 1.0, 1.0])) #Learnable parameter for far events

        self.scale = self.head_dim ** -0.5

        self.Wo = nn.Linear(out_channels, out_channels)

    def forward(self, batch: Batch):

        #B : Batch size
        #Nmax : number of nodes of the largest graph in the batch
        #H : Number of Heads
        #I : Input size
        #M : Output size

        # Groupe by graph in order to have global attention by graph in batch
        x, mask_dense = to_dense_batch(batch.x, batch.batch) #[B, Nmax, I]
        
        # batch_size = len(batch.unique())
        batch_size, Nmax, _ = x.size()

        qs = self.Wq(x).reshape(batch_size, -1, self.num_heads, self.head_dim) 
        ks = self.Wk(x).reshape(batch_size, -1, self.num_heads, self.head_dim)
        vs = self.Wv(x).reshape(batch_size, -1, self.num_heads, self.head_dim)

        #QK^T
        attn = torch.einsum("bnhm, blhm -> bhnl", qs, ks)

        #Compute attention bias with edge
        src = batch.edge_index[0,:]
        dst = batch.edge_index[1,:]

        batch_edge = batch.batch[src]

        cum_sum = torch.cumsum(batch.batch.unique(return_counts=True)[1], dim=0)

        offset = torch.cat([torch.zeros(1, dtype=cum_sum.dtype, device=cum_sum.device), cum_sum], dim=0)[:batch.num_graphs]

        offset = offset[batch_edge]

        # offset = batch.ptr[:-1][batch_edge]

        src_loc = src - offset
        dst_loc = dst - offset

        mul_bias = self.Wa(batch.edge_attr)
        add_att_bias = self.Wb(batch.edge_attr)
        add_val_bias = self.Wc(batch.edge_attr).reshape(-1, self.num_heads, self.head_dim)

        mul_bias_dense = torch.ones((batch_size, self.num_heads, Nmax, Nmax), device=mul_bias.device, dtype=mul_bias.dtype) * self.Wa(self.pa).view(1, self.num_heads, 1, 1)
        add_att_bias_dense = torch.ones((batch_size, self.num_heads, Nmax, Nmax), device=add_att_bias.device, dtype=add_att_bias.dtype) * self.Wb(self.pb).view(1, self.num_heads, 1, 1)
        add_val_bias_dense = torch.ones((batch_size, self.num_heads, Nmax, Nmax, self.head_dim), device=add_val_bias.device, dtype=add_val_bias.dtype) * self.Wc(self.pc).view(1, self.num_heads, 1, self.head_dim)

        mul_bias_dense[batch_edge, :, dst_loc, src_loc] = mul_bias
        add_att_bias_dense[batch_edge, :, dst_loc, src_loc] = add_att_bias
        add_val_bias_dense[batch_edge, :, dst_loc, src_loc, :] = add_val_bias 

        attn_bias  = attn * mul_bias_dense
        attn_bias *= self.scale
        attn_bias += add_att_bias_dense

        ##Set padding values to -inf before softmax
        attn_bias = attn_bias.masked_fill(~mask_dense[:, None, None, :],float("-inf"))
        attn_bias = F.softmax(attn_bias, dim=-1)

        #out V
        out_v = torch.einsum("bhnj, bjhd -> bnhd", attn_bias, vs)

        #out bias edge
        out_edge = torch.einsum("bhnj,bhnjd->bnhd", attn_bias, add_val_bias_dense)

        out = out_v + out_edge

        out = out[mask_dense]
        out = out.reshape(out.size(0), self.out_channels)

        out = self.Wo(out)

        return out
        



class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, batch : Batch):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](batch.x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, batch.batch)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]

