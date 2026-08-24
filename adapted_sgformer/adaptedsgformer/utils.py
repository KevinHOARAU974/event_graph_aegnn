import torch
import numpy as np

def embed_1D_scalar(t, dim, max_period):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def consecutive_cluster(src):
    unique, inv, counts = torch.unique(src, sorted=True, return_inverse=True, return_counts=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return unique, inv, perm, counts

def compute_pooling_at_each_layer(pooling_dim_at_output, num_layers):
    py, px = map(int, pooling_dim_at_output.split("x"))
    pooling_base = torch.tensor([px, py])
    poolings = []
    for i in range(num_layers):
        pooling = pooling_base * 2 ** (3 - i)
        poolings.append(pooling)
    poolings = torch.stack(poolings)
    return poolings

def to_dense(self, x, pos, pooling, batch=None, batch_size=None):
    # if hasattr(self, "batch_size"):
    #     B = self.batch_size
    if batch_size is not None:
        self.batch_size = batch_size
        B = batch_size
    elif batch is None:
        batch = torch.zeros(size=(len(x),), dtype=torch.long, device=x.device)
        B = 1
        self.batch_size = B
    else:
        B = batch.max().item() + 1
        self.batch_size = B

    if not hasattr(self, "dense"):
        W, H = (1 / pooling[:2] + 1e-3).long()
        C = x.shape[-1]
        self.dense = torch.zeros(size=(B, C, H, W), dtype=x.dtype, device=x.device)

    est_x, est_y = (pos[:, :2] / pooling[:2]).t().long()

    self.dense = self.dense.detach()
    self.dense.zero_()

    dense = self.dense[:B] if B < self.dense.shape[0] else self.dense
    dense[batch.long(), :, est_y, est_x] = x

    return dense