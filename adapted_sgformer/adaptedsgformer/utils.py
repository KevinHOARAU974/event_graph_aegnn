import torch
import numpy as np

from dagr.model.utils import batched_nms_coordinate_trick,_sequential_counter

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
    pooling_base = torch.tensor([1.0 / px, 1.0 / py, 1.0 / 1])
    sampling_base = torch.tensor([px, py])
    poolings = []
    samplings = []

    for i in range(num_layers):
        pooling = pooling_base / 2 ** (num_layers - 1 - i)
        sampling = sampling_base * 2 ** (num_layers - 1 - i)
        pooling[-1] = 1
        sampling = sampling[0] * sampling[1]
        poolings.append(pooling)
        samplings.append(sampling)

    poolings = torch.stack(poolings)
    samplings = torch.stack(samplings)

    return poolings, samplings

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

    
    W, H = (1 / pooling[:2] + 1e-3).long()
    C = x.shape[-1]

    if (
        not hasattr(self, "dense")
        or self.dense.shape[0] < B
        or self.dense.shape[1:] != (C, H, W)
        or self.dense.device != x.device
        or self.dense.dtype != x.dtype
    ):
        self.dense = x.new_zeros((B, C, H, W))

    est_x, est_y = (pos[:, :2] / pooling[:2]).t().long()

    self.dense = self.dense.detach()
    self.dense.zero_()

    dense = self.dense[:B] if B < self.dense.shape[0] else self.dense
    
    dense[batch.long(), :, est_y, est_x] = x

    return dense

def format_data(data, normalizer=None):
    if normalizer is None:
        normalizer = torch.stack([data.width[0], data.height[0], data.time_window[0]], dim=-1)

    if hasattr(data, "image"):
        data.image = data.image.float() / 255.0

    data.pos = torch.cat([data.pos, data.t.view((-1,1))], dim=-1)
    data.t = None
    data.x = ((1 - data.x)//2).float()
    data.pos = data.pos / normalizer
    return data

def check_graphs(data, stage, log_file="graph_debug.log"):

    # IDs de graphes qui possèdent effectivement des noeuds
    present_ids, counts = torch.unique(
        data.batch,
        return_counts=True
    )

    # Nombre de samples attendus dans le batch
    expected_graphs = data.num_graphs

    expected_ids = torch.arange(
        expected_graphs,
        device=data.batch.device
    )

    # IDs de graphes sans aucun noeud
    missing_ids = expected_ids[
        ~torch.isin(expected_ids, present_ids)
    ]

    # Indices des samples dans le Dataset
    sample_indices = (
        data.sample_idx.view(-1).detach().cpu().tolist()
        if hasattr(data, "sample_idx")
        else None
    )

    missing_ids_cpu = missing_ids.detach().cpu().tolist()

    # Correspondance :
    # graph id dans le batch -> index original dans le Dataset
    if sample_indices is not None:
        missing_samples = [
            sample_indices[i]
            for i in missing_ids_cpu
            if i < len(sample_indices)
        ]
    else:
        missing_samples = None

    line = (
        f"{stage} | "
        f"nodes={data.x.shape[0]} | "
        f"expected_graphs={expected_graphs} | "
        f"present_graphs={len(present_ids)} | "
        f"missing={missing_ids_cpu} | "
        f"missing_samples={missing_samples} | "
        f"sample_idx={sample_indices} | "
        f"counts={counts.detach().cpu().tolist()}\n"
    )

    with open(log_file, "a") as f:
        f.write(line)

    # Affiche uniquement les anomalies dans le terminal
    if missing_ids_cpu:
        print(
            f"\n⚠️ {stage}: "
            f"missing graph IDs={missing_ids_cpu}, "
            f"dataset samples={missing_samples}"
        )

def postprocess_network_output(prediction, num_classes, batch_size=None ,batch_pred=None, conf_thre=0.01, nms_thre=0.65, height=640, width=640, filtering=True, sparse=False):
    prediction[..., :2] -= prediction[...,2:4] / 2 # cxcywh->xywh
    prediction[..., 2:4] += prediction[...,:2]

    if sparse:
        predictions = []

        for batch_idx in range(batch_size):
            mask = batch_pred == batch_idx
            predictions.append(prediction[mask])

    else:
        predictions = prediction
    
    output = []
    for i, image_pred in enumerate(predictions):

        # If none are remaining => process next image
        if len(image_pred) == 0:
            device = predictions.device
            output.append({
                "boxes": torch.zeros(0, 4, dtype=torch.float32, device=device),
                "scores": torch.zeros(0, dtype=torch.float, device=device),
                "labels": torch.zeros(0, dtype=torch.long, device=device)
            })
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        image_pred[:, 4:5] *= class_conf

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_pred), 1)

        if filtering:
            detections = detections[conf_mask]

        if len(detections) == 0:
            device = predictions.device
            output.append({
                "boxes": torch.zeros(0, 4, dtype=torch.float32, device=device),
                "scores": torch.zeros(0, dtype=torch.float, device=device),
                "labels": torch.zeros(0, dtype=torch.long, device=device)
            })
            continue

        nms_out_index = batched_nms_coordinate_trick(detections[:, :4], detections[:, 4], detections[:, 5],
                                                      nms_thre, width=width, height=height)

        if filtering:
            detections = detections[nms_out_index]

        output.append({
            "boxes": detections[:, :4],
            "scores": detections[:, 4],
            "labels": detections[:, -1].long()
        })

    return output


def convert_to_training_format(bbox, batch, batch_size, bbox_batch=None, sparse=True):

    if sparse:

        labels = bbox
        labels[:, :2] += labels[:, 2:4] * .5
        labels = torch.roll(labels[:, :5], dims=1, shifts=1)

        targets = [labels, bbox_batch]

    else:
        max_detections = 100
        targets = torch.zeros(size=(batch_size, max_detections, 5), dtype=torch.float32, device=bbox.device)
        unique, counts = torch.unique(batch, return_counts=True)
        counter = _sequential_counter(counts)

        bbox = bbox.clone()
        # xywhlc pix -> lcxcywh pix
        bbox[:, :2] += bbox[:, 2:4] * .5
        bbox = torch.roll(bbox[:, :5], dims=1, shifts=1)

        targets[batch, counter] = bbox

    return targets