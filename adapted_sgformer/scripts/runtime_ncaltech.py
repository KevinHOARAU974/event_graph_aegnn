import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse

import yaml

from torch_geometric.loader import DataLoader

from dagr.data.ncaltech101_data import NCaltech101
from adaptedsgformer.models.detection_models import DetectionGT
from adaptedsgformer.utils import format_data

from dagr.model.networks.dagr import DAGR
from argparse import Namespace

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def measure_runtime(
    model,
    dataloader,
    device="cuda",
    warmup=20,
):
    model.eval()

    # -------------------------
    # GPU warm-up
    # -------------------------
    print(f"Warm-up ({warmup} batches)...")

    for i, data in enumerate(dataloader):
        if i >= warmup:
            break

        data = data.to(device)
        data = format_data(data)
        _ = model(data)

    torch.cuda.synchronize()

    # -------------------------
    # Runtime measurement
    # -------------------------
    times = []
    num_samples = 0

    for i, data in enumerate(tqdm(dataloader, desc="Runtime")):

        data = data.to(device)
        data = format_data(data)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        _ = model(data)

        end.record()

        # Wait until GPU execution is finished
        torch.cuda.synchronize()

        # CUDA events return milliseconds
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms)

        # PyG Batch
        if hasattr(data, "num_graphs"):
            num_samples += data.num_graphs
        else:
            num_samples += 1

    times = np.asarray(times)

    total_time_ms = times.sum()

    mean_batch = times.mean()
    std_batch = times.std()
    median_batch = np.median(times)

    mean_sample = total_time_ms / num_samples
    throughput = num_samples / (total_time_ms / 1000)

    print("\n========== Runtime ==========")
    print(f"Batches measured : {len(times)}")
    print(f"Samples measured : {num_samples}")

    print(f"\nMean / batch     : {mean_batch:.3f} ms")
    print(f"Std / batch      : {std_batch:.3f} ms")
    print(f"Median / batch   : {median_batch:.3f} ms")
    print(f"Min / batch      : {times.min():.3f} ms")
    print(f"Max / batch      : {times.max():.3f} ms")

    print(f"\nMean / sample    : {mean_sample:.3f} ms")
    print(f"Throughput       : {throughput:.2f} samples/s")

    print("=============================")

    return {
        "mean_batch_ms": mean_batch,
        "std_batch_ms": std_batch,
        "median_batch_ms": median_batch,
        "mean_sample_ms": mean_sample,
        "throughput": throughput,
        "times": times,
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to YAML config file",
        )
    
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(cfg['model_path'], weights_only=False)

        
    print("init datasets")
    dataset_path = Path(cfg["data_directory"]) / cfg["dataset"]

    dataset = NCaltech101(dataset_path, "test", None, num_events=checkpoint["args"]["n_nodes"])

    loader = DataLoader(dataset, follow_batch=['bbox', 'bbox0'], shuffle=True, drop_last=True, **cfg["dataloader"])

    if cfg["model"] == 'dagt':
        args_model = checkpoint["args"]["model_params"]
        model = DetectionGT(num_classes=dataset.num_classes, args=args_model, height=dataset.height, width=dataset.width)
    else:
        args_model = checkpoint["args"]
        model = DAGR(Namespace(**args_model), height=dataset.height, width=dataset.width)

    model.to(device)

    print("Model okay")

    print("Start measure:")

    mean_latencies = []

    for i in range(cfg["nb_runs"]):

        stats = measure_runtime(model, loader, device)
        mean_latencies.append(stats["mean_sample_ms"])

    mean_latencies = np.array(mean_latencies)

    print(f"\n========== Latencies results over {cfg['nb_runs']} runs ==========")
    print(f"\nMean Latency    : {mean_latencies.mean():.3f} ms")
    print(f"\nStd Latency    : {mean_latencies.std():.3f} ms")

