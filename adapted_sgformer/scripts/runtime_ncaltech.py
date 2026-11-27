import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import random
import gc
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        model(data)

    torch.cuda.synchronize()

    # -------------------------
    # Runtime measurement
    # -------------------------
    times = []
    num_samples = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for data in tqdm(dataloader, desc="Runtime"):

        data = data.to(device)
        data = format_data(data)

        start.record()

        model(data)

        end.record()

        end.synchronize()

        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms)

        num_samples += getattr(data, "num_graphs", 1)

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
        "num_samples": num_samples,
        "times": times,
    }


def load_model(model_cfg, checkpoint, dataset, device):

    model_type = model_cfg["type"].lower()

    if model_type == "dagt":

        args_model = checkpoint["args"]["model_params"]

        model = DetectionGT(
            num_classes=dataset.num_classes,
            args=args_model,
            height=dataset.height,
            width=dataset.width,
            sparse=False
        )

        model.load_state_dict(checkpoint['ema'], strict=False)



    elif model_type == "dagr":

        args_model = checkpoint["args"]

        model = DAGR(
            Namespace(**args_model),
            height=dataset.height,
            width=dataset.width,
        )

        model.load_state_dict(checkpoint['ema'], strict=False)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Adapt this key to your checkpoint structure if needed
    model.load_state_dict(checkpoint["model"])

    model = model.to(device)
    model.eval()

    return model


def benchmark_model(model_cfg, cfg, device):

    print("\n" + "=" * 70)
    print(f"Benchmark: {model_cfg['name']}")
    print(f"Checkpoint: {model_cfg['path']}")
    print("=" * 70)

    seed = cfg.get("seed", 12345)
    set_seed(seed)

    # Load checkpoint on CPU
    checkpoint = torch.load(
        model_cfg["path"],
        map_location="cpu",
        weights_only=False,
    )

    # -------------------------
    # Dataset
    # -------------------------
    dataset_path = (
        Path(cfg["data_directory"]) / cfg["dataset"]
    )

    dataset = NCaltech101(
        dataset_path,
        "test",
        None,
        num_events=checkpoint["args"]["n_nodes"],
    )

    loader = DataLoader(
        dataset,
        follow_batch=["bbox", "bbox0"],
        shuffle=False,
        drop_last=False,
        **cfg["dataloader"],
    )

    # -------------------------
    # Model
    # -------------------------
    model = load_model(
        model_cfg,
        checkpoint,
        dataset,
        device,
    )

    num_params = sum(
        p.numel() for p in model.parameters()
    )

    num_events = checkpoint["args"]["n_nodes"]

    del checkpoint

    print(f"Parameters : {num_params:,}")
    print(f"Max events : {num_events}")
    print(f"Dataset    : {len(dataset)} samples")

    # -------------------------
    # Benchmark repetitions
    # -------------------------
    mean_latencies = []

    for i in range(cfg["nb_runs"]):

        print(f"\nRun {i + 1}/{cfg['nb_runs']}")

        set_seed(seed + i)

        stats = measure_runtime(
            model,
            loader,
            device=device,
            warmup=cfg.get("warmup", 20),
        )

        mean_latencies.append(
            stats["mean_sample_ms"]
        )

    mean_latencies = np.asarray(mean_latencies)

    mean_latency = mean_latencies.mean()
    std_latency = mean_latencies.std()

    result = {
        "name": model_cfg["name"],
        "num_params": num_params,
        "num_events": num_events,
        "num_samples": stats["num_samples"],
        "mean_latency": mean_latency,
        "std_latency": std_latency,
        "throughput": 1000 / mean_latency,
        "latencies": mean_latencies,
    }

    print("\n========== Model results ==========")
    print(f"Model        : {result['name']}")
    print(f"Mean latency : {mean_latency:.3f} ms")
    print(f"Std latency  : {std_latency:.3f} ms")
    print("===================================")

    # -------------------------
    # GPU memory cleanup
    # -------------------------
    del model
    del loader
    del dataset

    gc.collect()
    torch.cuda.empty_cache()

    return result


def print_summary(results, nb_runs, batch_size):

    # First model is used as reference
    reference_latency = results[0]["mean_latency"]

    print("\n" + "=" * 110)
    print("FINAL RUNTIME COMPARISON")
    print(f"Runs: {nb_runs} | Batch size: {batch_size}")
    print(f"Reference: {results[0]['name']}")
    print("=" * 110)

    header = (
        f"{'Model':<22}"
        f"{'Params':>12}"
        f"{'Latency (ms)':>22}"
        f"{'Samples/s':>14}"
        f"{'Speedup':>12}"
        f"{'Reduction':>14}"
    )

    print(header)
    print("-" * 110)

    for result in results:

        latency = result["mean_latency"]
        std = result["std_latency"]

        speedup = reference_latency / latency

        reduction = (
            1 - latency / reference_latency
        ) * 100

        latency_str = f"{latency:.2f} ± {std:.2f}"
        speedup_str = f"{speedup:.2f}x"
        reduction_str = f"{reduction:+.2f}%"

        print(
            f"{result['name']:<22}"
            f"{result['num_params']:>12,}"
            f"{latency_str:>22}"
            f"{result['throughput']:>14.2f}"
            f"{speedup_str:>12}"
            f"{reduction_str:>14}"
        )

    print("=" * 110)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = "cuda"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    results = []

    for model_cfg in cfg["models"]:

        result = benchmark_model(
            model_cfg,
            cfg,
            device,
        )

        results.append(result)

    print_summary(
        results,
        nb_runs=cfg["nb_runs"],
        batch_size=cfg["dataloader"]["batch_size"],
    )