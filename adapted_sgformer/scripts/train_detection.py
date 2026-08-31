# avoid matlab error on server
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import tqdm
import wandb
from pathlib import Path
import argparse
import yaml

from torch_geometric.loader import DataLoader

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams, ResumeMode
from dagr.utils.buffers import DetectionBuffer
# from dagr.utils.args import FLAGS
from dagr.utils.learning_rate_scheduler import LRSchedule

from dagr.data.augment import Augmentations
# from dagr.utils.buffers import format_data
from dagr.data.ncaltech101_data import NCaltech101

from dagr.model.networks.ema import ModelEMA

from adaptedsgformer.models.detection_models import DetectionGT
from adaptedsgformer.utils import format_data
from adaptedsgformer.scheduler import WarmupCosineScheduler

from argparse import Namespace


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def gradients_broken(model):
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            valid_gradients = not (torch.isnan(param.grad).any())
            if not valid_gradients:
                break
    return not valid_gradients

def fix_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.nan_to_num(param.grad, nan=0.0)


def train(loader: DataLoader,
          model: torch.nn.Module,
          ema: ModelEMA,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          optimizer: torch.optim.Optimizer,
          clip_value: int,
          run_name=""):

    model.train()

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):
        data = data.cuda(non_blocking=True)
        data = format_data(data) #Normalize data

        optimizer.zero_grad(set_to_none=True)

        model_outputs = model(data)

        loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
        loss = loss_dict.pop("total_loss")

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)

        fix_gradients(model)

        optimizer.step()
        scheduler.step()

        ema.update(model)

        training_logs = {f"training/loss/{k}": v for k, v in loss_dict.items()}
        wandb.log({"training/loss": loss.item(), "training/lr": scheduler.get_last_lr()[-1], **training_logs})

def run_test(loader: DataLoader,
         model: torch.nn.Module,
         dry_run_steps: int=-1,
         dataset="gen1"):

    model.eval()

    mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)

    for i, data in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        data = format_data(data)

        detections, targets = model(data)
        if i % 10 == 0:
            torch.cuda.empty_cache()

        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])

        if dry_run_steps > 0 and i == dry_run_steps:
            break

    torch.cuda.empty_cache()

    return mapcalc

if __name__ == '__main__':
    import torch_geometric
    import random
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = cfg['seed']
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    resume_mode = ResumeMode(cfg["resume"])

    checkpoint_path = None
    wandb_run_id = None

    #Resume 
    if resume_mode != ResumeMode.NONE:
        if cfg["resume_directory"] is None:
            raise ValueError("--resume-directory is required when resuming training")

        temporary_checkpointer = Checkpointer()

        checkpoint_path = temporary_checkpointer.search_for_checkpoint(Path(cfg["resume_directory"]), best=resume_mode == ResumeMode.BEST)

        if checkpoint_path is None:
            raise FileExistsError(f'No checkpoint found in {cfg["resume_directory"]}')

        checkpoint_metadata = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        wandb_run_id = checkpoint_metadata.get("wandb_run_id")

        if wandb_run_id is None:
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain 'wandb_run_id"
            )

    
    cfg["output_directory"] = set_up_logging_directory(cfg["dataset"], cfg["task"], Path(cfg["output_directory"]), exp_name=cfg["exp_name"], wandb_run_id=wandb_run_id)

    wandb.define_metric("epoch")

    for split in ("validation", "test"):
        wandb.define_metric(
            f"{split}/metric/*",
            step_metric="epoch",
        )

    # log_hparams(args)

    augmentations = Augmentations(Namespace(**cfg["augmentations"]))

    #log config on wandb
    wandb.config.update(cfg, allow_val_change= resume_mode != ResumeMode.NONE)

    print("init datasets")
    dataset_path = Path(cfg["data_directory"]) / cfg["dataset"]

    train_dataset = NCaltech101(dataset_path, "training", augmentations.transform_training, num_events=cfg["n_nodes"])
    val_dataset = NCaltech101(dataset_path, "validation", augmentations.transform_testing, num_events=cfg["n_nodes"])
    test_dataset = NCaltech101(dataset_path, "test", augmentations.transform_testing, num_events=cfg["n_nodes"])
    

    train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], shuffle=True, drop_last=True, **cfg["dataloader"])
    num_iters_per_epoch = len(train_loader)

    sampler = np.random.permutation(np.arange(len(val_dataset)))
    val_loader = DataLoader(val_dataset, sampler=sampler, follow_batch=['bbox', 'bbox0'], shuffle=False, drop_last=False, **cfg["dataloader"])

    test_loader = DataLoader(test_dataset, sampler=sampler, follow_batch=['bbox', 'bbox0'], shuffle=False, drop_last=False, **cfg["dataloader"])

    print(f"height :{train_dataset.height}")
    print(f"width :{train_dataset.width}")

    print("init net")
    # load a dummy sample to get height, width
    model = DetectionGT(num_classes=train_dataset.num_classes, args=cfg["model_params"], height=train_dataset.height, width=train_dataset.width)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Training with {num_params} number of parameters.")

    wandb.config.update({
        'num_params': num_params
    })

    model = model.cuda()
    ema = ModelEMA(model)

    nominal_batch_size = 64
    cfg["optimizer"]["lr"] = float(cfg["optimizer"]["lr"]) * np.sqrt(cfg["dataloader"]["batch_size"]) / np.sqrt(nominal_batch_size)
    optimizer = torch.optim.AdamW(list(model.parameters()), **cfg["optimizer"])

    if cfg['scheduler_type'] == "cosineTrans":

        lr_func = WarmupCosineScheduler(warmup_epochs=cfg["warmup_epochs"],
                         num_iters_per_epoch=num_iters_per_epoch,
                         tot_num_epochs=cfg["scheduler_max_epochs"])

    else:
        lr_func = LRSchedule(warmup_epochs=cfg["warmup_epochs"],
                            num_iters_per_epoch=num_iters_per_epoch,
                            tot_num_epochs=cfg["scheduler_max_epochs"])

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    checkpointer = Checkpointer(output_directory=cfg["output_directory"],
                                model=model, optimizer=optimizer,
                                scheduler=lr_scheduler, ema=ema,
                                args=Namespace(**cfg))

    checkpoint_path = checkpointer.restore(cfg["output_directory"], mode=ResumeMode(cfg["resume"]))

    start_epoch = 0
    if ResumeMode(cfg["resume"]) != ResumeMode.NONE and checkpoint_path is not None:
        start_epoch = checkpointer.restore_checkpoint(checkpoint_path) + 1
        print(f"Resume from checkpoint at epoch {start_epoch}")

    with torch.no_grad():
        mapcalc = run_test(val_loader, ema.ema, dry_run_steps=2, dataset=cfg["dataset"])
        mapcalc.compute()

    print(model)

    print("starting to train")
    for epoch in range(start_epoch, cfg["max_epochs"]):

        print(f"epoch n°: {epoch}")
        train(train_loader, model, ema, lr_scheduler, optimizer, cfg["clip"], run_name=wandb.run.name)
        checkpointer.checkpoint(epoch, name=f"last_model")

        if epoch % 3 > 0:
            continue

        with torch.no_grad():
            mapcalc = run_test(val_loader, ema.ema, dataset=cfg["dataset"])
            metrics = mapcalc.compute()
            checkpointer.process(metrics, epoch, "validation")

    print("End of training")

    print("Starting to test")

    best_checkpoint_path = checkpointer.search_for_checkpoint(Path(cfg["output_directory"]), best= True)

    if best_checkpoint_path is None:
        raise FileExistsError(
            f'No best checkpoint found in {cfg["output_directory"]}'
        )

    best_epoch = checkpointer.restore_checkpoint(best_checkpoint_path)

    print(
        f"Running final test with checkpoint {best_checkpoint_path.name} from {best_epoch}"
    )

    with torch.no_grad():
            mapcalc = run_test(test_loader, ema.ema, dataset=cfg["dataset"])
            metrics = mapcalc.compute()
            checkpointer.process(metrics, epoch, "test")

    wandb.finish()