import os
import torch
import torch.nn as nn
import wandb
import argparse
import yaml
import datetime
import numpy as np
import matplotlib.pyplot as plt
import random

from aegnn.utils.git import get_git_info
from aegnn.models.networks.graph_res import GraphRes

from pathlib import Path
from tqdm import tqdm

from torch_geometric.loader import DataLoader

from timm.utils import ModelEmaV3

from adaptedsgformer.models.recognition_models import AdaptedSGFormer, AEGT, DAGT
from adaptedsgformer.dataset import GraphDataset
from torch_geometric.transforms import Cartesian
from torchmetrics.functional import accuracy
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def train_one_epoch(model, train_loader, criterion, optimizer, device='cuda', ema = None):

    model.train()

    tot_loss = 0.0
    tot_acc = 0.0
    nb_sample = 0


    for batch in tqdm(train_loader):

        batch = batch.to(device)
        
        optimizer.zero_grad()

        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()

        optimizer.step()

        if ema is not None:
            ema.update(model)

        tot_loss += loss.item() * batch.batch_size
        nb_sample += batch.batch_size

        y_prediction = torch.argmax(out, dim=-1)

        tot_acc += (y_prediction == batch.y).sum().item()
        # acc_ls.append(accuracy(preds=y_prediction, target=batch.y, task="multiclass", num_classes=num_classes).item())

    return tot_loss/nb_sample, tot_acc/nb_sample

def valid_one_epoch(model, val_loader, criterion, device='cuda'):

    model.eval()

    tot_loss = 0.0
    tot_acc = 0.0
    nb_sample = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):

            batch = batch.to(device)

            out = model(batch)
            loss = criterion(out, batch.y)

            tot_loss += loss.item() * batch.batch_size
            nb_sample += batch.batch_size

            y_prediction = torch.argmax(out, dim=-1)

            tot_acc += (y_prediction == batch.y).sum().item()

    return tot_loss/nb_sample, tot_acc/nb_sample

def test_model(model, test_loader, criterion, device='cuda'):

    model.eval()

    tot_loss = 0.0
    tot_acc = 0.0
    nb_sample = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):

            batch = batch.to(device)

            out = model(batch)
            loss = criterion(out, batch.y)

            tot_loss += loss.item() * batch.batch_size
            nb_sample += batch.batch_size

            y_prediction = torch.argmax(out, dim=-1)

            tot_acc += (y_prediction == batch.y).sum().item()

    return tot_loss/nb_sample, tot_acc/nb_sample


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(cfg)

    seed = cfg['seed']

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ### Datasets and Dataloader
    root = Path(os.path.expanduser(cfg['dataset']['root'])) 

    transform = None
    if cfg["model"] == "aegnn":
        transform = Cartesian(cat=False,norm=True)

    train_dataset = GraphDataset(root / cfg['dataset']['name'] / 'processed' / 'training',transform=transform)
    val_dataset = GraphDataset(root / cfg['dataset']['name'] / 'processed' / 'validation',transform=transform)
    test_dataset = GraphDataset(root / cfg['dataset']['name'] / 'processed' / 'test',transform=transform)

    cfg_ds = load_config(root / cfg['dataset']['name'] / "dataset.yaml")

    cfg_ds["max_time_training"] = float(cfg_ds["max_time_training"])
    cfg_ds["max_time_validation"] = float(cfg_ds["max_time_validation"])
    cfg_ds["max_time_test"] = float(cfg_ds["max_time_test"])

    cfg_ds["factors"][2] = float(cfg_ds["factors"][2])

    w = cfg_ds["width"]
    h = cfg_ds["height"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, **cfg['dataloader'])
    val_dataloader = DataLoader(val_dataset, shuffle=False, **cfg['dataloader'])
    test_dataloader = DataLoader(test_dataset, shuffle=False, **cfg['dataloader'])

    print("Dataloaders : Check")

    ### Model
    if cfg["model"] == 'adapted_sgformer':
        model = AdaptedSGFormer(**cfg['model_params'])

    elif cfg["model"] == 'aegt': 

        cfg["model_params"]["max_periods"] = [w, h, cfg_ds["max_time_training"]]
        cfg["model_params"]["input_shape"] = [w, h]
        cfg["model_params"]["factors"] = cfg_ds["factors"]

        model = AEGT(**cfg['model_params'])
        # cfg['model_params']['pooling_size'] = tuple(cfg['model_params']['pooling_size'])

    elif cfg["model"] == 'dagt':
        
        cfg["model_params"]["height"] = h
        cfg["model_params"]["width"] = w
        cfg["model_params"]["encoding_periods"] = [w, h, cfg_ds["max_time_training"]]
        cfg["model_params"]["factors"] = cfg_ds["factors"]

        model = DAGT(**cfg['model_params'])

    elif cfg["model"] == "aegnn":

        cfg["model_params"]["input_shape"] = torch.tensor([w, h, 3])
        model = GraphRes(**cfg['model_params'])
        transform = Cartesian(cat=False,norm=True)
    

    # num_classes = cfg['model_params']['out_channels']

    print(f'Model {cfg["model"]}: Check')

    ### Criterion, optimizer, scheduler

    criterion_train = nn.CrossEntropyLoss(**cfg['criterion'])
    criterion_test = nn.CrossEntropyLoss()

    cfg['optimizer']['lr'] = float(cfg['optimizer']['lr'])
    cfg['optimizer']['weight_decay'] = float(cfg['optimizer']['weight_decay']) if cfg['optimizer']['weight_decay'] != None else None
    cfg['scheduler']['eta_min'] = float(cfg['scheduler']['eta_min'])

    warmup_epochs = cfg["warmup_epochs"]
    max_epochs = cfg["max_epochs"] 

    if cfg["model"] == 'adapted_sgformer':
        optimizer = torch.optim.AdamW([
                {'params': model.params1},
                {'params': model.params2}
            ],
            **cfg['optimizer'])
    elif cfg["model"] in ("aegt", "dagt", "aegnn"):
        optimizer = torch.optim.AdamW(model.parameters(),
            **cfg['optimizer'])

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs-warmup_epochs, **cfg['scheduler'])

    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Optimizer : Check")
    
    #Log directory

    date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_path = os.path.expanduser(os.path.join(cfg["log_dir"], "checkpoints", cfg['model'] ,cfg["dataset"]["name"], cfg["task"], date_time))
    Path(checkpoint_path).mkdir(parents=True,exist_ok=True)
    
    #wandb setup

    wandb.init(
    project=cfg['project_name'],
    name=f"{cfg['model']}-{cfg['task']}-{date_time}",
    config=cfg)   

    wandb.config.update(get_git_info())
    wandb.config.update({'checkpoint_dir': checkpoint_path,
                         'device' : device})

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    #Early stopping
    best_acc = -1
    best_acc_ema = -1
    best_loss = float("inf")
    best_loss_ema = float("inf")

    patience = cfg['early_stopping']["patience"]
    min_delta_loss = 1e-3
    min_delta_acc = 1e-4
    min_epochs = int(cfg['early_stopping']["min_epochs"])
    metric = cfg['early_stopping']["metric"]

    epochs_without_improvement = 0
    
    model.to(device)


    # Ema model
    model_ema = None
    if cfg["ema"]:
        model_ema = ModelEmaV3(model, decay=cfg["ema_decay"])

    print("Start training")

    for epoch in tqdm(range(cfg['max_epochs'])):

        train_loss, train_acc = train_one_epoch(
            model,
            train_dataloader,
            criterion_train,
            optimizer,
            device=device,
            ema=model_ema
        )

        val_loss, val_acc = valid_one_epoch(
            model,
            val_dataloader,
            criterion_test,
            device=device
        )

        if model_ema is not None:
            val_loss_ema, val_acc_ema = valid_one_epoch(
                model_ema.module,
                val_dataloader,
                criterion_test,
                device=device
            )

        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log({
            'epoch' : epoch,
            'train/loss' : train_loss,
            'train/acc' : train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'lr': current_lr,
        })

        if model_ema is not None:
            wandb.log({
            'val/loss_ema': val_loss_ema,
            'val/acc_ema': val_acc_ema,
            })
        
        scheduler.step()
        
        #Compute the improvement of each metric
        improved_loss = val_loss < best_loss - min_delta_loss
        improved_acc = val_acc > best_acc + min_delta_acc

        if model_ema is not None:
            improved_loss_ema = val_loss_ema < best_loss_ema - min_delta_loss
            improved_acc_ema = val_acc_ema > best_acc + min_delta_acc


        if improved_loss:

            best_loss = val_loss

            if cfg["checkpoint_save"]["best_loss"]:
                torch.save({
                            "model_state_dict": model.state_dict(), 
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch,
                            "best_val_loss": best_loss,
                            "cfg": cfg,
                            }, f"{checkpoint_path}/best_loss.pth")
        
        if improved_acc:

            best_acc = best_acc

            if cfg["checkpoint_save"]["best_acc"]:
                torch.save({
                            "model_state_dict": model.state_dict(), 
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch,
                            "best_val_acc": best_acc,
                            "cfg": cfg,
                            }, f"{checkpoint_path}/best_acc.pth")
                
        if model_ema is not None:
            if improved_loss_ema:

                best_loss_ema = val_loss_ema

                if cfg["checkpoint_save"]["best_loss_ema"]:
                    torch.save({
                                "model_state_dict": model_ema.module.state_dict(), 
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "epoch": epoch,
                                "best_val_loss_ema": best_loss_ema,
                                "cfg": cfg,
                                }, f"{checkpoint_path}/best_loss_ema.pth")
            
            if improved_acc_ema:

                best_acc_ema = val_acc_ema

                if cfg["checkpoint_save"]["best_acc_ema"]:
                    torch.save({
                                "model_state_dict": model_ema.module.state_dict(), 
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "epoch": epoch,
                                "best_val_acc_ema": best_acc_ema,
                                "cfg": cfg,
                                }, f"{checkpoint_path}/best_acc_ema.pth")
        
        #early stopping criteria

        if metric == "loss" and improved_loss:
            epochs_without_improvement = 0
        elif metric == "acc" and improved_acc:
            epochs_without_improvement = 0
        elif model_ema is not None:
            if metric == "loss_ema" and improved_loss_ema:
                epochs_without_improvement = 0
            elif metric == "acc_ema" and improved_acc_ema:
                epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        
        torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_state_dict_ema": model_ema.module.state_dict() if model_ema is not None else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "last_val_acc": val_acc,
                    "last_val_loss": val_loss,
                    "last_val_acc_ema": val_acc_ema if model_ema is not None else None,
                    "last_val_loss_ema": val_loss_ema if model_ema is not None else None,
                    "cfg": cfg,
                    }, f"{checkpoint_path}/last.pth")
        
        print(f'Epoch {epoch}:')
        print(f'Train_loss : {train_loss}, Train_acc : {train_acc}')
        print(f'Val_loss : {val_loss}, Val_acc : {val_acc}')

        if model_ema is not None:
            print(f'Val_loss_ema : {val_loss_ema}, Val_acc_ema : {val_acc_ema}')

        if epochs_without_improvement >= patience and epoch >= min_epochs:
            print(f"Early stopping at epoch: {epoch}")
            break
    

    ### TO DO ###
    ################Test#######################

    #Load best model in accuracy


    if cfg["checkpoint_save"]["best_acc"]:
        best_checkpoint_acc = torch.load(f"{checkpoint_path}/best_acc.pth", weights_only=False)

        if cfg["model"] == 'adapted_sgformer':
            best_model = AdaptedSGFormer(**cfg['model_params'])
        elif cfg["model"] == 'aegt':
            best_model = AEGT(**cfg['model_params'])
        elif cfg["model"] == 'dagt':
            best_model = DAGT(**cfg['model_params'])
        elif cfg["model"] == "aegnn":
            best_model = GraphRes(**cfg['model_params'])

        best_model.load_state_dict(best_checkpoint_acc["model_state_dict"])

        best_model.to(device)

        test_loss, test_acc = test_model(best_model, test_dataloader, criterion_test, device=device)


        wandb.log({
                # 'epoch' : epoch,
                'best_acc_model/loss' : test_loss,
                'best_acc_model/acc' : test_acc,
            })
    

    #Load best model in loss
    if cfg["checkpoint_save"]["best_loss"]:

        best_checkpoint_loss = torch.load(f"{checkpoint_path}/best_loss.pth", weights_only=False)

        if cfg["model"] == 'adapted_sgformer':
            best_model = AdaptedSGFormer(**cfg['model_params'])
        elif cfg["model"] == 'aegt':
            best_model = AEGT(**cfg['model_params'])
        elif cfg["model"] == 'dagt':
            best_model = DAGT(**cfg['model_params'])
        elif cfg["model"] == "aegnn":
            best_model = GraphRes(**cfg['model_params'])

        best_model.load_state_dict(best_checkpoint_loss["model_state_dict"])

        best_model.to(device)

        # Test the best model

        test_loss, test_acc = test_model(best_model, test_dataloader, criterion_test, device=device)


        wandb.log({
                # 'epoch' : epoch,
                'best_loss_model/loss' : test_loss,
                'best_loss_model/acc' : test_acc,
            })

    if model_ema is not None: 

        if cfg["checkpoint_save"]["best_acc_ema"]:

            best_checkpoint_acc = torch.load(f"{checkpoint_path}/best_acc_ema.pth", weights_only=False)

            if cfg["model"] == 'adapted_sgformer':
                best_model = AdaptedSGFormer(**cfg['model_params'])
            elif cfg["model"] == 'aegt':
                best_model = AEGT(**cfg['model_params'])
            elif cfg["model"] == 'dagt':
                best_model = DAGT(**cfg['model_params'])
            elif cfg["model"] == "aegnn":
                best_model = GraphRes(**cfg['model_params'])

            best_model.load_state_dict(best_checkpoint_acc["model_state_dict"])

            best_model.to(device)

            test_loss, test_acc = test_model(best_model, test_dataloader, criterion_test, device=device)


            wandb.log({
                    # 'epoch' : epoch,
                    'best_acc_ema_model/loss' : test_loss,
                    'best_acc_ema_model/acc' : test_acc,
                })
        

        #Load best model in loss
        if cfg["checkpoint_save"]["best_loss_ema"]:

            best_checkpoint_loss = torch.load(f"{checkpoint_path}/best_loss_ema.pth", weights_only=False)

            if cfg["model"] == 'adapted_sgformer':
                best_model = AdaptedSGFormer(**cfg['model_params'])
            elif cfg["model"] == 'aegt':
                best_model = AEGT(**cfg['model_params'])
            elif cfg["model"] == 'dagt':
                best_model = DAGT(**cfg['model_params'])
            elif cfg["model"] == "aegnn":
                best_model = GraphRes(**cfg['model_params'])

            best_model.load_state_dict(best_checkpoint_loss["model_state_dict"])

            best_model.to(device)

            # Test the best model

            test_loss, test_acc = test_model(best_model, test_dataloader, criterion_test, device=device)


            wandb.log({
                    # 'epoch' : epoch,
                    'best_loss_ema_model/loss' : test_loss,
                    'best_loss_ema_model/acc' : test_acc,
                })

    wandb.finish()

if __name__ == "__main__":
    main()