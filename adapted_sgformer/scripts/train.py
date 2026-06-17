import os
import torch
import torch.nn as nn
import wandb
import argparse
import yaml
import datetime
import numpy as np
import matplotlib.pyplot as plt

from aegnn.utils.git import get_git_info

from pathlib import Path
from tqdm import tqdm

from torch_geometric.loader import DataLoader

from adaptedsgformer.model import AdaptedSGFormer, AEGT
from adaptedsgformer.dataset import GraphDataset
from torchmetrics.functional import accuracy
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, num_classes = 2, device='cuda'):

    model.to(device)
    model.train()

    loss_ls = []
    acc_ls = []

    for batch in tqdm(train_loader):

        batch = batch.to(device)
        
        optimizer.zero_grad()

        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()

        optimizer.step()

        loss_ls.append(loss.item())

        y_prediction = torch.argmax(out, dim=-1)
        acc_ls.append(accuracy(preds=y_prediction, target=batch.y, task="multiclass", num_classes=num_classes).item())

    scheduler.step()

    return np.mean(np.array(loss_ls)), np.mean(np.array(acc_ls))



def valid_one_epoch(model, val_loader, criterion, num_classes = 2, device='cuda'):

    model.to(device)
    model.eval()

    loss_ls = []
    acc_ls = []

    with torch.no_grad():
        for batch in tqdm(val_loader):

            batch = batch.to(device)

            out = model(batch)
            loss = criterion(out, batch.y)

            loss_ls.append(loss.item())

            y_prediction = torch.argmax(out, dim=-1)
            acc_ls.append(accuracy(preds=y_prediction, target=batch.y, task="multiclass", num_classes=num_classes).item())

    return np.mean(np.array(loss_ls)), np.mean(np.array(acc_ls))

def test_model(model, test_loader, criterion, checkpoint_path, num_classes = 2, device='cuda'):

    model.to(device)
    model.eval()

    loss_ls = []
    acc_ls = []

    y_targets = []
    y_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader):

            batch = batch.to(device)

            out = model(batch)
            loss = criterion(out, batch.y)

            loss_ls.append(loss.item())

            y_prediction = torch.argmax(out, dim=-1)
            acc_ls.append(accuracy(preds=y_prediction, target=batch.y, task="multiclass", num_classes=num_classes).item())

            y_targets.append(batch.y)
            y_preds.append(y_prediction)
    
    preds = torch.cat(y_preds)
    targets = torch.cat(y_targets)

    cm = confusion_matrix(targets.cpu().numpy(), 
                          preds.cpu().numpy())

    fig, ax = plt.subplots(figsize=(8, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        # display_labels=data_module.classes
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=True
    )

    plt.tight_layout()
    plt.savefig(f"{checkpoint_path}/confusion_matrix.png", dpi=300)
    plt.close()

    return np.mean(np.array(loss_ls)), np.mean(np.array(acc_ls))


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

    torch.manual_seed(cfg['seed'])

    ### Datasets and Dataloader
    root = Path(os.path.expanduser(cfg['dataset']['root'])) 

    train_dataset = GraphDataset(root / cfg['dataset']['name'] / 'processed' / 'training')
    val_dataset = GraphDataset(root / cfg['dataset']['name'] / 'processed' / 'validation')
    test_dataset = GraphDataset(root / cfg['dataset']['name'] / 'processed' / 'test')

    train_dataloader = DataLoader(train_dataset, shuffle=True, **cfg['dataloader'])
    val_dataloader = DataLoader(val_dataset, shuffle=False, **cfg['dataloader'])
    test_dataloader = DataLoader(test_dataset, shuffle=False, **cfg['dataloader'])

    print("Dataloaders : Check")
    ### Model

    if cfg["model"] == 'adapted_sgformer':
        model = AdaptedSGFormer(**cfg['model_params_asgf'])
        num_classes = cfg['model_params_asgf']['out_channels']
    elif cfg["model"] == 'AEGT':
        model = AEGT(**cfg['model_params_aegt'])
        num_classes = cfg['model_params_aegt']['out_channels']
        cfg['model_params_aegt']['pooling_size'] = tuple(cfg['model_params_aegt']['pooling_size'])

    print(f'Model {cfg["model"]}: Check')

    ### Criterion, optimizer, scheduler

    criterion_train = nn.CrossEntropyLoss(**cfg['criterion'])
    criterion_test = nn.CrossEntropyLoss()

    cfg['optimizer']['lr'] = float(cfg['optimizer']['lr'])
    cfg['optimizer']['weight_decay'] = float(cfg['optimizer']['weight_decay'])
    cfg['scheduler']['eta_min'] = float(cfg['scheduler']['eta_min'])

    if cfg["model"] == 'adapted_sgformer':
        optimizer = torch.optim.Adam([
                {'params': model.params1},
                {'params': model.params2}
            ],
            **cfg['optimizer'])
    elif cfg["model"] == 'AEGT':
        optimizer = torch.optim.Adam(model.parameters(),
            **cfg['optimizer'])

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['max_epochs'], **cfg['scheduler'])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Optimizer : Check")
    
    #Log directory

    date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_path = os.path.expanduser(os.path.join(cfg["log_dir"], "checkpoints", cfg["dataset"]["name"], cfg["task"], date_time))
    Path(checkpoint_path).mkdir(parents=True,exist_ok=True)
    
    #wandb setup

    wandb.init(
    project=cfg['project_name'],
    name=f"{cfg['project_name']}-{cfg['task']}-{date_time}",
    config=cfg)   

    wandb.config.update(get_git_info())
    wandb.config.update({'checkpoint_dir': checkpoint_path,
                         'device' : device})

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    #Training pipeline
    best_val = -1

    print("Start training")


    for epoch in tqdm(range(cfg['max_epochs'])):

        train_loss, train_acc = train_one_epoch(
            model,
            train_dataloader,
            criterion_train,
            optimizer,
            scheduler,
            num_classes=num_classes,
            device=device
        )

        val_loss, val_acc = valid_one_epoch(
            model,
            val_dataloader,
            criterion_test,
            num_classes=num_classes,
            device=device
        )

        wandb.log({
            'epoch' : epoch,
            'train/loss' : train_loss,
            'train/acc' : train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'lr': scheduler.get_last_lr()[0],
        })

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_val_acc": best_val,
                        "cfg": cfg,
                        }, f"{checkpoint_path}/best.pth")
        
        torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "last_val_acc": val_acc,
                    "cfg": cfg,
                    }, f"{checkpoint_path}/last.pth")

        print(f'Epoch {epoch}:')
        print(f'Train_loss : {train_loss}, Train_acc : {train_acc}')
        print(f'Val_loss : {val_loss}, Val_acc : {val_acc}')

    #Test

    #Load best model

    best_checkpoint = torch.load(f"{checkpoint_path}/best.pth", weights_only=False)

    if cfg["model"] == 'adapted_sgformer':
        best_model = AdaptedSGFormer(**cfg['model_params_asgf'])
    elif cfg["model"] == 'AEGT':
        best_model = AEGT(**cfg['model_params_aegt'])

    best_model.load_state_dict(best_checkpoint["model_state_dict"])

    # Test the best model

    test_loss, test_acc = test_model(best_model, test_dataloader, criterion_test, checkpoint_path, num_classes=num_classes, device=device)


    wandb.log({
            # 'epoch' : epoch,
            'test/loss' : test_loss,
            'test/acc' : test_acc,
        })

    wandb.finish()

if __name__ == "__main__":
    main()