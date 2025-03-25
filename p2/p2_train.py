import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime
import config as cfg
from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import set_seed, write_config_log, write_result_log

def plot_learning_curve(logfile_dir: str,result_lists: list):
    epochs = len(result_lists['train_acc'])
    x_range = range(1, epochs + 1)  

    # Training Accuracy
    plt.figure()
    plt.plot(x_range, result_lists['train_acc'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(logfile_dir, 'train_acc.png'))
    plt.close()

    # Training Loss
    plt.figure()
    plt.plot(x_range, result_lists['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(logfile_dir, 'train_loss.png'))
    plt.close()

    # Validation Accuracy
    plt.figure()
    plt.plot(x_range, result_lists['val_acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(logfile_dir, 'val_acc.png'))
    plt.close()

    # Validation Loss
    plt.figure()
    plt.plot(x_range, result_lists['val_loss'])
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(logfile_dir, 'val_loss.png'))
    plt.close()

def pseudo_label_unlabeled_data(model, unlabel_loader, device, threshold=0.9):
    model.eval()
    pseudo_images = []
    pseudo_labels = []
    with torch.no_grad():
        for batch in unlabel_loader:
            images = batch[0]
            outputs = model(images)  # shape (B, 10)
            probs = torch.softmax(outputs, dim=1) 
            conf, pred_cls = torch.max(probs, dim=1)  # conf: (B), pred_cls: (B)
            mask = (conf > threshold)
            selected_imgs = images[mask]          # shape (N_selected, 3, 32,32)
            selected_labels = pred_cls[mask]      # shape (N_selected,)
            if len(selected_imgs) > 0:
                pseudo_images.append(selected_imgs.cpu())
                pseudo_labels.append(selected_labels.cpu())
    if len(pseudo_images) > 0:
        pseudo_images = torch.cat(pseudo_images, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
    else:
        pseudo_images = torch.empty(0, 3, 32, 32)
        pseudo_labels = torch.empty(0, dtype=torch.long)
    return pseudo_images, pseudo_labels

def train(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        logfile_dir: str,
        model_save_dir: str,
        criterion: nn.Module,
        optimizer: torch.optim,
        scheduler: torch.optim,
        device: torch.device
    ):
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        ##### TRAINING #####
        train_start_time = time.time()
        train_loss = 0.0
        train_correct = 0.0
        model.train()
        for batch, data in enumerate(train_loader):
            sys.stdout.write(f'\r[{epoch + 1}/{cfg.epochs}] Train batch: {batch + 1} / {len(train_loader)}')
            sys.stdout.flush()
            # Data loading. (batch_size, 3, 32, 32), (batch_size)
            images, labels = data
            # Forward pass. input: (batch_size, 3, 32, 32), output: (batch_size, 10)
            pred = model(images)
            # Calculate loss.
            loss = criterion(pred, labels)
            # Backprop. (update model parameters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Evaluate.
            train_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
            train_loss += loss.item()
        # Print training result
        train_time = time.time() - train_start_time
        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader)
        train_acc_list.append(train_acc.cpu().numpy())
        train_loss_list.append(train_loss)
        print()
        print(f'[{epoch + 1}/{cfg.epochs}] {train_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Train Loss: {train_loss:.5f}')

        ##### VALIDATION #####
        model.eval()
        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0.0
            val_correct = 0.0
            for batch, data in enumerate(val_loader):
                sys.stdout.write(f'\r[{epoch + 1}/{cfg.epochs}] Val batch: {batch + 1} / {len(val_loader)}')
                sys.stdout.flush()
                images, labels = data
                pred = model(images)
                loss = criterion(pred, labels)
                val_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
                val_loss += loss.item()

        # Print validation result
        val_time = time.time() - val_start_time
        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc.cpu().numpy())
        val_loss_list.append(val_loss)
        print()
        print(f'[{epoch + 1}/{cfg.epochs}] {val_time:.2f} sec(s) Val Acc: {val_acc:.5f} | Val Loss: {val_loss:.5f}')
        
        # Scheduler step
        scheduler.step()

        ##### WRITE LOG #####
        is_better = val_acc >= best_acc
        epoch_time = train_time + val_time
        write_result_log(os.path.join(logfile_dir, 'result_log.txt'),
                         epoch, epoch_time,
                         train_acc, val_acc,
                         train_loss, val_loss,
                         is_better)

        ##### SAVE THE BEST MODEL #####
        if is_better:
            print(f'[{epoch + 1}/{cfg.epochs}] Save best model to {model_save_dir} ...')
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir, 'model_best.pth'))
            best_acc = val_acc

        ##### PLOT LEARNING CURVE #####
        current_result_lists = {
            'train_acc': train_acc_list,
            'train_loss': train_loss_list,
            'val_acc': val_acc_list,
            'val_loss': val_loss_list
        }
        plot_learning_curve(logfile_dir, current_result_lists)

from torch.utils.data import ConcatDataset, TensorDataset

def create_semi_supervised_dataloader(train_loader, pseudo_images, pseudo_labels):
    original_dataset = train_loader.dataset
    pseudo_dataset = TensorDataset(pseudo_images, pseudo_labels)
    combined_dataset = ConcatDataset([original_dataset, pseudo_dataset])
    new_train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=cfg.batch_size,   
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    return new_train_loader

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', 
                        help='dataset directory', 
                        type=str, 
                        default='../hw2_data/p2_data/')
    parser.add_argument('--use_semi',
                        action='store_true',
                        help='whether to use semi-supervised or not')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    use_semi = args.use_semi
    # Experiment name
    exp_name = cfg.model_type \
        + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S') \
        + '_' + cfg.exp_name

    # Write log file for config
    logfile_dir = os.path.join('./experiment', exp_name, 'log')
    os.makedirs(logfile_dir, exist_ok=True)
    write_config_log(os.path.join(logfile_dir, 'config_log.txt'))

    # Fix a random seed for reproducibility
    set_seed(4096)

    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ##### MODEL #####
    model_save_dir = os.path.join('./experiment', exp_name, 'model')
    os.makedirs(model_save_dir, exist_ok=True)

    if cfg.model_type == 'mynet':
        model = MyNet()
    elif cfg.model_type == 'resnet18':
        model = ResNet18()
    else:
        raise NameError('Unknown model type')

    model.to(device)

    ##### DATALOADER #####
    train_loader = get_dataloader(os.path.join(dataset_dir, 'train'),
                                  batch_size=cfg.batch_size, split='train')
    val_loader   = get_dataloader(os.path.join(dataset_dir, 'val'),
                                  batch_size=cfg.batch_size, split='val')

    ##### LOSS & OPTIMIZER #####
    criterion = nn.CrossEntropyLoss()
    if cfg.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=3e-4) # weight_decay to prevent overfitting
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                    momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg.milestones,
                                                     gamma=0.1)
    
    ##### TRAINING & VALIDATION #####
    print("==== Stage 1: Train on labeled data only ====")
    best_acc = 0.0
    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          logfile_dir=logfile_dir,
          model_save_dir=model_save_dir,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device)
    
    if use_semi:
        print("==== Stage 2: Pseudo-label unlabeled data ====")
        unlabel_loader = get_dataloader(os.path.join(dataset_dir, 'unlabel'),
                                        batch_size=cfg.batch_size,
                                        split='unlabel')
        pseudo_images, pseudo_labels = pseudo_label_unlabeled_data(
            model, unlabel_loader, device, threshold=0.9
        )
        print(f"Pseudo-labeled images: {len(pseudo_images)} / {len(unlabel_loader.dataset)}")

        if len(pseudo_images) > 0:
            new_train_loader = create_semi_supervised_dataloader(
                train_loader, pseudo_images, pseudo_labels
            )
            extra_epochs = 10
            print("==== Stage 3: Fine-tune with pseudo-labeled data (extra epochs) ====")
            for extra_epoch in range(extra_epochs):
                train_start_time = time.time()
                train_loss = 0.0
                train_correct = 0.0
                model.train()
                for batch, data in enumerate(new_train_loader):
                    images, labels = data
                    pred = model(images)
                    loss = criterion(pred, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
                    train_loss += loss.item()
                train_time = time.time() - train_start_time
                avg_train_loss = train_loss / len(new_train_loader)
                train_acc = train_correct / len(new_train_loader.dataset)
                model.eval()
                val_loss = 0.0
                val_correct = 0.0
                with torch.no_grad():
                    for data in val_loader:
                        images, labels = data
                        pred = model(images)
                        loss = criterion(pred, labels)
                        val_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
                        val_loss += loss.item()
                val_acc = val_correct / len(val_loader.dataset)
                avg_val_loss = val_loss / len(val_loader)
                scheduler.step()
                print(f"[FineTune E{extra_epoch+1}/{extra_epochs}] "
                      f"TrainAcc={train_acc:.4f}, TrainLoss={avg_train_loss:.4f} | "
                      f"ValAcc={val_acc:.4f}, ValLoss={avg_val_loss:.4f}")
                is_better = val_acc >= best_acc
                if is_better:
                    print(f'Save best model to {model_save_dir} ...')
                    torch.save(model.state_dict(),
                            os.path.join(model_save_dir, 'model_best.pth'))
                    best_acc = val_acc

    print("==== All training stages done ====")

if __name__ == '__main__':
    main()
