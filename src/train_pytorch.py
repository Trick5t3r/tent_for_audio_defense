#! /usr/bin/env python3
"""
Détecteur de mots-clés audio basé sur CNN en PyTorch.

Le dataset consiste en ~2,000 enregistrements d'une seconde par mot-clé.
Il y a 30 mots-clés au total. Dataset disponible ici:
https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

Les fichiers audio .wav sont convertis en spectrogramme log-mel à 30 bins
avec des trames de 32ms et un chevauchement de 16ms. Le spectrogramme résultant
est une "image" de 61x30. Les spectrogrammes sont passés dans un CNN, avec
un pooling global moyen comme couche de sortie finale.

Voir tfrecords.py pour la préparation du dataset.

Mise à jour: Ajout de la possibilité de prendre l'audio encodé avec un réseau
entraîné avec Contrastive Predictive Coding. Créer le dataset avec cpc_tfrecords.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm

from dataset import build_dataset, TensorFlowDataset
from model_pytorch import CNN, TallKernel
from callbacks import build_callbacks

def add_logpaths(config):
    """Ajoute les chemins pour les checkpoints et les données TensorBoard au dictionnaire de configuration."""
    config['ckpt_path'] = str(Path.cwd() / 'outputs' / 'checkpoints')
    config['tblog_path'] = str(Path.cwd() / 'logs')
    Path(config['ckpt_path']).mkdir(parents=True, exist_ok=True)

def save_state(config):
    """Sauvegarde la configuration de l'entraînement: tous les fichiers *.py et le dictionnaire de configuration."""
    filenames = Path.cwd().glob('*.py')
    for filename in filenames:
        shutil.copy(str(filename), config['ckpt_path'])
    with (Path(config['ckpt_path']) / 'config.json').open('w') as f:
        json.dump(config, f)

def init_azure_logging(config):
    """Obtient le logger Azure et enregistre tous les paramètres de configuration."""
    from azureml.core.run import Run
    run_logger = Run.get_context()
    for k in config:
        run_logger.log(k, config[k])
    return run_logger

def get_optimizer(model, config):
    """Crée l'optimiseur PyTorch selon la configuration."""
    if config['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=config['lr'])
    if config['optimizer'] == 'sgd':
        return optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîne le modèle pendant une époque."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Valide le modèle sur l'ensemble de validation."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

def main(
    lr=0.01,
    momentum=0.9,
    epochs=25,
    batch_sz=64,
    data_dir=str(Path("..") / 'Data' / 'speech_commands' / 'tfrecords'),
    ds_type='log-mel',
    optimizer='adam',
    batch_norm=1,
    tall_kernel=False,
    dense_final=0,
    load_genc=0,
    train_genc=0,
    verbose=2
):
    # Configuration du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Création du modèle
    if tall_kernel:
        model = TallKernel(
            batch_norm=batch_norm,
            dense_final=dense_final,
            load_genc=load_genc,
            train_genc=train_genc,
            ds_type=ds_type
        )
    else:
        model = CNN(
            batch_norm=batch_norm,
            dense_final=dense_final,
            load_genc=load_genc,
            train_genc=train_genc,
            ds_type=ds_type
        )
    model = model.to(device)
    
    # Création des dataloaders
    train_dataset = TensorFlowDataset(build_dataset({'data_dir': data_dir, 'ds_type': ds_type, 'batch_sz': batch_sz}, 'train'))
    val_dataset = TensorFlowDataset(build_dataset({'data_dir': data_dir, 'ds_type': ds_type, 'batch_sz': batch_sz}, 'val'))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz,
                          shuffle=False, num_workers=0)
    
    # Configuration de l'entraînement
    criterion = nn.CrossEntropyLoss()
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # Configuration des chemins et sauvegarde de l'état
    ckpt_path = str(Path.cwd() / 'outputs' / 'checkpoints')
    tblog_path = str(Path.cwd() / 'logs')
    Path(ckpt_path).mkdir(parents=True, exist_ok=True)
    
    # Configuration de TensorBoard
    writer = SummaryWriter(tblog_path)
    
    # Boucle d'entraînement
    best_val_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Mise à jour du learning rate
        scheduler.step(val_loss)
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        if verbose > 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 
                      Path(ckpt_path) / 'best_model.pth')
        
        # Sauvegarde périodique
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                      Path(ckpt_path) / f'model_epoch_{epoch+1}.pth')
    
    writer.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Entraîner le modèle.')
    parser.add_argument('-lr', '--lr', type=float, default=0.01)
    parser.add_argument('-m', '--momentum', type=float, default=0.9)
    parser.add_argument('-e', '--epochs', type=int, default=25)
    parser.add_argument('-b', '--batch_sz', type=int, default=64)
    parser.add_argument('--data_dir', type=str, 
                       default=str(Path("..") / 'Data' / 'speech_commands'))
    parser.add_argument('-ds', '--ds_type', type=str, default='log-mel')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('-bn', '--batch_norm', type=int, default=1)
    parser.add_argument('-tk', '--tall_kernel', type=int, default=0)
    parser.add_argument('-df', '--dense_final', type=int, default=0)
    parser.add_argument('--load_genc', type=int, default=0)
    parser.add_argument('--train_genc', type=int, default=0)
    parser.add_argument('-v', '--verbose', type=int, default=2)
    
    args = parser.parse_args()
    main(**vars(args)) 