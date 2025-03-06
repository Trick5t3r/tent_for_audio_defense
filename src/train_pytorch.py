import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torchaudio
import torch.nn.functional as F

from model_pytorch import CNN
from callbacks import build_callbacks  # si nécessaire
import torch.nn.functional as F

def pad_collate(batch):
    # Sépare les waveforms et les labels
    waveforms, labels = zip(*batch)
    # Trouve la longueur maximale dans le batch
    max_length = max(waveform.shape[1] for waveform in waveforms)
    padded_waveforms = []
    for waveform in waveforms:
        pad_amount = max_length - waveform.shape[1]
        # Si besoin, on padde à droite
        if pad_amount > 0:
            waveform = F.pad(waveform, (0, pad_amount))
        padded_waveforms.append(waveform)
    # Empile les waveforms et transforme les labels en tensor
    return torch.stack(padded_waveforms, 0), torch.tensor(labels)


# Nouveau dataset pour charger les fichiers .wav
class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, ds_type='log-mel', transform=None):
        """
        Args:
            root_dir (str): chemin vers le dossier contenant les sous-dossiers par catégorie.
            ds_type (str): type de données à retourner ('samples' pour les échantillons bruts, 'log-mel' pour spectrogrammes).
            transform (callable, optionnel): transformation à appliquer sur le signal audio.
        """
        self.root_dir = root_dir
        print(root_dir)
        self.ds_type = ds_type
        self.transform = transform

        self.samples = []
        # Liste triée des catégories (sous-dossiers)
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # Parcours de chaque catégorie pour récupérer les fichiers .wav
        for c in self.classes:
            class_dir = os.path.join(root_dir, c)
            for file in os.listdir(class_dir):
                if file.lower().endswith('.wav'):
                    self.samples.append((os.path.join(class_dir, file), self.class_to_idx[c]))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        waveform, sample_rate = torchaudio.load(path)
        # On conserve un seul canal si besoin
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]
        
        if self.ds_type == 'samples':
            # Retourne directement l'échantillon brut
            return waveform, label
        else:
            # Transformation par défaut : conversion en spectrogramme log-mel
            if self.transform:
                spec = self.transform(waveform)
            else:
                win_length = int(0.032 * sample_rate)
                hop_length = int(0.016 * sample_rate)
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_mels=30,
                    win_length=win_length,
                    hop_length=hop_length
                )
                mel_spec = mel_transform(waveform)
                spec = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10, amin=1e-10, db_multiplier=0)
            # Ajustement à 61 frames
            if spec.shape[2] < 61:
                pad_amount = 61 - spec.shape[2]
                spec = F.pad(spec, (0, pad_amount))
            elif spec.shape[2] > 61:
                spec = spec[:, :, :61]
            return spec, label

def add_logpaths(config):
    """Ajoute les chemins pour les checkpoints et TensorBoard dans le dictionnaire de configuration."""
    config['ckpt_path'] = str(Path.cwd() / 'outputs' / 'checkpoints')
    config['tblog_path'] = str(Path.cwd() / 'logs')
    Path(config['ckpt_path']).mkdir(parents=True, exist_ok=True)

def save_state(config):
    """Sauvegarde la configuration de l'entraînement (fichiers *.py et config.json)."""
    filenames = Path.cwd().glob('*.py')
    for filename in filenames:
        shutil.copy(str(filename), config['ckpt_path'])
    with (Path(config['ckpt_path']) / 'config.json').open('w') as f:
        json.dump(config, f)

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
    data_dir=str(Path("speech_commands")),  # dossier racine contenant les catégories
    optimizer='adam',
    batch_norm=1,
    ds_type='samples',
    dense_final=0,
    load_genc=0,
    train_genc=0,
    verbose=2
):
    # Configuration du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Création du modèle (ici, on utilise toujours CNN)
    model = CNN(
        batch_norm=batch_norm,
        dense_final=dense_final,
        load_genc=load_genc,
        train_genc=train_genc,
        ds_type=ds_type
    )
    model = model.to(device)
    
    # Création du dataset en fonction du type de données souhaité
    dataset = SpeechCommandsDataset(root_dir=data_dir, ds_type=ds_type)
    # Séparation aléatoire : 80% pour l'entraînement, 20% pour la validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=0, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False, num_workers=0, collate_fn=pad_collate)

    
    # Configuration de l'entraînement
    criterion = nn.CrossEntropyLoss()
    if optimizer == 'adam':
        optim_obj = optim.Adam(model.parameters(), lr=lr)
    else:
        optim_obj = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_obj, 'min', patience=3)
    
    # Chemins de sauvegarde et TensorBoard
    ckpt_path = str(Path.cwd() / 'outputs' / 'checkpoints')
    tblog_path = str(Path.cwd() / 'logs')
    Path(ckpt_path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tblog_path)
    
    # Boucle d'entraînement
    best_val_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optim_obj, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        if verbose > 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Sauvegarde du meilleur modèle et sauvegardes périodiques
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(ckpt_path) / 'best_model.pth')
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), Path(ckpt_path) / f'model_epoch_{epoch+1}.pth')
    
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner le modèle sur des fichiers WAV (données brutes ou spectrogrammes).")
    parser.add_argument('-lr', '--lr', type=float, default=0.01)
    parser.add_argument('-m', '--momentum', type=float, default=0.9)
    parser.add_argument('-e', '--epochs', type=int, default=25)
    parser.add_argument('-b', '--batch_sz', type=int, default=64)
    parser.add_argument('--data_dir', type=str, default=str(Path("..") / "Data" / "speech_commands"))
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('-bn', '--batch_norm', type=int, default=1)
    parser.add_argument('--ds_type', type=str, default='samples', 
                        help="Type de données : 'samples' pour les échantillons bruts, 'log-mel' pour spectrogrammes")
    parser.add_argument('-df', '--dense_final', type=int, default=0)
    parser.add_argument('--load_genc', type=int, default=0)
    parser.add_argument('--train_genc', type=int, default=0)
    parser.add_argument('-v', '--verbose', type=int, default=2)
    
    args = parser.parse_args()
    main(**vars(args))
