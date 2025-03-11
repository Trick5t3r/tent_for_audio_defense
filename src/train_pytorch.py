import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import logging
from tqdm import tqdm
import numpy as np
from model_pytorch import CNN
from pathlib import Path
import torch.nn.functional as F
import random

class ASRAttacks:
    def __init__(self, model, device, labels):
        self.model = model
        self.device = device
        self.labels = labels

    def BIM_ATTACK(self, audio, target, epsilon=0.0015, alpha=0.00009, num_iter=500, targeted=True, early_stop=True):
        """
        Perform the Basic Iterative Method (BIM) attack.

        Args:
            audio (torch.Tensor): Input audio tensor.
            target (int): Target class index.
            epsilon (float): Maximum perturbation.
            alpha (float): Step size for each iteration.
            num_iter (int): Number of iterations.
            targeted (bool): Whether the attack is targeted.
            early_stop (bool): Whether to stop early if the attack succeeds.

        Returns:
            torch.Tensor: Adversarial audio tensor.
        """
        audio = audio.to(self.device)
        original_audio = audio.clone()
        target_tensor = torch.tensor([target], device=self.device)

        for i in range(num_iter):
            audio.requires_grad = True
            outputs = self.model(audio)
            loss = F.cross_entropy(outputs, target_tensor)

            self.model.zero_grad()
            loss.backward()
            audio_grad = audio.grad.data

            if targeted:
                perturbed_audio = audio - alpha * audio_grad.sign()
            else:
                perturbed_audio = audio + alpha * audio_grad.sign()

            perturbation = torch.clamp(perturbed_audio - original_audio, min=-epsilon, max=epsilon)
            audio = torch.clamp(original_audio + perturbation, min=-1, max=1).detach_()

            if early_stop:
                with torch.no_grad():
                    outputs = self.model(audio)
                    _, predicted = torch.max(outputs, 1)
                    if predicted.item() == target_tensor.item():
                        print(f"Early stop at iteration {i}")
                        break

        return audio.cpu()
    
    def CW_ATTACK(self, audio, target, c=0.1, num_iter=500, lr=0.01, targeted=True, early_stop=True, kappa=0, epsilon=0.0015, verbose=False):
        """
        Perform the Carlini-Wagner (CW) attack with an additional perturbation bound.

        Args:
            audio (torch.Tensor): Input audio tensor.
            target (int): Target class index.
            c (float): Trade-off constant for the CW loss.
            num_iter (int): Number of iterations.
            lr (float): Learning rate for the optimizer.
            targeted (bool): Whether the attack is targeted.
            early_stop (bool): Whether to stop early if the attack succeeds.
            kappa (float): Confidence margin.
            epsilon (float): Maximum allowed perturbation (L∞ bound).
            
        Returns:
            torch.Tensor: Adversarial audio tensor.
        """
        # Move audio to the device and keep a copy of the original
        audio = audio.to(self.device)
        original_audio = audio.clone()
        target_tensor = torch.tensor([target], device=self.device)

        # Initialize the perturbation delta as a trainable parameter
        delta = torch.zeros_like(audio, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)

        for i in range(num_iter):
            # Compute the adversarial example and enforce valid audio range
            adv_audio = torch.clamp(audio + delta, min=-1, max=1)
            outputs = self.model(adv_audio)

            # Compute the CW loss f(x+delta)
            if targeted:
                target_logit = outputs[0, target]
                # Exclude the target class to get the maximum logit among the rest
                mask = torch.ones(outputs.size(), dtype=torch.bool, device=self.device)
                mask[0, target] = False
                other_logits = outputs[mask].view(1, -1)
                max_other_logit, _ = torch.max(other_logits, dim=1)
                # f should be <= 0 when target_logit is larger than every other logit by at least kappa.
                f = torch.clamp(max_other_logit - target_logit, min=-kappa)
            else:
                target_logit = outputs[0, target]
                mask = torch.ones(outputs.size(), dtype=torch.bool, device=self.device)
                mask[0, target] = False
                other_logits = outputs[mask].view(1, -1)
                max_other_logit, _ = torch.max(other_logits, dim=1)
                f = torch.clamp(target_logit - max_other_logit, min=-kappa)

            # Loss: L2 norm squared of perturbation plus c times the CW loss term.
            loss = torch.norm(delta, p=2)**2 + c * f

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Project delta onto the epsilon ball (L∞ bound)
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            # Additionally, ensure that audio + delta stays within [-1, 1]
            delta.data = torch.clamp(delta.data, -1 - audio, 1 - audio)

            # Early stopping if the attack succeeds
            if early_stop:
                with torch.no_grad():
                    adv_audio = torch.clamp(audio + delta, min=-1, max=1)
                    outputs = self.model(adv_audio)
                    _, predicted = torch.max(outputs, 1)
                    if targeted:
                        if predicted.item() == target_tensor.item():
                            if verbose:
                                print(f"Early stop at iteration {i}")
                            break
                    else:
                        if predicted.item() != target_tensor.item():
                            if verbose:
                                print(f"Early stop at iteration {i}")
                            break

        adv_audio = torch.clamp(audio + delta, min=-1, max=1)
        return adv_audio.cpu()

    
    def MIM_ATTACK(self, audio, target, epsilon=0.0015, alpha=0.00009, num_iter=500, targeted=True, early_stop=True, decay_factor=1.0):
        """
        Perform the Momentum Iterative Method (MIM) attack.

        Args:
            audio (torch.Tensor): Input audio tensor.
            target (int): Target class index.
            epsilon (float): Maximum perturbation.
            alpha (float): Step size for each iteration.
            num_iter (int): Number of iterations.
            targeted (bool): Whether the attack is targeted.
            early_stop (bool): Whether to stop early if the attack succeeds.
            decay_factor (float): Momentum decay factor (commonly denoted as μ).

        Returns:
            torch.Tensor: Adversarial audio tensor.
        """
        audio = audio.to(self.device)
        original_audio = audio.clone()
        target_tensor = torch.tensor([target], device=self.device)
        # Initialize the momentum term with zeros (same shape as the audio)
        momentum = torch.zeros_like(audio)

        for i in range(num_iter):
            audio.requires_grad = True
            outputs = self.model(audio)
            loss = F.cross_entropy(outputs, target_tensor)

            self.model.zero_grad()
            loss.backward(retain_graph=True)
            audio_grad = audio.grad.data

            # Normalize the gradient by its L1 norm to stabilize updates
            grad_norm = torch.norm(audio_grad.view(audio_grad.shape[0], -1), p=1, dim=1)
            # Reshape for broadcasting and add a small constant to avoid division by zero
            grad_norm = grad_norm.view(-1, *([1] * (len(audio_grad.shape) - 1))) + 1e-8
            normalized_grad = audio_grad / grad_norm

            # Update the momentum: accumulate a fraction of the previous momentum and the current normalized gradient
            momentum = decay_factor * momentum + normalized_grad

            # Use the sign of the momentum for the update step (flip sign based on targeted or untargeted attack)
            if targeted:
                perturbed_audio = audio - alpha * momentum.sign()
            else:
                perturbed_audio = audio + alpha * momentum.sign()

            # Ensure the perturbation remains within the epsilon-ball and the audio stays within valid bounds
            perturbation = torch.clamp(perturbed_audio - original_audio, min=-epsilon, max=epsilon)
            audio = torch.clamp(original_audio + perturbation, min=-1, max=1).detach_()

            if early_stop:
                with torch.no_grad():
                    outputs = self.model(audio)
                    _, predicted = torch.max(outputs, 1)
                    if predicted.item() == target_tensor.item():
                        print(f"Early stop at iteration {i}")
                        break

        return audio.cpu()



    def INFER(self, audio):
        """
        Infer the transcription from the audio.

        Args:
            audio (torch.Tensor): Input audio tensor.

        Returns:
            str: Inferred transcription.
        """
        audio = audio.to(self.device)
        with torch.no_grad():
            outputs = self.model(audio)
            _, predicted = torch.max(outputs, 1)
            transcription = " ".join([self.labels[idx] for idx in predicted])
        return transcription

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Créer les répertoires de sortie
    Path('outputs/checkpoints').mkdir(parents=True, exist_ok=True)

    # Paramètres
    batch_size = 32
    num_epochs = 15
    learning_rate = 0.01
    epsilon = 0.0015  # Taille maximale de la perturbation
    alpha = 0.003   # Taille du pas pour chaque itération
    num_iter = 15   # Nombre d'itérations
    adversarial_frequency = 3  # Fréquence de l'entraînement adversarial (toutes les X époques)
    num_classes = 3

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation du device : {device}")

    # Chemin du dataset
    dataset_path = Path('./Data/speech_commands_v0.01')
    logger.info(f"Chemin du dataset : {dataset_path}")
    logger.info(f"Le dossier existe : {dataset_path.exists()}")

    if not dataset_path.exists():
        raise ValueError(f"Le dossier du dataset n'existe pas : {dataset_path}")

    # Créer les datasets
    logger.info("Création du dataset d'entraînement...")
    train_dataset = SpeechCommandsDataset(dataset_path, subset='train')
    logger.info(f"Nombre d'échantillons d'entraînement : {len(train_dataset)}")

    logger.info("Création du dataset de validation...")
    val_dataset = SpeechCommandsDataset(dataset_path, subset='validation')
    logger.info(f"Nombre d'échantillons de validation : {len(val_dataset)}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Le dataset est vide. Vérifiez le chemin et la structure des dossiers.")

    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Créer le modèle
    model = CNN(batch_norm=True, num_classes=num_classes)  # 12 classes pour les commandes vocales
    model = model.to(device)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Créer l'instance de ASRAttacks
    #labels = ['silence', 'unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

    labels = ['down', 'up','left']
    attack = ASRAttacks(model, device, labels)

    # Entraîner le modèle
    logger.info("Début de l'entraînement...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, epsilon, alpha, num_iter, attack, num_classes)
    logger.info("Entraînement terminé!")

import torch
import random
from tqdm import tqdm
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

import torch
import random
from tqdm import tqdm
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    num_epochs, 
    device, 
    epsilon, 
    alpha, 
    num_iter, 
    attack, 
    num_classes, 
    attack_fraction=0.1, 
    beta=0.3
):
    """
    Train the model using both clean and adversarial examples at every epoch,
    but only generating adversarial attacks for a fraction of samples per batch.
    
    For multi-class classification, for each attacked sample a target is chosen 
    at random from all classes except the true label.
    
    Additionally, a consistency regularisation term (via KL divergence) is applied
    to the attacked samples to encourage similar predictions between clean and adversarial inputs.
    
    Parameters:
    - model: the neural network to be trained.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - criterion: loss function (e.g., nn.CrossEntropyLoss() or with label smoothing).
    - optimizer: optimizer for updating model parameters.
    - num_epochs: number of training epochs.
    - device: device to run training on (e.g., 'cuda' or 'cpu').
    - epsilon, alpha, num_iter: parameters for the adversarial attack.
    - attack: adversarial attack module (with methods such as CW_ATTACK).
    - num_classes: total number of classes in the dataset.
    - attack_fraction: fraction of samples in each batch to be attacked (0 <= attack_fraction <= 1).
    - beta: weight for the adversarial loss term.
    """

    best_val_acc = 0.0
    
    # Lists to store metrics per epoch
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # If you have a separate measure for "Smoothing Clean Accuracy":
    smoothing_accuracies = []
    
    # If you want to track robust accuracy (accuracy on adversarial examples):
    robust_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        # ------------------
        #   TRAINING LOOP
        # ------------------
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            adv_inputs = inputs.clone()
            
            # Keep track of indices for which an attack was applied
            attacked_indices = []
            
            # Decide which samples to attack
            for i, label in enumerate(labels):
                if random.random() < attack_fraction:
                    # Choose a random target that's not the true label
                    possible_targets = [c for c in range(num_classes) if c != label.item()]
                    target = random.choice(possible_targets)
                    adv_example = attack.CW_ATTACK(adv_inputs[i].unsqueeze(0), target)
                    adv_inputs[i] = adv_example.squeeze(0)
                    attacked_indices.append(i)
            
            """logger.info(
                f"Batch labels: {labels.cpu().numpy()}. "
                f"Number of attacked samples: {len(attacked_indices)}"
            )"""

            optimizer.zero_grad()

            # Forward pass on clean inputs
            outputs_clean = model(inputs)
            loss_clean = criterion(outputs_clean, labels)

            # Forward pass on adversarial inputs (only for attacked samples)
            outputs_adv = model(adv_inputs)
            loss_adv = 0.0
            if attacked_indices:
                attacked_indices_tensor = torch.tensor(attacked_indices, device=device)
                loss_adv = criterion(
                    outputs_adv[attacked_indices_tensor], 
                    labels[attacked_indices_tensor]
                )

            total_loss = loss_clean + beta * loss_adv
            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()
            _, predicted = outputs_clean.max(1)
            epoch_train_total += labels.size(0)
            epoch_train_correct += predicted.eq(labels).sum().item()

        # Compute mean training loss and accuracy
        train_loss_epoch = epoch_train_loss / len(train_loader)
        train_acc_epoch = 100.0 * epoch_train_correct / epoch_train_total

        train_losses.append(train_loss_epoch)
        train_accuracies.append(train_acc_epoch)

        # ------------------
        #   VALIDATION LOOP
        # ------------------
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                _, predicted = outputs.max(1)
                epoch_val_total += labels.size(0)
                epoch_val_correct += predicted.eq(labels).sum().item()

        val_loss_epoch = epoch_val_loss / len(val_loader)
        val_acc_epoch = 100.0 * epoch_val_correct / epoch_val_total

        val_losses.append(val_loss_epoch)
        val_accuracies.append(val_acc_epoch)

        # ------------------------------
        #   OPTIONAL: SMOOTHING ACCURACY
        # ------------------------------
        # If you have a separate measure for "smoothing clean accuracy," compute it here.
        # We'll just set it to val_acc_epoch as a placeholder:
        smoothing_acc_epoch = val_acc_epoch
        smoothing_accuracies.append(smoothing_acc_epoch)

        # ------------------------------
        #   OPTIONAL: ROBUST ACCURACY
        # ------------------------------
        val_correct_adv = 0
        val_total_adv = 0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Generate adversarial examples WITHOUT torch.no_grad()
            adv_inputs = []
            for i, label in enumerate(labels):
                possible_targets = [c for c in range(num_classes) if c != label.item()]
                target = random.choice(possible_targets)
                adv_example = attack.CW_ATTACK(inputs[i].unsqueeze(0), target)
                adv_inputs.append(adv_example.squeeze(0))
            
            adv_inputs = torch.stack(adv_inputs).to(device)
            
            # Now, use torch.no_grad() for the model inference on adversarial examples.
            with torch.no_grad():
                outputs_adv = model(adv_inputs)
                _, predicted_adv = outputs_adv.max(1)
            
            val_total_adv += labels.size(0)
            val_correct_adv += predicted_adv.eq(labels).sum().item()

        val_acc_robust_epoch = 100.0 * val_correct_adv / val_total_adv
        robust_accuracies.append(val_acc_robust_epoch)


        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.2f}%')
        logger.info(f'  Val   Loss: {val_loss_epoch:.4f},   Val Acc: {val_acc_epoch:.2f}%')
        logger.info(f'  Robust Val Acc: {val_acc_robust_epoch:.2f}%')

        # ----------------------
        #  SAVE BEST MODEL, ETC.
        # ----------------------
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            torch.save(model.state_dict(), 'outputs/checkpoints/best_model.pth')
            logger.info(f'New best model saved with accuracy {val_acc_epoch:.2f}%')

        torch.save(model.state_dict(), f'outputs/checkpoints/model_epoch_{epoch+1}.pth')

        # -------------------------------------------------
        #  PLOT METRICS (UP TO CURRENT EPOCH) & SAVE TO PNG
        # -------------------------------------------------
        fig = make_subplots(
            rows=2, 
            cols=2, 
            subplot_titles=(
                "Clean Loss", 
                "Clean Accuracy", 
                "Smoothing Clean Accuracy", 
                "Robust Accuracy"
            )
        )

        epochs_range = list(range(1, epoch + 2))  # up to current epoch

        # 1) Clean Loss
        fig.add_trace(
            go.Scatter(
                x=epochs_range, 
                y=train_losses, 
                mode='lines+markers', 
                name='Clean Loss'
            ), 
            row=1, col=1
        )

        # 2) Clean Accuracy
        fig.add_trace(
            go.Scatter(
                x=epochs_range, 
                y=val_accuracies, 
                mode='lines+markers', 
                name='Clean Accuracy'
            ), 
            row=1, col=2
        )

        # 3) Smoothing Clean Accuracy
        fig.add_trace(
            go.Scatter(
                x=epochs_range, 
                y=smoothing_accuracies, 
                mode='lines+markers', 
                name='Smoothing Clean Acc'
            ), 
            row=2, col=1
        )

        # 4) Robust Accuracy
        fig.add_trace(
            go.Scatter(
                x=epochs_range, 
                y=robust_accuracies, 
                mode='lines+markers', 
                name='Robust Accuracy'
            ), 
            row=2, col=2
        )

        fig.update_layout(
            title="Évolution des métriques d'entraînement (up to Epoch {})".format(epoch+1),
            height=800, 
            width=1000
        )

        # Save the figure for this epoch
        fig.write_image(f"outputs/training_metrics_epoch_{epoch+1}.png")
        logger.info(f"Saved training metrics figure to outputs/training_metrics_epoch_{epoch+1}.png")
    
    # (Optional) You could also create a final plot after all epochs if desired.
    # For example, to see the entire training in one figure:
    # final_fig = make_subplots(...)
    # ... add traces ...
    # final_fig.write_image("outputs/training_metrics_final.png")



class SpeechCommandsDataset(Dataset):
    """Dataset pour Speech Commands."""
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.transform = transform

        # Liste des commandes (labels)
        #self.commands = ['silence', 'unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        self.commands = ['down', 'up','left']
        # Charger les fichiers audio et leurs labels
        self.files = []
        self.labels = []

        # Parcourir le répertoire
        for command in self.commands:
            command_dir = self.root_dir / command
            if command_dir.exists():
                logger.info(f"Traitement du dossier {command}...")
                files = list(command_dir.glob('*.wav'))
                logger.info(f"Nombre de fichiers trouvés pour {command}: {len(files)}")

                # Séparer les fichiers en train et validation
                if subset == 'train':
                    # Utiliser 80% des fichiers pour l'entraînement
                    files = files[:int(0.8 * len(files))]
                else:
                    # Utiliser 20% des fichiers pour la validation
                    files = files[int(0.8 * len(files)):]

                self.files.extend([str(f) for f in files])
                self.labels.extend([self.commands.index(command)] * len(files))
            else:
                logger.warning(f"Dossier non trouvé : {command_dir}")

        logger.info(f"Dataset {subset} chargé : {len(self.files)} fichiers")
        if len(self.files) == 0:
            logger.error(f"Aucun fichier trouvé dans {self.root_dir}")
            logger.error("Structure des dossiers trouvée :")
            for item in self.root_dir.iterdir():
                logger.error(f"- {item}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Charger l'audio
        audio_path = self.files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Normaliser la durée à 1 seconde (16000 échantillons)
        target_length = 16000
        if waveform.shape[1] > target_length:
            # Si l'audio est trop long, on le tronque
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            # Si l'audio est trop court, on le remplit avec des zéros
            pad_length = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # Retourner l'audio et le label
        return waveform, self.labels[idx]

if __name__ == "__main__":
    main()
