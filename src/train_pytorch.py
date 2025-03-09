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
    
    def CW_ATTACK(self, audio, target, c=0.1, num_iter=500, lr=0.01, targeted=True, early_stop=True, kappa=0):
        """
        Perform the Carlini-Wagner (CW) attack.

        Args:
            audio (torch.Tensor): Input audio tensor.
            target (int): Target class index.
            c (float): Trade-off constant for the CW loss.
            num_iter (int): Number of iterations.
            lr (float): Learning rate for the optimizer.
            targeted (bool): Whether the attack is targeted.
            early_stop (bool): Whether to stop early if the attack succeeds.
            kappa (float): Confidence margin. A higher value forces a larger gap between logits (default: 0).

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
                # For a targeted attack, we want the target logit to be higher than all others.
                target_logit = outputs[0, target]
                # Exclude the target class to get the maximum logit among the rest
                mask = torch.ones(outputs.size(), dtype=torch.bool, device=self.device)
                mask[0, target] = False
                other_logits = outputs[mask].view(1, -1)
                max_other_logit, _ = torch.max(other_logits, dim=1)
                # f should be <= 0 when target_logit is larger than every other logit by at least kappa.
                f = torch.clamp(max_other_logit - target_logit, min=-kappa)
            else:
                # For an untargeted attack, we want the true class logit to fall below some other logit.
                target_logit = outputs[0, target]
                mask = torch.ones(outputs.size(), dtype=torch.bool, device=self.device)
                mask[0, target] = False
                other_logits = outputs[mask].view(1, -1)
                max_other_logit, _ = torch.max(other_logits, dim=1)
                f = torch.clamp(target_logit - max_other_logit, min=-kappa)

            # The loss is the L2 norm squared of the perturbation plus c times the CW loss term.
            loss = torch.norm(delta, p=2)**2 + c * f

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Ensure that the perturbation keeps the adversarial example within valid bounds
            delta.data = torch.clamp(delta.data, -1 - audio, 1 - audio)

            # Early stopping if the attack succeeds
            if early_stop:
                with torch.no_grad():
                    adv_audio = torch.clamp(audio + delta, min=-1, max=1)
                    outputs = self.model(adv_audio)
                    _, predicted = torch.max(outputs, 1)
                    if targeted:
                        if predicted.item() == target_tensor.item():
                            print(f"Early stop at iteration {i}")
                            break
                    else:
                        if predicted.item() != target_tensor.item():
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
    learning_rate = 0.001
    epsilon = 0.0015  # Taille maximale de la perturbation
    alpha = 0.00009   # Taille du pas pour chaque itération
    num_iter = 15   # Nombre d'itérations
    adversarial_frequency = 5  # Fréquence de l'entraînement adversarial (toutes les X époques)

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
    model = CNN(batch_norm=True, num_classes=3)  # 12 classes pour les commandes vocales
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
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, epsilon, alpha, num_iter, adversarial_frequency, attack)
    logger.info("Entraînement terminé!")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, epsilon, alpha, num_iter, adversarial_frequency, attack):
    """Fonction d'entraînement du modèle avec phases normales et adversariales."""
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)

            # Alterner entre entraînement normal et adversarial
            if (epoch + 1) % adversarial_frequency == 0:
                # Générer des exemples adversariaux avec BIM uniquement pour les échantillons "down"
                target_labels = []
                for i, label in enumerate(labels):
                    if label.item() == 0:  # Si c'est un "down"
                        # Transformer en "up" (index 1)
                        # inputs[i] = attack.BIM_ATTACK(inputs[i].unsqueeze(0), 1, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
                        inputs[i] = attack.CW_ATTACK(inputs[i].unsqueeze(0), 1)
                        # inputs[i] = attack.MIM_ATTACK(inputs[i].unsqueeze(0), 1)
                logger.info(f"Labels after adversarial transformation: {labels}")


            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total

        # Phase de validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'outputs/checkpoints/best_model.pth')
            logger.info(f'Nouveau meilleur modèle sauvegardé avec une précision de {val_acc:.2f}%')

        # Sauvegarder le modèle à chaque époque
        torch.save(model.state_dict(), f'outputs/checkpoints/model_epoch_{epoch+1}.pth')

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
