import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchaudio
import torch
from dent.dent import Dent

from dent.conf import cfg
import logging
from model_pytorch import CNN

logger = logging.getLogger(__name__)

import torch
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
            target (list): Target transcription.
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
        target_tensor = torch.tensor([self.labels.index(t) for t in target], device=self.device)

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


def convert_to_log_mel(audio, sample_rate=16000):
    """Convertit l'audio en spectrogramme log-mel."""
    # Paramètres pour le spectrogramme mel
    n_fft = 1024
    hop_length = 512
    n_mels = 40
    f_min = 0
    f_max = sample_rate / 2
    
    # Créer le transformateur mel
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max
    )
    
    # Convertir en spectrogramme mel
    mel_spec = mel_transform(audio)
    
    # Convertir en log-mel
    log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    # Normaliser comme dans le dataset
    log_mel = torch.clamp(log_mel, -10.0, 5.0)
    log_mel = (log_mel + 2.5) / 7.5
    
    # Ajouter la dimension du canal
    log_mel = log_mel.unsqueeze(0)
    
    return log_mel

def evaluate(input_audio, target, dent_on=False, num_iter=500):
    """
    Évalue la défense TENT sur un fichier audio
    Args:
        input_audio: Chemin vers le fichier audio ou tensor audio
        target: Transcription cible
        dent_on: Si True, active la défense TENT
        num_iter: Nombre d'itérations pour l'attaque
    """
    # Configuration par défaut
    cfg.merge_from_file("dent/cfgs/dent.yaml")
    cfg.merge_from_list([
        "TEST.BATCH_SIZE", "128",
        "MODEL.ADAPTATION", "dent",
        "MODEL.ARCH", "Ding2020MMA",
        "MODEL.EPISODIC", "True",
        "OPTIM.METHOD", "AdaMod",
        "OPTIM.STEPS", "6",
        "OPTIM.BETA", "0.99",
        "OPTIM.BETA3", "0.6",
        "OPTIM.LR", "6e-3",
        "OPTIM.WD", "0.",
        "OPTIM.LOSS", "shot",
        "OPTIM.BN_FUN", "SampleAwareOnlineBatchNorm2d"
    ])
    cfg.freeze()

    # Load the model
    logger.info("Chargement du modèle...")
    model = CNN(batch_norm=True, num_classes=3)
    chemin_du_modele = 'outputs/checkpoints/model_epoch_30.pth'
    model.load_state_dict(torch.load(chemin_du_modele))
    logger.info("Modèle chargé avec succès")
    
    # Checking the device available during the current environment (CUDA is recommended!)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation du device : {device}")
    model = model.to(device)
    
    # S'assurer que le modèle est en mode évaluation
    model.eval()
    labels = ['down', 'up','left']
    
    if dent_on:
        logger.info("Configuration du modèle TENT...")
        dent_model = Dent(model, cfg.OPTIM)
        logger.info(f"Modèle TENT configuré avec : {cfg.OPTIM.METHOD}")
        logger.info(f"Paramètres d'optimisation : LR={cfg.OPTIM.LR}, BETA={cfg.OPTIM.BETA}, STEPS={cfg.OPTIM.STEPS}")
        
        params = [p for p in dent_model.parameters() if p.requires_grad]
        logger.info(f"Nombre de paramètres à adapter : {len(params)}")

        attack = ASRAttacks(dent_model, device, labels)
    else:
        attack = ASRAttacks(model, device, labels)

    # Charger l'audio si c'est un chemin de fichier
    logger.info("Chargement de l'audio...")
    if isinstance(input_audio, str):
        audio, sample_rate = torchaudio.load(input_audio)
        logger.info(f"Audio chargé depuis {input_audio}")
    else:
        audio = input_audio
        sample_rate = 16000

    # Normaliser la durée à 1 seconde (16000 échantillons)
    target_length = 16000
    if audio.shape[1] > target_length:
        audio = audio[:, :target_length]
    elif audio.shape[1] < target_length:
        pad_length = target_length - audio.shape[1]
        audio = torch.nn.functional.pad(audio, (0, pad_length))
    
    # Ajouter la dimension du batch si nécessaire
    if len(audio.shape) == 2:
        audio = audio.unsqueeze(0)  # [1, channels, time]
    
    # Convertir la cible en format approprié
    target = target.lower()  # Convertir en minuscules
    if target not in labels:
        raise ValueError(f"La cible '{target}' n'est pas dans la liste des labels valides : {labels}")
    
    logger.info(f"Cible : {target}")
    logger.info(f"Shape de l'audio : {audio.shape}")

    try:
        # Effectuer l'attaque
        logger.info("Début de l'attaque BIM...")
        logger.info(f"Paramètres de l'attaque : epsilon=0.0015, alpha=0.00009, num_iter={num_iter}")
        res = attack.BIM_ATTACK(audio, [target], epsilon=0.0015, alpha=0.00009, 
                              num_iter=num_iter, targeted=True, early_stop=True)
        logger.info("Attaque BIM terminée")

        if res is None:
            raise ValueError("L'attaque n'a pas réussi à générer un résultat")

        # Sauvegarder le résultat
        output_filename = f"./output/audio_files/result_tent_{'on' if dent_on else 'off'}.wav"
        if torch.is_tensor(res):
            res = res.detach().cpu()
        
        # S'assurer que l'audio est au bon format [channels, samples]
        if len(res.shape) == 3:
            res = res.squeeze(1)  # Supprimer la dimension du milieu
        if len(res.shape) == 2 and res.shape[0] == 1:
            res = res.squeeze(0)  # Supprimer la dimension du canal si mono
            
        torchaudio.save(output_filename, res.unsqueeze(0), sample_rate)

        # Obtenir la transcription
        # Ajouter les dimensions nécessaires pour l'inférence [batch, channels, time]
        audio_for_inference = res.unsqueeze(0).unsqueeze(0)  # [1, 1, time]
        transcription = attack.INFER(audio_for_inference).replace("|", " ")
        
        logger.info(f"Fichier audio sauvegardé dans : {output_filename}")
        logger.info(f"Transcription obtenue : {transcription}")
        
        return res, transcription

    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation : {str(e)}")
        logger.error(f"Type de l'erreur : {type(e)}")
        import traceback
        logger.error(f"Traceback complet : {traceback.format_exc()}")
        return None, None

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(level=logging.INFO)
    
    # Exemple d'utilisation
    test_file = "down.wav"
    target_transcription = "up"
    
    print("Test sans DENT...")
    res1, trans1 = evaluate(test_file, target_transcription, dent_on=False)
    if res1 is not None:
        print(f"Transcription sans DENT : {trans1}")
    
    print("\nTest avec DENT...")
    res2, trans2 = evaluate(test_file, target_transcription, dent_on=True)
    if res2 is not None:
        print(f"Transcription avec DENT : {trans2}")
    