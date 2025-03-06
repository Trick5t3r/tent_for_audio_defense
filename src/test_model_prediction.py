import torchaudio
import torch
import logging
from model_pytorch import CNN
from dent.dent import Dent
from conf import cfg

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_audio(audio_path, tent_on=False):
    """
    Prédit la classe d'un fichier audio en utilisant le modèle CNN.
    
    Args:
        audio_path (str): Chemin vers le fichier audio
        tent_on (bool): Si True, utilise la défense TENT
        
    Returns:
        str: Classe prédite
    """
    # Labels possibles
    labels = ['down', 'up','left']
    
    # Configuration par défaut
    cfg.merge_from_file("tent_for_audio_defense/configs/dent.yaml")
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
    
    # Charger le modèle
    logger.info("Chargement du modèle...")
    model = CNN(batch_norm=True, num_classes=3)
    model.load_state_dict(torch.load('outputs/checkpoints/best_model.pth'))
    
    if tent_on:
        logger.info("Configuration du modèle TENT...")
        model = Dent(model, cfg.OPTIM)
        logger.info(f"Modèle TENT configuré avec : {cfg.OPTIM.METHOD}")
        logger.info(f"Paramètres d'optimisation : LR={cfg.OPTIM.LR}, BETA={cfg.OPTIM.BETA}, STEPS={cfg.OPTIM.STEPS}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation du device : {device}")
    model = model.to(device)
    model.eval()
    
    # Charger l'audio
    logger.info(f"Chargement de l'audio depuis {audio_path}...")
    audio, sample_rate = torchaudio.load(audio_path)
    
    # Normaliser la durée à 1 seconde (16000 échantillons)
    target_length = 16000
    if audio.shape[1] > target_length:
        audio = audio[:, :target_length]
    elif audio.shape[1] < target_length:
        pad_length = target_length - audio.shape[1]
        audio = torch.nn.functional.pad(audio, (0, pad_length))
    
    # Ajouter les dimensions nécessaires [batch, channels, time]
    audio = audio.unsqueeze(0)  # [1, channels, time]
    
    # Faire la prédiction
    with torch.no_grad():
        audio = audio.to(device)
        outputs = model(audio)
        _, predicted = torch.max(outputs, 1)
        predicted_label = labels[predicted.item()]
        
    logger.info(f"Prédiction : {predicted_label}")
    return predicted_label

if __name__ == "__main__":
    # Exemple d'utilisation
    test_file = "result_tent_on.wav"
    
    print("Test sans DENT...")
    predicted_class = predict_audio(test_file, tent_on=False)
    print(f"Classe prédite sans DENT : {predicted_class}")
    
    print("\nTest avec DENT...")
    predicted_class = predict_audio(test_file, tent_on=True)
    print(f"Classe prédite avec DENT : {predicted_class}") 