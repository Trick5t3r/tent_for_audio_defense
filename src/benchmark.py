import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from train_pytorch import SpeechCommandsDataset, CNN, ASRAttacks
from test_defense_with_tent import evaluate
from dent.dent import Dent
from dent.conf import cfg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_dent_model(model, device):
    """Configure le modèle avec DENT."""
    # Configuration par défaut de DENT
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
    
    logger.info("Configuration du modèle DENT...")
    dent_model = Dent(model, cfg.OPTIM)
    logger.info(f"Modèle DENT configuré avec : {cfg.OPTIM.METHOD}")
    return dent_model

def benchmark_model(model_path, test_dataset, device, use_dent=False, num_attacks=100):
    """
    Effectue un benchmark du modèle sur des attaques adverses.
    
    Args:
        model_path: Chemin vers le modèle à tester
        test_dataset: Dataset de test
        device: Device pour les calculs
        use_dent: Si True, utilise DENT pour la défense
        num_attacks: Nombre d'attaques à effectuer
    """
    # Charger le modèle
    model = CNN(batch_norm=True, num_classes=3)
    
    # Vérifier l'extension du fichier
    if model_path.endswith('.pth'):
        # Charger directement le dictionnaire
        new_state_dict = torch.load(model_path)
    else:
        # Charger comme actuellement (fichier .pt)
        checkpoint = torch.load(model_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remapper les clés en retirant le préfixe "module." s'il existe.
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

    # Charger le state_dict remappé dans le modèle.
    model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()
    
    # Appliquer DENT si demandé
    if use_dent:
        model = setup_dent_model(model, device)
    
    # Initialiser les métriques
    success_rate = 0
    avg_perturbation = 0
    attack_times = []
    
    # Créer l'instance d'attaque
    labels = ['down', 'up', 'left']
    attack = ASRAttacks(model, device, labels)
    
    # Sélectionner des échantillons "down" pour l'attaque
    down_samples = [(i, audio, label) for i, (audio, label) in enumerate(test_dataset) if label == 0]
    if len(down_samples) < num_attacks:
        logger.warning(f"Moins de {num_attacks} échantillons 'down' disponibles. Utilisation de {len(down_samples)} échantillons.")
        num_attacks = len(down_samples)
    
    # Effectuer les attaques
    for i, (_, audio, _) in tqdm(enumerate(down_samples[:num_attacks]), desc="Benchmark des attaques"):
        # Sauvegarder l'audio original
        original_audio = audio.clone()
        
        # Effectuer l'attaque
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        attacked_audio = attack.BIM_ATTACK(audio.unsqueeze(0), 1, epsilon=0.0015, alpha=0.00009, num_iter=100)
        #attacked_audio = attack.CW_ATTACK(audio.unsqueeze(0), 1, epsilon=0.0015, num_iter=100)
        end_time.record()
        
        torch.cuda.synchronize()
        attack_times.append(start_time.elapsed_time(end_time))
        
        # Calculer la perturbation
        perturbation = torch.abs(attacked_audio - original_audio).mean().item()
        avg_perturbation += perturbation
        
        # Vérifier si l'attaque a réussi
        with torch.no_grad():
            outputs = model(attacked_audio.to(device))
            _, predicted = torch.max(outputs, 1)
            if predicted.item() == 1:  # Si prédit comme "up"
                success_rate += 1
    
    # Calculer les moyennes
    success_rate = (success_rate / num_attacks) * 100
    avg_perturbation = avg_perturbation / num_attacks
    avg_attack_time = np.mean(attack_times)
    
    return {
        'success_rate': success_rate,
        'avg_perturbation': avg_perturbation,
        'avg_attack_time': avg_attack_time
    }

def compare_models(model_paths, test_dataset, device):
    """
    Compare les performances de différents modèles avec et sans DENT.
    
    Args:
        model_paths: Liste des chemins vers les modèles à comparer
        test_dataset: Dataset de test
        device: Device pour les calculs
    """
    results = {}
    
    for model_path in model_paths:
        # Test sans DENT
        logger.info(f"\nBenchmark du modèle sans DENT : {model_path}")
        results[model_path + "_no_dent"] = benchmark_model(model_path, test_dataset, device, use_dent=False)
        
        # Test avec DENT
        logger.info(f"\nBenchmark du modèle avec DENT : {model_path}")
        results[model_path + "_with_dent"] = benchmark_model(model_path, test_dataset, device, use_dent=True)
        
        # Afficher les résultats
        for version in ['no_dent', 'with_dent']:
            key = model_path + "_" + version
            logger.info(f"\nRésultats {version}:")
            logger.info(f"Taux de réussite : {results[key]['success_rate']:.2f}%")
            logger.info(f"Perturbation moyenne : {results[key]['avg_perturbation']:.6f}")
            logger.info(f"Temps moyen d'attaque : {results[key]['avg_attack_time']:.2f} ms")
    
    # Plot des résultats avec Plotly
    plot_benchmark_results(results)
    
    return results

def plot_benchmark_results(results):
    """Plot les résultats des benchmarks avec Plotly."""
    # Organiser les données par modèle
    model_names = []
    success_rates_no_dent = []
    success_rates_with_dent = []
    perturbations_no_dent = []
    perturbations_with_dent = []
    attack_times_no_dent = []
    attack_times_with_dent = []
    
    # Extraire les noms de base des modèles (sans les suffixes _no_dent ou _with_dent)
    base_models = set()
    for path in results.keys():
        base_name = path.replace("_no_dent", "").replace("_with_dent", "")
        base_models.add(base_name)
    
    # Organiser les données par modèle
    for base_model in sorted(base_models):
        # Extraire le nom du fichier pour l'affichage
        display_name = Path(base_model).stem
        model_names.append(display_name)
        
        # Sans DENT
        no_dent_key = base_model + "_no_dent"
        success_rates_no_dent.append(results[no_dent_key]['success_rate'])
        perturbations_no_dent.append(results[no_dent_key]['avg_perturbation'])
        attack_times_no_dent.append(results[no_dent_key]['avg_attack_time'])
        
        # Avec DENT
        with_dent_key = base_model + "_with_dent"
        success_rates_with_dent.append(results[with_dent_key]['success_rate'])
        perturbations_with_dent.append(results[with_dent_key]['avg_perturbation'])
        attack_times_with_dent.append(results[with_dent_key]['avg_attack_time'])
    
    # Création des sous-graphes avec Plotly
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Taux de réussite des attaques", "Perturbation moyenne", "Temps moyen d'attaque"
    ))
    
    x = model_names
    # Graphique du taux de réussite avec couleurs personnalisées
    fig.add_trace(
        go.Bar(name="Sans DENT", x=x, y=success_rates_no_dent, marker=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name="Avec DENT", x=x, y=success_rates_with_dent, marker=dict(color='blue')),
        row=1, col=1
    )
    
    # Graphique de la perturbation moyenne avec couleurs personnalisées
    fig.add_trace(
        go.Bar(name="Sans DENT", x=x, y=perturbations_no_dent, marker=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(name="Avec DENT", x=x, y=perturbations_with_dent, marker=dict(color='blue')),
        row=1, col=2
    )
    
    # Graphique du temps moyen d'attaque avec couleurs personnalisées
    fig.add_trace(
        go.Bar(name="Sans DENT", x=x, y=attack_times_no_dent, marker=dict(color='red')),
        row=1, col=3
    )
    fig.add_trace(
        go.Bar(name="Avec DENT", x=x, y=attack_times_with_dent, marker=dict(color='blue')),
        row=1, col=3
    )
    
    # Mise à jour de la mise en page
    fig.update_layout(
        barmode='group',
        title_text="Résultats des benchmarks",
        template="plotly_white",
        width=1200,
        height=500
    )
    
    # Mise à jour des axes pour chaque sous-graphe
    fig.update_xaxes(title_text="Modèles", row=1, col=1)
    fig.update_xaxes(title_text="Modèles", row=1, col=2)
    fig.update_xaxes(title_text="Modèles", row=1, col=3)
    
    fig.update_yaxes(title_text="Taux de réussite (%)", row=1, col=1)
    fig.update_yaxes(title_text="Perturbation", row=1, col=2)
    fig.update_yaxes(title_text="Temps (ms)", row=1, col=3)
    
    # Sauvegarder la figure en tant qu'image PNG (nécessite kaleido)
    output_path = 'outputs/benchmark_results.png'
    fig.write_image(output_path)
    logger.info(f"Graphique sauvegardé sous : {output_path}")

    
def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path('./Data/speech_commands_v0.01')
    print(dataset_path)
    
    # Créer le dataset de test
    test_dataset = SpeechCommandsDataset(dataset_path, subset='validation')
    
    # Chemins des modèles à comparer
    model_paths = [
        # 'outputs/checkpoints/trained_model/BIM_trained_model/model_epoch_15.pth',  # Modèle avec adversarial training
        'outputs/checkpoints/checkpoint-epoch3.pt',  # Modèle standard
    ]
    
    # Effectuer la comparaison
    results = compare_models(model_paths, test_dataset, device)
    
    # Sauvegarder les résultats dans un fichier
    print("Terliné")
    with open('outputs/benchmark_results.txt', 'w', encoding='utf-8') as f:
        f.write("Résultats des benchmarks :\n\n")
        for model_path, metrics in results.items():
            f.write(f"\nModèle : {model_path}\n")
            f.write(f"Taux de réussite : {metrics['success_rate']:.2f}%\n")
            f.write(f"Perturbation moyenne : {metrics['avg_perturbation']:.6f}\n")
            f.write(f"Temps moyen d'attaque : {metrics['avg_attack_time']:.2f} ms\n")

if __name__ == "__main__":
    main()
