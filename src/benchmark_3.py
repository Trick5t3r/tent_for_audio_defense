import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # import corrigé pour Plotly
from train_pytorch import SpeechCommandsDataset, CNN, ASRAttacks
from test_defense_with_tent import evaluate
from dent.dent import Dent
from dent.conf import cfg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader

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
    # Vérifier l'extension du fichier pour charger correctement le state dict
    if model_path.endswith('.pth'):
        new_state_dict = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

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
        # Utilisation de BIM_ATTACK (comme dans le script fonctionnel)
        attacked_audio = attack.BIM_ATTACK(audio.unsqueeze(0), 1, epsilon=0.0015, alpha=0.00009, num_iter=100)
        # Pour utiliser une autre attaque, décommentez l'une des lignes suivantes :
        # attacked_audio = attack.CW_ATTACK(audio.unsqueeze(0), 1, epsilon=0.0015, num_iter=100)
        # attacked_audio = attack.MIM_ATTACK(audio.unsqueeze(0), 1, epsilon=0.0015, alpha=0.00009, num_iter=100)
        end_time.record()
        
        torch.cuda.synchronize()
        attack_times.append(start_time.elapsed_time(end_time))
        
        # Calculer la perturbation
        perturbation = torch.abs(attacked_audio - original_audio).mean().item()
        avg_perturbation += perturbation
        
        # Vérifier si l'attaque a réussi (si le modèle prédit "up" => index 1)
        with torch.no_grad():
            outputs = model(attacked_audio.to(device))
            _, predicted = torch.max(outputs, 1)
            if predicted.item() == 1:
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

def evaluate_model(model_path, test_dataset, device, use_dent=False):
    """
    Évalue les performances du modèle sur le dataset de test sans attaque.
    
    Args:
        model_path: Chemin vers le modèle à tester
        test_dataset: Dataset de test
        device: Device pour les calculs
        use_dent: Si True, utilise DENT pour la défense
    
    Returns:
        dict: Dictionnaire contenant l'accuracy et le temps d'inférence moyen
    """
    # Charger le modèle
    model = CNN(batch_norm=True, num_classes=3)
    if model_path.endswith('.pth'):
        new_state_dict = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    # Appliquer DENT si demandé
    if use_dent:
        model = setup_dent_model(model, device)
    
    # Créer un DataLoader pour le test
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialiser les métriques
    correct = 0
    total = 0
    inference_times = []
    
    # Évaluer le modèle
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Évaluation du modèle"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mesurer le temps d'inférence
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            outputs = model(inputs)
            end_time.record()
            
            torch.cuda.synchronize()
            inference_times.append(start_time.elapsed_time(end_time))
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_inference_time = np.mean(inference_times)
    
    return {
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time
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
        # Évaluation sans attaque
        logger.info(f"\nÉvaluation du modèle sans attaque : {model_path}")
        
        # Sans DENT
        logger.info("Sans DENT:")
        results[model_path + "_no_dent_clean"] = evaluate_model(model_path, test_dataset, device, use_dent=False)
        logger.info(f"Accuracy: {results[model_path + '_no_dent_clean']['accuracy']:.2f}%")
        logger.info(f"Temps d'inférence moyen: {results[model_path + '_no_dent_clean']['avg_inference_time']:.2f} ms")
        
        # Avec DENT
        logger.info("Avec DENT:")
        results[model_path + "_with_dent_clean"] = evaluate_model(model_path, test_dataset, device, use_dent=True)
        logger.info(f"Accuracy: {results[model_path + '_with_dent_clean']['accuracy']:.2f}%")
        logger.info(f"Temps d'inférence moyen: {results[model_path + '_with_dent_clean']['avg_inference_time']:.2f} ms")
        
        # Test avec attaques
        logger.info(f"\nBenchmark du modèle sans DENT avec attaques : {model_path}")
        results[model_path + "_no_dent"] = benchmark_model(model_path, test_dataset, device, use_dent=False)
        
        logger.info(f"\nBenchmark du modèle avec DENT avec attaques : {model_path}")
        results[model_path + "_with_dent"] = benchmark_model(model_path, test_dataset, device, use_dent=True)
        
        # Afficher les résultats des attaques
        for version in ['no_dent', 'with_dent']:
            key = model_path + "_" + version
            logger.info(f"\nRésultats {version} avec attaques:")
            logger.info(f"Taux de réussite : {results[key]['success_rate']:.2f}%")
            logger.info(f"Perturbation moyenne : {results[key]['avg_perturbation']:.6f}")
            logger.info(f"Temps moyen d'attaque : {results[key]['avg_attack_time']:.2f} ms")
    
    # Plot des résultats
    plot_benchmark_results(results)
    plot_clean_results(results)
    
    return results

def plot_benchmark_results(results):
    """Plot les résultats des benchmarks avec attaques avec Plotly."""
    model_names = []
    success_rates_no_dent = []
    success_rates_with_dent = []
    perturbations_no_dent = []
    perturbations_with_dent = []
    attack_times_no_dent = []
    attack_times_with_dent = []
    
    base_models = set()
    for path in results.keys():
        if "_clean" not in path:
            base_name = path.replace("_no_dent", "").replace("_with_dent", "")
            base_models.add(base_name)
    
    for base_model in sorted(base_models):
        display_name = Path(base_model).stem
        model_names.append(display_name)
        
        no_dent_key = base_model + "_no_dent"
        success_rates_no_dent.append(results[no_dent_key]['success_rate'])
        perturbations_no_dent.append(results[no_dent_key]['avg_perturbation'])
        attack_times_no_dent.append(results[no_dent_key]['avg_attack_time'])
        
        with_dent_key = base_model + "_with_dent"
        success_rates_with_dent.append(results[with_dent_key]['success_rate'])
        perturbations_with_dent.append(results[with_dent_key]['avg_perturbation'])
        attack_times_with_dent.append(results[with_dent_key]['avg_attack_time'])
    
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=('Taux de réussite des attaques', 
                                          'Perturbation moyenne', 
                                          "Temps moyen d'attaque"),
                        horizontal_spacing=0.1)
    
    fig.add_trace(
        go.Bar(x=model_names, y=success_rates_no_dent, name='Sans DENT', marker_color='royalblue', offsetgroup=0),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=model_names, y=success_rates_with_dent, name='Avec DENT', marker_color='darkorange', offsetgroup=1),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=model_names, y=perturbations_no_dent, name='Sans DENT', marker_color='royalblue', offsetgroup=0, showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=model_names, y=perturbations_with_dent, name='Avec DENT', marker_color='darkorange', offsetgroup=1, showlegend=False),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=model_names, y=attack_times_no_dent, name='Sans DENT', marker_color='royalblue', offsetgroup=0, showlegend=False),
        row=1, col=3
    )
    fig.add_trace(
        go.Bar(x=model_names, y=attack_times_with_dent, name='Avec DENT', marker_color='darkorange', offsetgroup=1, showlegend=False),
        row=1, col=3
    )
    
    fig.update_layout(
        title='Résultats des benchmarks avec attaques',
        height=500,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(title_text='Taux de réussite (%)', row=1, col=1)
    fig.update_yaxes(title_text='Perturbation', row=1, col=2)
    fig.update_yaxes(title_text='Temps (ms)', row=1, col=3)
    
    fig.write_html('outputs/benchmark_results.html')
    fig.write_image('outputs/benchmark_results.png')

def plot_clean_results(results):
    """Plot les résultats d'évaluation sur données propres avec Plotly."""
    model_names = []
    accuracy_no_dent = []
    accuracy_with_dent = []
    inference_time_no_dent = []
    inference_time_with_dent = []
    
    base_models = set()
    for path in results.keys():
        if "_clean" in path:
            base_name = path.replace("_no_dent_clean", "").replace("_with_dent_clean", "")
            base_models.add(base_name)
    
    for base_model in sorted(base_models):
        display_name = Path(base_model).stem
        model_names.append(display_name)
        
        no_dent_key = base_model + "_no_dent_clean"
        accuracy_no_dent.append(results[no_dent_key]['accuracy'])
        inference_time_no_dent.append(results[no_dent_key]['avg_inference_time'])
        
        with_dent_key = base_model + "_with_dent_clean"
        accuracy_with_dent.append(results[with_dent_key]['accuracy'])
        inference_time_with_dent.append(results[with_dent_key]['avg_inference_time'])
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Accuracy sur données propres', 
                                          "Temps d'inférence moyen"),
                        horizontal_spacing=0.15)
    
    fig.add_trace(
        go.Bar(x=model_names, y=accuracy_no_dent, name='Sans DENT', marker_color='royalblue', offsetgroup=0),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=model_names, y=accuracy_with_dent, name='Avec DENT', marker_color='darkorange', offsetgroup=1),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=model_names, y=inference_time_no_dent, name='Sans DENT', marker_color='royalblue', offsetgroup=0, showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=model_names, y=inference_time_with_dent, name='Avec DENT', marker_color='darkorange', offsetgroup=1, showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Évaluation sur données propres',
        height=500,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(title_text='Accuracy (%)', row=1, col=1)
    fig.update_yaxes(title_text='Temps (ms)', row=1, col=2)
    
    fig.write_html('outputs/clean_evaluation_results.html')
    fig.write_image('outputs/clean_evaluation_results.png')

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path('./Data/speech_commands_v0.01')
    
    # Créer le dataset de test
    test_dataset = SpeechCommandsDataset(dataset_path, subset='validation')
    
    # Chemin(s) des modèles à comparer
    model_paths = [
        'outputs/checkpoints/checkpoint-epoch27.pt',  # Modèle standard
    ]
    
    # Effectuer la comparaison
    results = compare_models(model_paths, test_dataset, device)
    
    # Sauvegarder les résultats dans un fichier
    with open('outputs/benchmark_results.txt', 'w', encoding='utf-8') as f:
        f.write("Résultats des benchmarks :\n\n")
        
        f.write("\n=== ÉVALUATION SUR DONNÉES PROPRES ===\n")
        for model_path in model_paths:
            clean_key = model_path + "_no_dent_clean"
            f.write(f"\nModèle sans DENT : {model_path}\n")
            f.write(f"Accuracy : {results[clean_key]['accuracy']:.2f}%\n")
            f.write(f"Temps d'inférence moyen : {results[clean_key]['avg_inference_time']:.2f} ms\n")
            
            clean_key = model_path + "_with_dent_clean"
            f.write(f"\nModèle avec DENT : {model_path}\n")
            f.write(f"Accuracy : {results[clean_key]['accuracy']:.2f}%\n")
            f.write(f"Temps d'inférence moyen : {results[clean_key]['avg_inference_time']:.2f} ms\n")
        
        f.write("\n\n=== RÉSULTATS DES ATTAQUES ===\n")
        for model_path in model_paths:
            attack_key = model_path + "_no_dent"
            f.write(f"\nModèle sans DENT : {model_path}\n")
            f.write(f"Taux de réussite : {results[attack_key]['success_rate']:.2f}%\n")
            f.write(f"Perturbation moyenne : {results[attack_key]['avg_perturbation']:.6f}\n")
            f.write(f"Temps moyen d'attaque : {results[attack_key]['avg_attack_time']:.2f} ms\n")
            
            attack_key = model_path + "_with_dent"
            f.write(f"\nModèle avec DENT : {model_path}\n")
            f.write(f"Taux de réussite : {results[attack_key]['success_rate']:.2f}%\n")
            f.write(f"Perturbation moyenne : {results[attack_key]['avg_perturbation']:.6f}\n")
            f.write(f"Temps moyen d'attaque : {results[attack_key]['avg_attack_time']:.2f} ms\n")

if __name__ == "__main__":
    main()
