# Défense contre les Attaques Adversaires sur l'Audio avec TENT

Ce projet implémente une défense contre les attaques adversaires sur des modèles de reconnaissance vocale en utilisant la méthode TENT (Test-Time Entropy Minimization). Le projet est basé sur le framework DENT (Dequan Wang's Entropy Minimization) et est adapté pour la défense contre les attaques BIM (Basic Iterative Method) sur des fichiers audio.

## 🚀 Fonctionnalités

- Implémentation de la défense TENT pour la reconnaissance vocale
- Support des attaques BIM (Basic Iterative Method)
- Traitement de fichiers audio WAV
- Visualisation des résultats avec TensorBoard
- Modèle CNN personnalisé pour la classification audio

## 📋 Prérequis

- Python 3.8+
- PyTorch 2.6.0
- CUDA (recommandé pour l'accélération GPU)

## 🛠️ Installation

1. Cloner le repository :
```bash
git clone https://github.com/Trick5t3r/tent_for_audio_defense.git
cd tent_for_audio_defense
```

2. Créer et activer un environnement virtuel :
```bash
python -m venv .venv
source .venv/bin/activate  # Sur Linux/Mac
.venv\Scripts\activate     # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 📁 Structure du Projet

```
tent_for_audio_defense/
├── src/
│   └── test_defense_with_tent.py    # Script principal de test
├── dent/                            # Module DENT
│   ├── cfgs/                        # Fichiers de configuration
│   └── dent.py                      # Implémentation de DENT
├── outputs/                         # Dossiers des résultats
│   └── checkpoints/                 # Points de contrôle du modèle
├── Data/                           # Données d'entraînement/test
└── requirements.txt                # Dépendances du projet
```

## 🎯 Utilisation

Pour tester la défense sur un fichier audio :

```python
from src.test_defense_with_tent import evaluate

# Test sans défense TENT
result1, transcription1 = evaluate("chemin/vers/audio.wav", "cible", dent_on=False)

# Test avec défense TENT
result2, transcription2 = evaluate("chemin/vers/audio.wav", "cible", dent_on=True)
```

Les paramètres disponibles :
- `input_audio` : Chemin vers le fichier audio ou tensor audio
- `target` : Transcription cible
- `dent_on` : Active/désactive la défense TENT
- `num_iter` : Nombre d'itérations pour l'attaque (défaut: 500)

## 🔧 Configuration

Les paramètres de configuration sont définis dans `dent/cfgs/dent.yaml`. Les principaux paramètres incluent :
- Taille du batch
- Architecture du modèle
- Paramètres d'optimisation
- Fonction de normalisation

## 📊 Résultats

Les résultats sont sauvegardés dans le dossier `outputs/` :
- Fichiers audio générés : `result_tent_on.wav` et `result_tent_off.wav`
- Logs TensorBoard pour la visualisation des métriques

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 🙏 Remerciements

- [DENT](https://github.com/DequanWang/dent) - Framework original pour la défense contre les attaques adversaires
- [Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) - Dataset créé par l'équipe TensorFlow pour la reconnaissance vocale
- PyTorch et la communauté open source pour les outils et bibliothèques utilisés

## Description

Le modèle prend en entrée des fichiers audios de type .wav
Pour utiliser la méthode de tent qui prend en entrée des images, les audios sont tranformés en spectrogramme. Les spectrogrammes sont passés dans un CNN ( du fichier model_pytorch.py) pour determiner la classe d'appartenance

(Pas encore dans le github) Des exemples sont fournis pour tester des attaques.

## Installation 
#### Téléchargement du dataset:
http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

Puis de-zipper le fichier et le mettre dans le dossier Data

#### Installation des packages
```bash
pip install -r requirements.txt
```

```bash
git clone https://github.com/DequanWang/dent.git
```

## Sources
https://github.com/dataflowr/Project-Tent-Test-Time-Domain-Adaptation?tab=readme-ov-file

http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

https://github.com/hammaad2002/ASRAdversarialAttacks.git
