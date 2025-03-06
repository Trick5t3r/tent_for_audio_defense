# tent_for_audio_defense


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

https://github.com/zhenghuatan/Audio-adversarial-examples/tree/main?tab=readme-ov-file

https://github.com/hammaad2002/ASRAdversarialAttacks.git
