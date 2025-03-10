import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

def pgd(model,
        X,
        y,
        epsilon=0.031,
        num_steps=20,
        step_size=0.007,
        random_start=True):
    """
    PGD attack adapté pour des signaux audio dont les valeurs sont supposées
    être dans l'intervalle [-1, 1].
    
    Args:
        model: le modèle cible.
        X: tenseur d'entrée (batch de signaux audio).
        y: labels cibles.
        epsilon: perturbation maximale.
        num_steps: nombre d'itérations de PGD.
        step_size: pas de mise à jour.
        random_start: initialisation aléatoire de la perturbation.
    
    Returns:
        is_correct_natural: prédictions sur les exemples non perturbés.
        is_correct_adv: tableau indiquant pour chaque étape si la prédiction est correcte.
    """
    # Prédictions sur les données naturelles
    out = model(X)
    is_correct_natural = (out.max(1)[1] == y).float().cpu().numpy()
    
    # Initialisation de la perturbation
    perturbation = torch.zeros_like(X, requires_grad=True)
    if random_start:
        perturbation = torch.rand_like(X, requires_grad=True)
        perturbation.data = perturbation.data * 2 * epsilon - epsilon

    is_correct_adv = []
    optimizer = optim.SGD([perturbation], lr=1e-3)  # Un optimiseur temporaire pour réinitialiser le gradient

    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X + perturbation), y)
        loss.backward()

        # Mise à jour de la perturbation par gradient sign
        perturbation.data = (perturbation + step_size * perturbation.grad.detach().sign()).clamp(-epsilon, epsilon)
        # Contraindre X + perturbation à rester dans [-1, 1] pour l'audio :
        perturbation.data = torch.min(torch.max(perturbation.detach(), -1 - X), 1 - X)
        # Calcul de l'entrée perturbée
        X_pgd = torch.clamp(X + perturbation.data, -1, 1)
        is_correct_adv.append((model(X_pgd).max(1)[1] == y).float().cpu().numpy().reshape(-1, 1))
    
    is_correct_adv = np.concatenate(is_correct_adv, axis=1)
    return is_correct_natural, is_correct_adv
