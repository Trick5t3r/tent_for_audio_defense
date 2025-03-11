"""
Robust training losses. Based on code from
https://github.com/yaodongyu/TRADES
Adapté pour la formation robuste sur des données audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def entropy_loss(unlabeled_logits):
    unlabeled_probs = F.softmax(unlabeled_logits, dim=1)
    return -(unlabeled_probs * F.log_softmax(unlabeled_logits, dim=1)).sum(dim=1).mean(dim=0)

def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                adversarial=True,
                distance='inf',
                entropy_weight=0):
    """
    TRADES KL-robustness regularization with additional support for stability training
    and entropy regularization.
    
    Adapté pour le traitement audio. Remarquez que :
      - Si beta == 0, on effectue uniquement une classification standard.
      - Si adversarial est True, on suppose un entraînement adversarial en l_inf;
        sinon, pour distance == 'l_2', on réalise un training de stabilité en ajoutant du bruit.
    """
    if beta == 0:
        logits = model(x_natural)
        loss = F.cross_entropy(logits, y, ignore_index=-1)
        inf = torch.tensor([float('inf')], device=x_natural.device)
        zero = torch.tensor([0.0], device=x_natural.device)
        return loss, loss, inf, zero

    # Définition de la perte KL avec reduction 'sum'
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()  # Mode évaluation pour geler les stats des batchnorm
    batch_size = x_natural.size(0)
    # Génération de l'exemple adversarial
    x_adv = x_natural.detach().clone()

    if adversarial:
        if distance == 'l_inf':
            # Ajout d'un bruit de départ (device-agnostique)
            x_adv += 0.001 * torch.randn_like(x_natural).detach()
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                            F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                # Projection dans la boule de perturbation
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise ValueError(f'No support for distance {distance} in adversarial training')
    else:
        if distance == 'l_2':
            x_adv = x_adv + epsilon * torch.randn_like(x_adv)
        else:
            raise ValueError(f'No support for distance {distance} in stability training')

    model.train()  # Retour en mode entraînement pour mettre à jour les stats des batchnorm

    # S'assurer que les exemples adversariaux sont dans la plage valide
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    logits_adv = F.log_softmax(model(x_adv), dim=1)
    logits = model(x_natural)

    loss_natural = F.cross_entropy(logits, y, ignore_index=-1)
    p_natural = F.softmax(logits, dim=1)
    loss_robust = criterion_kl(logits_adv, p_natural) / batch_size

    loss = loss_natural + beta * loss_robust

    is_unlabeled = (y == -1)
    if torch.sum(is_unlabeled) > 0:
        logits_unlabeled = logits[is_unlabeled]
        loss_entropy_unlabeled = entropy_loss(logits_unlabeled)
        loss = loss + entropy_weight * loss_entropy_unlabeled
    else:
        loss_entropy_unlabeled = torch.tensor(0.0, device=x_natural.device)

    return loss, loss_natural, loss_robust, loss_entropy_unlabeled

def noise_loss(model,
               x_natural,
               y,
               epsilon=0.25,
               clamp_x=True):
    """
    Ajout de bruit aléatoire sur l'entrée, comme dans Cohen et al.
    """
    x_noise = x_natural + epsilon * torch.randn_like(x_natural)
    if clamp_x:
        x_noise = x_noise.clamp(0.0, 1.0)
    logits_noise = model(x_noise)
    loss = F.cross_entropy(logits_noise, y, ignore_index=-1)
    return loss
