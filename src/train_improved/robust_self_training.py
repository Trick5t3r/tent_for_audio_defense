"""
Main robust self-training script modifié pour un dataset audio (SpeechCommandsDataset).
Basé largement sur le code de https://github.com/yaodongyu/TRADES
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# On importe notre modèle audio via get_model
from utils import get_model
# On importe le dataset audio
from datasets import SpeechCommandsDataset

# Import des fonctions de perte et d'attaque (restent inchangées)
from losses import trades_loss, noise_loss
from attack_pgd import pgd
from smoothing import quick_smoothing

# ----------------------------- CONFIGURATION ----------------------------------
parser = argparse.ArgumentParser(description='PyTorch TRADES Adversarial Training (Audio version)')

# Dataset config
parser.add_argument('--dataset', type=str, default='speechcommands',
                    choices=['cifar10', 'svhn', 'speechcommands'],
                    help='Dataset à utiliser pour l\'entrainement. Ici, speechcommands pour le dataset audio.')
parser.add_argument('--data_dir', default='data', type=str,
                    help='Répertoire où le dataset est situé')

# Model config
parser.add_argument('--model', '-m', default='audio', type=str,
                    help='Nom du modèle (voir utils.get_model). Pour le dataset audio, il utilisera AudioCNN.')
parser.add_argument('--model_dir', default='./rst-model',
                    help='Répertoire pour sauvegarder les checkpoints du modèle')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Annule l\'exécution si un checkpoint approprié est trouvé')
parser.add_argument('--normalize_input', action='store_true', default=False,
                    help='Applique la normalisation standard dès le début dans le réseau')

# Logging et checkpointing
parser.add_argument('--log_interval', type=int, default=5,
                    help='Nombre de batchs entre deux logs de la progression')
parser.add_argument('--save_freq', default=25, type=int,
                    help='Fréquence de sauvegarde des checkpoints (en epochs)')

# Configuration de l'entrainement
parser.add_argument('--seed', type=int, default=1,
                    help='Graine aléatoire')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='Taille des batchs pour l\'entrainement')
parser.add_argument('--test_batch_size', type=int, default=500, metavar='N',
                    help='Taille des batchs pour l\'évaluation')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='Nombre d\'epochs pour l\'entrainement')

# Config d'évaluation
parser.add_argument('--eval_freq', default=1, type=int,
                    help='Fréquence d\'évaluation (en epochs)')
parser.add_argument('--train_eval_batches', default=None, type=int,
                    help='Nombre maximum de batchs pour l\'évaluation sur le train')
parser.add_argument('--eval_attack_batches', default=1, type=int,
                    help='Nombre de batchs à attaquer pour PGD ou certifier avec le smoothing')

# Optimiseur
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='Taux d\'apprentissage')
parser.add_argument('--lr_schedule', type=str, default='cosine',
                    choices=('trades', 'trades_fixed', 'cosine', 'wrn'),
                    help='Plan de décroissance du taux d\'apprentissage')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Momentum pour SGD')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='Utiliser Nesterov')

# Configuration pour l'entrainement adversarial/stabilité
parser.add_argument('--loss', default='trades', type=str,
                    choices=('trades', 'noise'),
                    help='Choix de la fonction de perte : TRADES ou bruit')
parser.add_argument('--distance', '-d', default='l_2', type=str,
                    choices=['l_inf', 'l_2'],
                    help='Métrique pour l\'attaque : l_inf pour adversarial training, l_2 pour stability training')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='Taille de la perturbation adversariale (ou sigma pour stability training)')
parser.add_argument('--pgd_num_steps', default=10, type=int,
                    help='Nombre d\'étapes PGD en entrainement adversarial')
parser.add_argument('--pgd_step_size', default=0.007, type=float,
                    help='Pas PGD pour l\'entrainement adversarial')
parser.add_argument('--beta', default=6.0, type=float,
                    help='Regularisation de stabilité (1/lambda dans TRADES)')

args = parser.parse_args()

# ------------------------------ OUTPUT SETUP ----------------------------------
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.model_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logger.info('Robust self-training (Audio version)')
logger.info('Args: %s', args)

if not args.overwrite:
    final_checkpoint_path = os.path.join(model_dir, 'checkpoint-epoch{}.pt'.format(args.epochs))
    if os.path.exists(final_checkpoint_path):
        logger.info('Checkpoint existant trouvé - arrêt de l\'exécution!')
        sys.exit(0)
# ------------------------------------------------------------------------------

# ------------------------------- CUDA SETUP -----------------------------------
cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')
# ------------------------------------------------------------------------------

# ---------------------- DATASET ET DATA LOADER --------------------------------
if args.dataset == 'speechcommands':
    # Pour le dataset audio, aucune transformation supplémentaire n'est appliquée
    trainset = SpeechCommandsDataset(root_dir=args.data_dir, subset='train')
    testset = SpeechCommandsDataset(root_dir=args.data_dir, subset='test')
    # On fixe le nombre de classes à 3 (down, up, left)
    num_classes = 3
else:
    # Vous pouvez laisser ici le code original pour les datasets image (cifar10, svhn)
    raise ValueError("Ce script modifié est conçu pour le dataset audio 'speechcommands'.")

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)

# Pour l'évaluation sur le train, on peut réutiliser le même dataset (ou en créer une version spécifique)
eval_train_loader = DataLoader(trainset, batch_size=args.test_batch_size, shuffle=True, num_workers=1, pin_memory=True)
eval_test_loader = test_loader
# ------------------------------------------------------------------------------

# ----------------------- FONCTIONS D'ENTRAINEMENT & D'ÉVALUATION ------------------
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_metrics = []
    epsilon = args.epsilon
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Calcul de la perte robuste
        if args.loss == 'trades':
            (loss, natural_loss, robust_loss, entropy_loss_unlabeled) = trades_loss(
                model=model,
                x_natural=data,
                y=target,
                optimizer=optimizer,
                step_size=args.pgd_step_size,
                epsilon=epsilon,
                perturb_steps=args.pgd_num_steps,
                beta=args.beta,
                distance=args.distance,
                adversarial=(args.distance == 'l_inf'),
                entropy_weight=0.0)
        elif args.loss == 'noise':
            loss = noise_loss(model=model, x_natural=data, y=target, clamp_x=True, epsilon=epsilon)
            entropy_loss_unlabeled = torch.Tensor([0.])
            natural_loss = robust_loss = loss

        loss.backward()
        optimizer.step()

        train_metrics.append({
            'epoch': epoch,
            'loss': loss.item(),
            'natural_loss': natural_loss.item(),
            'robust_loss': robust_loss.item(),
            'entropy_loss_unlabeled': entropy_loss_unlabeled.item()
        })

        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return train_metrics

def eval(args, model, device, eval_set, loader):
    loss = 0
    total = 0
    correct = 0
    adv_correct = 0
    adv_correct_clean = 0
    adv_total = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Attaque adversariale pour quelques batchs
            if batch_idx < args.eval_attack_batches:
                if args.distance == 'l_2':
                    incorrect_clean, incorrect_rob = quick_smoothing(
                        model, data, target,
                        sigma=args.epsilon,
                        eps=args.epsilon,
                        num_smooth=100, batch_size=1000)
                elif args.distance == 'l_inf':
                    is_correct_clean, is_correct_rob = pgd(
                        model, data, target,
                        epsilon=args.epsilon,
                        num_steps=2 * args.pgd_num_steps,
                        step_size=args.pgd_step_size,
                        random_start=False)
                    incorrect_clean = (1 - is_correct_clean).sum()
                    incorrect_rob = (1 - np.prod(is_correct_rob, axis=1)).sum()
                else:
                    raise ValueError('Distance %s non supportée' % args.distance)
                adv_correct_clean += (len(data) - int(incorrect_clean))
                adv_correct += (len(data) - int(incorrect_rob))
                adv_total += len(data)
            total += len(data)
    loss /= total
    accuracy = correct / total
    if adv_total > 0:
        robust_clean_accuracy = adv_correct_clean / adv_total
        robust_accuracy = adv_correct / adv_total
    else:
        robust_accuracy = robust_clean_accuracy = 0.

    eval_data = {
        'loss': loss,
        'accuracy': accuracy,
        'robust_accuracy': robust_accuracy,
        'robust_clean_accuracy': robust_clean_accuracy
    }
    eval_data = {eval_set + '_' + k: v for k, v in eval_data.items()}
    logger.info('{}: Clean loss: {:.4f}, Clean accuracy: {}/{} ({:.2f}%), {} clean accuracy: {}/{} ({:.2f}%), Robust accuracy: {}/{} ({:.2f}%)'.format(
        eval_set.upper(), loss, correct, total, 100. * accuracy,
        ('Smoothing' if args.distance == 'l_2' else 'PGD'),
        adv_correct_clean, adv_total, 100. * robust_clean_accuracy,
        adv_correct, adv_total, 100. * robust_accuracy))
    return eval_data

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    schedule = args.lr_schedule
    if schedule == 'trades':
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
    elif schedule == 'trades_fixed':
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
        if epoch >= 0.9 * args.epochs:
            lr = args.lr * 0.01
        if epoch >= args.epochs:
            lr = args.lr * 0.001
    elif schedule == 'cosine':
        lr = args.lr * 0.5 * (1 + np.cos((epoch - 1) / args.epochs * np.pi))
    elif schedule == 'wrn':
        if epoch >= 0.3 * args.epochs:
            lr = args.lr * 0.2
        if epoch >= 0.6 * args.epochs:
            lr = args.lr * 0.04
        if epoch >= 0.8 * args.epochs:
            lr = args.lr * 0.008
    else:
        raise ValueError('Unknown LR schedule %s' % schedule)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# ----------------------------- BOUCLE D'ENTRAINEMENT ----------------------------
def main():
    train_df = pd.DataFrame()
    eval_df = pd.DataFrame()

    # Pour le dataset audio, le nombre de classes est fixé à 3
    num_classes = 3
    model = get_model(args.model, num_classes=num_classes, normalize_input=args.normalize_input)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)

    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(optimizer, epoch)
        logger.info('Learning rate réglé à %g' % lr)
        train_data = train(args, model, device, train_loader, optimizer, epoch)
        train_df = pd.concat([train_df, pd.DataFrame(train_data)], ignore_index=True)


        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            eval_data = {'epoch': epoch}
            eval_data.update(eval(args, model, device, 'train', eval_train_loader))
            eval_data.update(eval(args, model, device, 'test', eval_test_loader))
            #eval_df = eval_df.append(pd.Series(eval_data), ignore_index=True)
            eval_df = pd.concat([eval_df, pd.DataFrame([eval_data])], ignore_index=True)
            logger.info('=' * 120)

        train_df.to_csv(os.path.join(model_dir, 'stats_train.csv'))
        eval_df.to_csv(os.path.join(model_dir, 'stats_eval.csv'))

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            torch.save({
                'num_classes': num_classes,
                'state_dict': model.state_dict(),
                'normalize_input': args.normalize_input
            }, os.path.join(model_dir, 'checkpoint-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(model_dir, 'opt-checkpoint_epoch{}.tar'.format(epoch)))

if __name__ == '__main__':
    main()
