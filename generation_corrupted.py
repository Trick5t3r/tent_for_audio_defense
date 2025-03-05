from ASRAdversarialAttacks.AdversarialAttacks import ASRAttacks
import torchaudio
import torch
import numpy as np
from IPython.display import Audio
import os
import soundfile as sf



# Loading the model from torchaudio model hub
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()

# Checking the device available during the current environment (CUDA is recommended!)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

attack = ASRAttacks(model, device, bundle.get_labels())


def path_fichier(path_dossier_audio):
    # Liste pour stocker les chemins des fichiers audio
    chemins_fichiers = []

    # Lister tous les fichiers dans le répertoire
    for fichier in os.listdir(path_dossier_audio):
        # Vérifier si le fichier est un fichier .wav
        if fichier.endswith('.wav'):
            chemin_fichier = os.path.join(path_dossier_audio, fichier)
            chemins_fichiers.append(chemin_fichier)

    chemins_fichiers=np.array(chemins_fichiers)

    return chemins_fichiers



def generate_corrupted_data(path_dossier:str,true_transcription:str,target_transcription:str,attack_type:str,early_stop:bool=True):
    '''Crée les fichier audio corrompus en fonction de la cible et du type d'algorithme utilisé
    attack_type = PGD, BIM, CW, IMPERCEPTIBLE_ASR
    '''

    target = list(target_transcription.upper().replace(" ", "|"))


    #Path to the files
    chemins_fichiers = path_fichier(path_dossier)

    data=[]
    for chemin in chemins_fichiers:
        input_audio, sample_rate = torchaudio.load(chemin)
        data.append( (input_audio, sample_rate))

    data = np.array(data, dtype=object)

    
    L_corrupted=[]
    for x in data:
        input_audio, sample_rate=x

        if attack_type=="FGSM":
            res=attack.FGSM_ATTACK(input_audio, target, epsilon = 0.01, targeted = True)
        elif attack_type=="BIM":
            res=attack.BIM_ATTACK(input_audio, target, epsilon = 0.0015, alpha = 0.00009,num_iter = 3000, targeted = True, early_stop = early_stop)
        elif attack_type=="PGD":
            res=attack.PGD_ATTACK(input_audio, target, epsilon = 0.0015, alpha = 0.00009,num_iter = 2500, targeted = True, early_stop = early_stop)
        elif attack_type=="CW":
            res=attack.CW_ATTACK(input_audio, target, epsilon = 0.0015, c = 10,
                    learning_rate = 0.00001, num_iter = 10000, decrease_factor_eps = 1,
                    num_iter_decrease_eps = 10, optimizer = None, nested = True,
                    early_stop = early_stop, search_eps = False, targeted = True)
        elif attack_type=="IMPERCEPTIBLE_ASR":
            res=attack.IMPERCEPTIBLE_ATTACK(torch.nn.functional.pad(input_audio, (0, 1000)), target, epsilon = 0.015, c = 10, learning_rate1 = 0.001,
                                        learning_rate2 = 0.0001, num_iter1 = 1000, num_iter2 = 15000, decrease_factor_eps = 1,
                                        num_iter_decrease_eps = 10, optimizer1 = None, optimizer2 = "Adam",nested = True ,
                                        early_stop_cw = early_stop, search_eps_cw = False, alpha = 0.05)
        
        L_corrupted.append(res)

    
    return L_corrupted


def transcription(liste):
    '''Affiche ce que le modèle détecte de la transcription des fichiers '''
    res=[]
    for x in liste:
        res.append(attack.INFER(torch.from_numpy(x)).replace("|"," "))
    return res


def attacks_sur_un_fichier(path:str,target_transcription:str,early_stop:bool=True):

    input_audio, sample_rate = torchaudio.load(path)
    target = list(target_transcription.upper().replace(" ", "|"))

    res1=attack.FGSM_ATTACK(input_audio, target, epsilon = 0.01, targeted = True)
    res2=attack.BIM_ATTACK(input_audio, target, epsilon = 0.0015, alpha = 0.00009,num_iter = 3000, targeted = True, early_stop = early_stop)
    res3=attack.PGD_ATTACK(input_audio, target, epsilon = 0.0015, alpha = 0.00009,num_iter = 2500, targeted = True, early_stop = early_stop)
    res4=attack.CW_ATTACK(input_audio, target, epsilon = 0.0015, c = 10,
        learning_rate = 0.00001, num_iter = 10000, decrease_factor_eps = 1,
        num_iter_decrease_eps = 10, optimizer = None, nested = True,
        early_stop = early_stop, search_eps = False, targeted = True)
    res5=attack.IMPERCEPTIBLE_ATTACK(torch.nn.functional.pad(input_audio, (0, 1000)), target, epsilon = 0.015, c = 10, learning_rate1 = 0.001,
                            learning_rate2 = 0.0001, num_iter1 = 1000, num_iter2 = 15000, decrease_factor_eps = 1,
                            num_iter_decrease_eps = 10, optimizer1 = None, optimizer2 = "Adam",nested = True ,
                            early_stop_cw = early_stop, search_eps_cw = False, alpha = 0.05)
    
    # Chemin de destination pour le fichier .wav
    destination1 =os.path.splitext(os.path.basename(path))[0] + "_corrupted_" + 'FGSM_ATTACK' + '.wav'
    destination2 =os.path.splitext(os.path.basename(path))[0] + "_corrupted_" + 'BIM_ATTACK' + '.wav'
    destination3 =os.path.splitext(os.path.basename(path))[0] + "_corrupted_" + 'PGD_ATTACK' + '.wav'
    destination4 =os.path.splitext(os.path.basename(path))[0] + "_corrupted_" + 'CW_ATTACK' + '.wav'
    destination5 =os.path.splitext(os.path.basename(path))[0] + "_corrupted_" + 'IMPERCEPTIBLE_ATTACK' + '.wav'

    sf.write(destination1, res1[0], sample_rate)
    sf.write(destination2, res2[0], sample_rate)
    sf.write(destination3, res3[0], sample_rate)
    sf.write(destination4, res4[0], sample_rate)
    sf.write(destination5, res5[0], sample_rate)

    print(f"Fichiers audios enregistrés")

if __name__=="__main__":
    path_dossier= 'adversarial_dataset-B/Normal-Examples/down/dataset_dir/down'
    attack_type = "FGSM" #FGSM, PGD, BIM, CW, IMPERCEPTIBLE_ASR
    early_stop=True # Stop the attack when the target transcription is reached

    true_transcription = 'DOWN'
    target_transcription = 'UP'
    

    x=generate_corrupted_data(path_dossier,true_transcription,target_transcription,attack_type,early_stop)
    y=transcription(x)
    print(y[0])
    print("Finished")

