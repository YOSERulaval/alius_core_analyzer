"""
Fichier source des méthodes utilitaires
"""
from sympy.printing.pytorch import torch

def clean_cache():
    """
    Vide la mémoire allouée à CUDA. Important d'appeler cette fonction au début de chaque méthode d'analyse. \n
    Est une fonction stand alone au cas où on voudrait ajouter quelque chose d'autre.
    """
    torch.cuda.empty_cache()

def make_new_dir(dir_name):
    """
    Auteur : Yohan Sercia \n
    Dernière MAJ : 22-09-2025 \n
    Description :  \n
    :param dir_name:
    :return:
    """
    import os

    root_dir = 'project_output'
    output_dir = os.path.join(root_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir

def pixel_to_meter(core_runs_sorted, core_diameter):
    """
    Calcul la hauteur (y2-y1) de TOUS les cores dans le dataset et fait la moyenne de cette hauteur.
    :param core_runs_sorted: Liste des cores triés.
    :param core_diameter: Diamètre du core à analyser (NQ, HQ, etc). Utiliser comme mesure de référence
    :return: Le ratio de conversion pixel<->mètre
    """
    d_core = core_diameter
    heights = [core.bbox[3] - core.bbox[1] for core in core_runs_sorted]
    d_average = sum(heights) / len(heights) # moyenne

    return d_core / d_average

def bbox_contains_point(bbox, point):
    """
    Vérifie si un point (x, y) se trouve à l'intérieur d'une bbox [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox # Délimitation de la bbox
    x, y = point # Point centre du joint (ou d'un autre attribut)
    return x1 <= x <= x2 and y1 <= y <= y2 # return True si le point est à l'intérieur de la bbox, alors on sait que le joint (ou l'attribut) appartient à cette 'run' (core)

def sort_cores(core_runs):
    """
    :param core_runs: Accepte une liste de liste, ex: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...] qui représente tous
    les bbox des cores
    :return: Une liste de liste trié en fonction de y1 d'abord suivi de x1.
    """
    #return sorted(core_runs, key=lambda core: (core.bbox[0], core.bbox[1]))

    sorted_by_x = sorted(core_runs, key=lambda core: core.bbox[0])
    return sorted(sorted_by_x, key=lambda core: core.bbox[1]) # gauche à droite, haut en bas