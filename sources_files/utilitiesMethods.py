"""
Fichier source des méthodes utilitaires
"""

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