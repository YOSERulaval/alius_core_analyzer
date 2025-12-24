"""
Alius Consultation Minière inc.
====================================
CRÉÉ PAR:       Yohan Sercia
CRÉE LE:        17-12-2025
------------------------------------
PROJET:         [PHOTO LOGGING]
DESCRIPTION:    [DESCRIPTION]
------------------------------------
DERNIÈRE MAJ:   17-12-2025
====================================
"""

"""
==========================
!! CECI EST UN TEMPLATE !!
==========================
APT = Algorithme Post Traitement
Créer un fichier .py pour votre APT à l'intérieur de 'root/APT/ '

Normalement, l'interface va pouvoir trouver votre APT automatiquement. Pour fonctionnement, voir 'useModel.py -> ModelPredictor.get_methods(self)'
"""

from sources_files import methodLinker as mlinker

class MonAPT(mlinker.Method):
    name = "APT - [NOM DE L'APT] ([INITIALS])" # TODO

    def run(self, *args, **kwargs):
        """
        # TODO « Point d'entrée » de l'APT dans un objet de type ModelPredictor "model".
        # TODO :param args: Mettre vos arguments
        # TODO :param kwargs: Mettre vos arguments
        # TODO :return: à remplir
        """
        return self._mon_algorithme() # La méthode virtuelle est remplacée par votre APT lorsque vous l'appelé dans le return

    def _mon_algorithme(self, *args, **kwargs):
        """
        # TODO C'est ici que la logique de votre APT doit être implémentée
        # TODO :param args: Mettre vos arguments
        # TODO :param kwargs: Mettre vos arguments
        # TODO :return: à remplir
        """
        # TODO logique de l'APT