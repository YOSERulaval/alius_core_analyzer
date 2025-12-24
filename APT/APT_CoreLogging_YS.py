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
# TODO porter mon APT ici
from sources_files import methodLinker as mlinker

class CoreLoggingAPT(mlinker.Method):
    name = "APT - Core Logging (YS)"

    def run(self, *args, **kwargs):
        """
        Ici tu utilises self.model (ModelPredictor) comme API centrale.
        Exemples possibles:
          - self.model.analyze_folder(...)
          - self.model.predictor(...) (si tu exposes un predict propre)
          - self.model.output_path, etc.
        """

        # Demo simple: appeler une méthode déjà existante dans ModelPredictor
        # (adapte les args à ta signature réelle)
        print("[CoreLoggingAPT] Running APT...")
        return self.model.analyze_folder()