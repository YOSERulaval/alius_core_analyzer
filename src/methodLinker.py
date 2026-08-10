"""
Alius Consultation Minière inc.
====================================
CRÉÉ PAR:       Yohan Sercia
CRÉE LE:        17-12-2025
------------------------------------
PROJET:         Geotech IA
DESCRIPTION:    [DESCRIPTION]
------------------------------------
DERNIÈRE MAJ:   17-12-2025
====================================
"""

# sources_files/methodLinker.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

import importlib.util
import inspect
import traceback
from abc import ABC, abstractmethod


class Method(ABC):
    """
    Classe de base pour les APT (plugins).
    Chaque APT doit hériter de Method et implémenter run().
    """
    # Ce nom sera affiché dans l'UI (ou dans ton dict get_methods)
    name: str = "Unnamed APT"

    def __init__(self, model, APT_output_dir = "APT", **kwargs):
        """
        model: instance de ModelPredictor (ton coeur Detectron2 + helpers)
        APT_output_dir: si tu veux offrir un dossier d'export au plugin
        kwargs: futur-proof (tu pourras injecter d'autres trucs plus tard)
        """
        self.model = model
        self.APT_output_dir = APT_output_dir

    @abstractmethod
    def run(self, *args, **kwargs):
        """Point d'entrée standard de l'APT."""
        raise NotImplementedError


def default_apt_dir() -> Path:
    """
    Suppose structure:
    alius_core_analyzer/
      APT/
      sources_files/
    """
    here = Path(__file__).resolve()
    project_root = here.parents[1]  # .../alius_core_analyzer
    return project_root / "APT"


def discover_method_classes(apt_dir: Optional[Path] = None) -> List[Type[Method]]:
    """
    Scanne le dossier APT et retourne toutes les classes qui héritent de Method.
    """
    apt_dir = Path(apt_dir) if apt_dir else default_apt_dir()
    classes: List[Type[Method]] = []

    if not apt_dir.exists():
        return classes

    for py_file in apt_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        module_name = f"apt_plugins.{py_file.stem}"  # nom arbitraire mais stable
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(py_file))
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # exécute le fichier -> attention: code arbitraire

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Method) and obj is not Method:
                    classes.append(obj)

        except Exception:
            print(f"[APT loader] Impossible de charger {py_file.name}")
            traceback.print_exc()

    return classes


def build_apt_callables(model, apt_dir: Optional[Path] = None) -> Dict[str, Callable]:
    """
    Instancie chaque APT et renvoie un dict: "Nom APT" -> callable à exécuter.
    """
    callables: Dict[str, Callable] = {}

    for cls in discover_method_classes(apt_dir):
        try:
            apt_instance = cls(model=model)  # injection du ModelPredictor
            label = getattr(apt_instance, "name", cls.__name__)

            # On expose apt_instance.run comme "fonction" exécutable
            callables[label] = apt_instance.run

        except Exception:
            print(f"[APT loader] Erreur à l'instanciation de {cls.__name__}")
            traceback.print_exc()

    return callables
