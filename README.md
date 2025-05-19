# alius_core_analyzer
Application d'analyse de 'core' utilisant du ML.

# Que fait ce code ?
Pour l'instant, l'idée est d'utiliser plusieurs modèle entraîner à l'aide des modèles par défaut de la librairie *detectron2*.
Un premier modèle s'assure que les images à étudier sont du bon format (cropped, normalized). Un deuxième s'occupera de l'analyse des cores.

## Image Cropper
D'abord, l'application gère le formatage des images:
  1. Import d'un fichier de base donnée
  2. Cropping
  3. Export en amont dans le pipeline

## Core Analyzer
Utilisation des images ''préparées'' pour débuter l'analyse des cores:
  1. Séparation en catégories
  2. etc.
