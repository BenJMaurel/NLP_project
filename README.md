# Safe AI: Toward Safer Systems Against Data Drift

Projet pour le cours **Machine Learning for Natural Language Processing** de l'ENSAE.

* Louise Demoor
* Benjamin Maurel

## Introduction

Notre rapport est disponible [ici]() à la racine du projet.

### Exemples
Voici un lien vers un colab où l'on peut reproduire les graphes utilisés dans notre projet.



## Installation

Avant toutes choses, il faut se déplacer dans le dossier `package`:

```bash
cd /path/to/package
```

### Avec [`poetry`](https://python-poetry.org/)

```bash
# Creation d'un environement virtuel et installation des packages
poetry install

# Activation de l'environement
poetry shell  # sub shell
# OU source $(poetry env info --path)/bin/activate  # activer l'environement dans le shell actuel
```

### Avec `pip`

```bash
# Creation d'un environement virtuel
python -m venv .venv

# Activation de l'environement
.venv/Script/activate  # pour Windows
# OU source .venv/bin/activate  # pour Linux / MacOS

# Installation des packages
pip install -r requirements.txt
```

### Processing des données

Nous avons mis directement les fichiers .npy correspondant aux extractions des features de chaque couches du modèle ROBERTA-base. Vous pouvez les retrouver dans "/embeding/", on retrouve les embedding de sst-2 train, sst-2 test, news20, trec et wm16.
