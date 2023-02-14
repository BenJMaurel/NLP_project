# Safe AI: Toward Safer Systems Against Data Drift

Projet pour le cours **Machine Learning for Natural Language Processing** de l'ENSAE.

* Louise Demoor
* Benjamin Maurel

## Introduction

Notre rapport est disponible [ici]() à la racine du projet.

### Le package


### Les notebooks


### Le reste


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

Pour générer les données à partir du fichier brut, il faut lancer le script `process_data.py` :

```python
python smc_movement_models/process_data.py
# Pour plus d'informations
# python smc_movement_models/process_data.py --help
```

### `pre-commit`

Pour activer les pre-commit qui formattent le code avant chaque commit :

```bash
pre-commit install
pre-commit run --all-files  # Pour installer les modules et lancer les tests
```

![Exemple de pre-commit](images/pre-commit-exemple.png)

Pour forcer un commit sans vérifier :

```bash
git commit -m "..." --no-verify
```
