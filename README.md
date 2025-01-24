# Atelier Pratique : Détection d'Anomalies avec Python

## Description

Dans cet atelier, nous allons explorer les algorithmes de détection d’anomalies (également appelée détection d’outliers ou de nouveauté) en utilisant Python. Ces algorithmes permettent d’identifier des instances atypiques ou inhabituelles dans les données. Vous utiliserez plusieurs techniques supervisées et non supervisées, en vous basant sur les bibliothèques Python comme Scikit-learn, Pandas et Matplotlib.

## Prérequis

Python installé sur votre machine.

Jupyter Notebook pour exécuter les notebooks interactifs.

Les bibliothèques suivantes doivent être installées :

numpy

pandas

matplotlib

scikit-learn

Installez-les avec la commande suivante si elles ne sont pas disponibles :

pip install numpy pandas matplotlib scikit-learn

Lancement de Jupyter Notebook

Pour ouvrir un notebook, tapez la commande suivante dans votre terminal :

jupyter notebook

Cela ouvrira une interface dans votre navigateur. Créez un nouveau notebook Python et commencez par exécuter le code suivant dans une cellule :

import numpy as np
np.set_printoptions(threshold=10000, suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

## Jeux de Données

Mouse Dataset : Disponible ici.

Credit Card Fraud Dataset : Disponible sur Kaggle.

KDDCup99 Dataset : Disponible sur Scikit-learn ou sur Kaggle.

## Tâches à Réaliser

1. Détection d’Anomalies sur le Dataset "Mouse"

Ce jeu de données contient 500 instances représentées par deux variables x1 et x2. Les 10 dernières instances sont des outliers.

Tâches :

Téléchargez et analysez le fichier.

Représentez graphiquement les données.

Appliquez les techniques suivantes :

Isolation Forest

Local Outlier Factor

Proposez une méthode pour choisir le seuil de contamination.

Modifiez la représentation graphique pour visualiser les anomalies.

Comparez les résultats obtenus numériquement et visuellement.

2. Détection de Fraudes et d’Intrusions

a) Fraudes sur les Transactions de Cartes Bancaires

Ce jeu de données contient 284 807 transactions avec seulement 0.172 % de fraudes.

Tâches :

Préparez les données (exclure la variable Time).

Proposez une méthodologie pour comparer :

Deux approches supervisées classiques.

Trois approches gérant le déséquilibre des données.

Les approches Isolation Forest et Local Outlier Factor.

Votre méthodologie doit inclure :

Un protocole de comparaison (échantillonnage stratifié ou validation croisée stratifiée).

L’évaluation des performances avec des métriques adaptées.

La normalisation des données si nécessaire (StandardScaler, MinMaxScaler, RobustScaler).

Le traitement des données catégorielles (OneHotEncoder, OrdinalEncoder).

L’affichage des courbes ROC et Précision/Rappel.

Le choix des meilleurs hyperparamètres (nombre d’arbres pour Isolation Forest, voisins pour LOF).

Factorisation et automatisation du code.

b) Intrusions dans les Réseaux (KDDCup99 Dataset)

Tâches :

Appliquez votre méthodologie à ce dataset.

Comparez les approches Isolation Forest et Local Outlier Factor pour la détection de nouveautés.

## Notes

Référez-vous aux diapositives du cours sur Moodle pour plus de détails.

Consultez la documentation Scikit-learn pour explorer les algorithmes utilisés.

Réalisez cet atelier en factorisant votre code et en automatisant autant que possible les tâches pour une analyse reproductible et efficace.

## Env

- pip install -r requirements.txt (packages)
- Python 3.12.8 (version du Python utilisé)

