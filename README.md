# ML_Project

## Prédiction de Prix Boursiers et Backtesting de Stratégies Trading

Ce projet vise à prédire le prix de clôture futur d’actions (avec un focus sur AAPL, MSFT, et JPM) en utilisant plusieurs techniques de machine learning et deep learning. Il compare principalement deux approches :

- **Modèle de Régression Linéaire**
- **Réseau de Neurones LSTM (Long Short-Term Memory)**

En complément, un backtesting simplifié d’une stratégie de trading basée sur ces prédictions est réalisé afin d’évaluer leur potentiel en conditions de marché réelles.

---

## Table des Matières

- [Contexte et Objectifs](#contexte-et-objectifs)
- [Acquisition et Prétraitement des Données](#acquisition-et-prétraitement-des-données)
- [Calcul des Indicateurs Techniques](#calcul-des-indicateurs-techniques)
- [Modélisation](#modélisation)
  - [Régression Linéaire](#régression-linéaire)
  - [Modèle LSTM](#modèle-lstm)
- [Backtesting et Évaluation](#backtesting-et-évaluation)
- [Résultats et Analyse](#résultats-et-analyse)
- [Installation et Exécution](#installation-et-exécution)
- [Perspectives et Améliorations Futures](#perspectives-et-améliorations-futures)
- [Licence](#licence)

---

## Contexte et Objectifs

Ce projet a pour but de comparer la capacité prédictive d’un modèle de régression linéaire et d’un réseau de neurones LSTM sur des données boursières historiques (de 2013 à 2023). En outre, nous évaluons l’application pratique de ces modèles dans une stratégie de trading simple via un backtest.

**Les objectifs sont :**

- Prétraiter les données boursières et calculer divers indicateurs techniques (RSI, MACD, bandes de Bollinger, ATR, etc.) ainsi que des indicateurs financiers (P/E Ratio, MarketSentiment dérivé du VIX).
- Construire et comparer des modèles de prédiction :
  - Une régression linéaire (baseline simple et performante pour J+1).
  - Un LSTM, capable de capturer les dépendances temporelles dans les séries chronologiques.
- Backtester une stratégie de trading basée sur les signaux générés par ces modèles pour évaluer leur applicabilité pratique.

---

## Acquisition et Prétraitement des Données

Les données historiques sont récupérées via **yfinance** pour plusieurs tickers (AAPL, MSFT, JPM) couvrant la période 2013–2023.

### Pipeline d'Acquisition et Enrichissement

- **Téléchargement des données** avec yfinance.
- **Calcul des Indicateurs Financiers :**
  - **VIX** : Téléchargé et transformé en un indice de MarketSentiment (l’opposé du z-score du VIX).
  - **P/E Ratio** dynamique calculé via l’API de yfinance.
- **Feature Engineering :**
  - Calcul de divers indicateurs techniques tels que RSI, MACD (et sa ligne de signal), bandes de Bollinger (MA20, Upper_BB, Lower_BB), ATR.
- **Création de la cible :**
  - La colonne `Target` est définie comme `df['Close'].shift(-1)`, c'est-à-dire la valeur de clôture du jour suivant (J+1).

---

## Calcul des Indicateurs Techniques

Les fonctions de calcul suivantes sont utilisées pour enrichir le DataFrame :

- **RSI** : Calcul du Relative Strength Index.
- **MACD et Signal_MACD** : Calcul du Moving Average Convergence Divergence et de sa ligne de signal.
- **Bandes de Bollinger** : Calcul de MA20, Upper_BB et Lower_BB.
- **ATR** : Calcul de l’Average True Range.

> **Remarque :** Ces indicateurs sont ajoutés au DataFrame avant le nettoyage des valeurs manquantes (`dropna`).

---

## Modélisation

Les modèles sont entraînés sur les mêmes données (mêmes features, même split 80/20 chronologique) afin d’assurer une comparaison équitable.

### Régression Linéaire

Un modèle de régression linéaire simple est utilisé pour prédire le prix de clôture du jour suivant.

**Étapes :**

- Extraction des features et de la cible.
- Split 80/20 sans mélange (pour préserver l’ordre chronologique).
- Scaling des données (MinMaxScaler).
- Entraînement du modèle.
- Évaluation via le calcul du MSE.
- Visualisation des courbes réelles vs prédictions.

### Modèle LSTM

Un réseau de neurones LSTM est utilisé pour prédire le prix du jour suivant en capturant la dynamique temporelle via une fenêtre de 20 jours.

**Pipeline LSTM :**

- Split 80/20 identique à la régression.
- Scaling des features et de la cible.
- Création des séquences : Transformation des données en séquences de 20 jours pour prédire la valeur au jour suivant.
- Construction du modèle LSTM avec :
  - Deux couches LSTM (128 et 64 neurones)
  - Dropout pour la régularisation
  - Une couche Dense de sortie
- Entraînement avec **EarlyStopping** et **ReduceLROnPlateau** pour optimiser la convergence.
- Évaluation via le calcul du MSE et visualisation des prédictions.

---

## Backtesting et Évaluation

Un backtest simplifié a été implémenté pour mesurer la performance de trading basée sur les signaux générés par les modèles.

### Stratégie de Trading Simple

- **Signal :**
  - Si la prédiction du modèle (J+1) est supérieure au prix du jour précédent, un signal d'achat (1) est généré.
  - Sinon, aucune position n’est prise (0).
- **Rendement :**
  - Calcul du rendement journalier comme la variation en pourcentage du prix.
  - Multiplication du rendement par le signal pour simuler l'entrée en position.
- **Performance :**
  - Simulation d’un portefeuille virtuel en multipliant le capital initial par le produit cumulatif des rendements journaliers.
  - Comparaison avec une stratégie "Buy & Hold".

Le code de backtesting inclut également le calcul de métriques telles que le rendement cumulé et le ratio de Sharpe.

---

## Résultats et Analyse

Pour la prédiction sur un horizon J+1 :

- **Régression Linéaire :**  
  Atteint un MSE très faible (ex. ~8–9), confirmant que le prix de clôture du jour suivant est fortement corrélé avec le prix actuel.
- **Modèle LSTM :**  
  Affiche un MSE plus élevé sur le même horizon, indiquant que pour des horizons très courts, la simplicité du modèle linéaire se révèle souvent optimale.
- **Backtesting :**  
  La stratégie basée sur la régression linéaire surpasse celle basée sur le LSTM pour J+1.

> **Note :** Il est envisagé d’explorer des horizons plus longs (J+7, J+30) et d’intégrer davantage de features (données fondamentales, sentiment, etc.) pour voir si le LSTM peut apporter une plus-value.

---

## Installation et Exécution

### Prérequis

- **Python 3.7** ou supérieur
- **Bibliothèques :**
  - numpy
  - pandas
  - matplotlib
  - yfinance
  - scikit-learn
  - tensorflow
  - xgboost (optionnel)

### Installation

Utilisez `pip` pour installer les dépendances :

```bash
pip install numpy pandas matplotlib yfinance scikit-learn tensorflow xgboost
