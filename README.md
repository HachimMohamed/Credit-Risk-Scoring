# Credit-Risk-Scoring
Modélisation prédictive du risque de crédit (Python, Scikit-Learn, XGBoost). Prétraitement des données financières, benchmarking de 4 algorithmes de classification supervisée sur métriques métier (ROC-AUC, F1-Score) et mise en production d'un simulateur interactif via Streamlit.
 ====================================================================================================================================================================

 # 🏦 Prédiction du Risque de Crédit (Credit Scoring) & Déploiement Web

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

## 📌 Contexte et Objectif
L'évaluation du risque de défaut est un enjeu stratégique pour les institutions financières. Ce projet vise à développer un modèle de **Machine Learning supervisé** capable de prédire la probabilité de défaut de paiement d'un client, puis à le déployer sous forme d'une **application web interactive** destinée aux analystes crédit.

## 🏗️ Architecture du Projet
* `src/credit_risk.py` : Pipeline complet de nettoyage, analyse exploratoire (EDA), entraînement et comparaison des modèles.
* `app/application_web.py` : Application web interactive développée avec **Streamlit** permettant de simuler un profil client et d'obtenir un score de risque en temps réel.
* `data/` : Jeu de données utilisé pour l'entraînement et le test du modèle.
* `requirements.txt` : Dépendances nécessaires à l'exécution du projet.

## ⚙️ Méthodologie Data Science
1. **Préparation des données :** Traitement des valeurs manquantes, encodage des variables catégorielles et gestion du déséquilibre des classes.
2. **Modélisation & Comparaison :** Entraînement et évaluation de 4 algorithmes de classification :
   * Régression Logistique *(modèle de référence)*
   * Arbre de Décision (Decision Tree)
   * Random Forest
   * **XGBoost** *(modèle retenu pour ses performances)*
3. **Évaluation :** Comparaison basée sur les métriques métier (Accuracy, Précision, Rappel, F1-Score et courbe ROC-AUC) afin de minimiser les faux négatifs.

## 🖥️ Aperçu de l'Application (Streamlit)
![Démo Application](assets/demo_streamlit.png)

## 🚀 Installation et Exécution en local

1. **Cloner le dépôt :**
```bash
git clone [https://github.com/HachimMohamed/Credit-Risk-Scoring.git](https://github.com/HachimMohamed/Credit-Risk-Scoring.git)
cd Credit-Risk-Scoring
```

2. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

3. **Lancer l'application web :**
```bash
streamlit run app/application_web.py
```
