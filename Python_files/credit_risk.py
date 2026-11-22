import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. CHARGEMENT ET NETTOYAGE PRÉLIMINAIRE
# ==========================================
df = pd.read_csv('credit_risk_dataset.csv')
df = df.drop_duplicates()

X = df.drop(columns='loan_status')
y = df['loan_status']

plt.figure()
plt.subplot(1, 3, 1)
sns.boxplot(df['person_emp_length'])
plt.subplot(1, 3, 2)
sns.boxplot(df['person_age'])
plt.subplot(1, 3, 3)
sns.boxplot(df['loan_int_rate'])
plt.show()
# ==========================================
# 2. SÉPARATION TRAIN / TEST
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. GESTION DES VALEURS ABERRANTES
# ==========================================

# A. Ancienneté max 50 ans & Cohérence âge min 18 ans
for dataset in [X_train, X_test]:
    dataset.loc[dataset['person_emp_length']
                > 50, 'person_emp_length'] = np.nan
    dataset.loc[dataset['person_age'] > 85, 'person_age'] = np.nan
    dataset.loc[dataset['person_age'] < 18, 'person_age'] = np.nan

# B. Calcul des seuils d'outliers pour 'loan_int_rate' UNIQUEMENT sur X_train
qr_1 = X_train['loan_int_rate'].quantile(0.25)
qr_3 = X_train['loan_int_rate'].quantile(0.75)
iqr = qr_3 - qr_1
borne_supp = qr_3 + 1.5 * iqr

# Application du seuil calculé du train sur le train et le test
for dataset in [X_train, X_test]:
    dataset.loc[dataset['loan_int_rate'] >
                borne_supp, 'loan_int_rate'] = np.nan


# ==========================================
# 4. DÉFINITION DES GROUPES DE COLONNES POUR LE PIPELINE
# ==========================================
cols_imputer_median = ['person_age', 'person_emp_length', 'loan_int_rate']
cols_ordinales = ['loan_grade']
cols_nominales = ['person_home_ownership',
                  'loan_intent', 'cb_person_default_on_file']

ordre_grades = [['A', 'B', 'C', 'D', 'E', 'F', 'G']]


# ==========================================
# 5. CONSTRUCTION DU PREPROCESSOR
# ==========================================
preprocessor = ColumnTransformer(
    transformers=[
        # 1. Imputation par la médiane pour les colonnes numériques nettoyées
        ('imputer', SimpleImputer(strategy='median'), cols_imputer_median),

        # 2. Encodage ordinal pour loan_grade (avec ordre explicite)
        ('ordinal', OrdinalEncoder(categories=ordre_grades,
         handle_unknown='use_encoded_value', unknown_value=-1), cols_ordinales),

        # 3. One-Hot Encoding pour les variables nominales
        ('nominal', OneHotEncoder(drop='first', sparse_output=False,
         handle_unknown='ignore'), cols_nominales)
    ],
    # Conserve toutes les autres colonnes numériques intactes (ex: loan_amnt, income...)
    remainder='passthrough'
)


# ==========================================
# 6. CRÉATION DU PIPELINE FINAL (Prétraitement + Modèle)
# ==========================================
# Vous pouvez remplacer LogisticRegression
model_logit = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=3000, random_state=42, solver='lbfgs'))
])


# ==========================================
# 7. ENTRAÎNEMENT ET ÉVALUATION
# ==========================================
# Le pipeline s'occupe tout seul d'apprendre sur X_train et de transformer X_test proprement
model_logit.fit(X_train, y_train)

# Prédictions
logit_pred = model_logit.predict(X_test)

# Résultats
print("--- Évaluation du Modèle LOGIT ---")
print(
    f"Accuracy (Précision) : {accuracy_score(y_test, logit_pred) * 100:.2f}%\n")
print("Matrice de confusion :")
print(confusion_matrix(y_test, logit_pred))
print("\nRapport de classification :")
print(classification_report(y_test, logit_pred))

# ==========================================
# 8. DECISION TREE
# ==========================================
model_dt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])


# ==========================================
# 9. ENTRAÎNEMENT ET ÉVALUATION
# ==========================================
# Le pipeline s'occupe tout seul d'apprendre sur X_train et de transformer X_test proprement
model_dt.fit(X_train, y_train)

# Prédictions
dt_pred = model_dt.predict(X_test)

# Résultats
print("--- Évaluation du Modèle DECISION TREE ---")
print(f"Accuracy (Précision) : {accuracy_score(y_test, dt_pred) * 100:.2f}%\n")
print("Matrice de confusion :")
print(confusion_matrix(y_test, dt_pred))
print("\nRapport de classification :")
print(classification_report(y_test, dt_pred))

# ==========================================
# 10. RANDOMN FOREST
# ==========================================
model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])


# ==========================================
# 11. ENTRAÎNEMENT ET ÉVALUATION
# ==========================================
# Le pipeline s'occupe tout seul d'apprendre sur X_train et de transformer X_test proprement
model_rf.fit(X_train, y_train)

# Prédictions
rf_pred = model_rf.predict(X_test)

# Résultats
print("--- Évaluation du Modèle RANDOM FOREST ---")
print(f"Accuracy (Précision) : {accuracy_score(y_test, rf_pred) * 100:.2f}%\n")
print("Matrice de confusion :")
print(confusion_matrix(y_test, rf_pred))
print("\nRapport de classification :")
print(classification_report(y_test, rf_pred))

# ==========================================
# 12. XGBoost
# ==========================================
model_xg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('Classifier', XGBClassifier())
])


# ==========================================
# 13. ENTRAÎNEMENT ET ÉVALUATION
# ==========================================
# Le pipeline s'occupe tout seul d'apprendre sur X_train et de transformer X_test proprement
model_xg.fit(X_train, y_train)

# Prédictions
xg_pred = model_xg.predict(X_test)


# Résultats
print("--- Évaluation du Modèle XGBOOST ---")
print(f"Accuracy (Précision) : {accuracy_score(y_test, xg_pred) * 100:.2f}%\n")
print("Matrice de confusion :")
print(confusion_matrix(y_test, xg_pred))
print("\nRapport de classification :")
print(classification_report(y_test, xg_pred))

# SAUVEGARDE DU MODELE
joblib.dump(model_xg, 'credit_risk_model')

# COURBE_ROC_AUC
# 1. Définition de vos 4 modèles (supposons qu'ils soient déjà initialisés)
# Vous pouvez remplacer ces modèles par vos propres instances entraînées
models = {
    "Régression Logistique": model_logit,
    "Random Forest": model_rf,
    "XGBoost": model_xg,
    "Arbre de Décision": model_dt
}

# 2. Création de la figure pour le graphique comparatif
plt.figure(figsize=(9, 7))

# 3. Boucle sur chaque modèle pour tracer sa courbe ROC
for name, model in models.items():
    # Récupération des probabilités de la classe positive (risque = 1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calcul du score AUC
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"Score ROC-AUC ({name}) : {auc_score:.4f}")

    # Calcul des points de la courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Tracé de la courbe pour ce modèle
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.2f})')

# 4. Ligne diagonale de référence (hasard parfait = 0.5)
plt.plot([0, 1], [0, 1], color='gray', lw=2,
         linestyle='--', label='Hasard (AUC = 0.5)')

# 5. Mise en forme du graphique
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=12)
plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=12)
plt.title('Comparaison des Courbes ROC - Modèles de Credit Scoring', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.show()

# MATRICE DE CONFUSION GRAPHE
# 2. Calcul de la matrice de confusion
cm = confusion_matrix(y_test, xg_pred)

# 3. Affichage graphique propre
plt.figure(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                              'Bon Payeur (0)', 'Défaut (1)'])
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
plt.title("Matrice de Confusion - XGBoost", fontsize=14)
plt.grid(False)  # Désactive la grille pour garder un affichage net
plt.show()

model = joblib.load('credit_risk_model')

# Si votre modèle est un pipeline, extrayez le Random Forest (dernière étape) :
rf = model[-1]

# ==========================================
# 3. IMPORTANCE DES VARIABLES
# ==========================================
# Récupération propre des noms de colonnes (via le préprocesseur ou les colonnes du DataFrame)
try:
    feature_names = model[0].get_feature_names_out()
except Exception:
    feature_names = X_train.columns

importances = rf.feature_importances_
df_imp = pd.DataFrame({'Variable': feature_names, 'Importance': importances})
df_imp = df_imp.sort_values(by='Importance', ascending=True).tail(10)  # Top 10

plt.figure(figsize=(10, 6))
plt.barh(df_imp['Variable'], df_imp['Importance'], color='forestgreen')
plt.xlabel("Score d'importance", fontsize=12)
plt.ylabel("Variables", fontsize=12)
plt.title("Top 10 - Importance des variables (XGBOOST)", fontsize=14)
plt.grid(axis='x', alpha=0.3)
plt.show()


# ==========================================
# 4. VISUALISATION D'UN ARBRE DE LA FORÊT
# ==========================================
plt.figure(figsize=(20, 10))
plot_tree(
    rf.estimators_[0],          # Affiche le 1er arbre de la forêt
    feature_names=feature_names,  # Vrais noms des colonnes
    class_names=["Bon (0)", "Défaut (1)"],
    filled=True,
    max_depth=3,                # Limite la profondeur pour garder le graphe lisible
    fontsize=10
)
plt.title("Exemple d'un arbre de décision XGBOOST", fontsize=16)
plt.show()
