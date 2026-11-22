import streamlit as st
import joblib
import pandas as pd


@st.cache_resource
def charge_model():
    print('chargement du modele en cours ...')
    credit_scoring = joblib.load('credit_risk_model')
    return credit_scoring


model = charge_model()

st.title("Credit scoring application")
st.header("analyse des credit")

age = st.number_input(label="age", min_value=18, max_value=90)
salaire_annuel = st.slider("salaire", min_value=9000, max_value=900000)
home_ownership = st.selectbox(
    'ownership', options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
annciente = st.slider("annciennte", min_value=0, max_value=50)
type_de_credit = st.selectbox('type de credit', options=[
    'EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
grade_de_credit = st.selectbox('grade', options=[
    'A', 'B', 'C', 'D', 'E', 'F', 'G'])
montant_credit = st.number_input(
    label='montant de credit', min_value=1000, max_value=50000)
taux_int = st.slider(
    'taux interet', min_value=0.0, max_value=30.0)
taux_credit_salaire = st.slider(
    'taux (credit/salaire)', min_value=0.0, max_value=1.0)
defaut_paiment = st.selectbox("defaut de paiment", options=["Y", "N"])
long_credit = st.slider(
    'longueur du credit', min_value=0, max_value=10)

if st.button("prediction du risque ..."):
    df = pd.DataFrame({
        'person_age': [age],
        'person_income': [salaire_annuel],
        'person_home_ownership': [home_ownership],
        'person_emp_length': [annciente],
        'loan_intent': [type_de_credit],
        'loan_grade': [grade_de_credit],
        'loan_amnt': [montant_credit],
        'loan_int_rate': [taux_int],
        'loan_percent_income': [taux_credit_salaire],
        'cb_person_default_on_file': [defaut_paiment],
        'cb_person_cred_hist_length': [long_credit]
    })

    prediction = model.predict(df)
    probabilite = model.predict_proba(df)

    if prediction[0] == 1:
        st.error(
            f"⚠️ Risque élevé de défaut ! (Probabilité : {probabilite[0][1]*100:.2f}%)")
    else:
        st.success(
            f"✅ Risque faible - Crédit accordé. (Probabilité de défaut : {probabilite[0][1]*100:.2f}%)")
