import streamlit as st
import pandas as pd
import joblib

# Titre de l'app
st.title("📊 ScoreCredit : Outil d'Aide à la Décision")

# Barre latérale pour les entrées de données
st.sidebar.header("Paramètres du Dossier")
montant = st.sidebar.slider("Montant du prêt", 500000, 5000000, 1000000)
revenu = st.sidebar.number_input("Revenu Mensuel", 150000, 5000000, 500000)
secteur = st.sidebar.selectbox("Secteur", ["Agriculture", "Commerce", "Transport", "Services"])

# Calcul en temps réel
ratio_endettement = (montant / 12) / revenu

st.write(f"### Ratio d'endettement calculé : {ratio_endettement:.2f}")

# Simuler la prédiction (ici on appellerait le modèle entraîné)
if st.button("Évaluer le risque"):
    if ratio_endettement > 0.4:
        st.error("⚠️ Risque Élevé : Dossier à vérifier manuellement")
    else:
        st.success("✅ Risque Faible : Profil favorable")
