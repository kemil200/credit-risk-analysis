import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Configuration de la page
st.set_page_config(page_title="ScoreCredit Pro", layout="wide")

st.title("ScoreCredit : Outil d'analyse de risque credit")

# 1. Barre latérale : Paramètres de Pondération du modèle
st.sidebar.header("Paramètres de Scoring")
poids_endettement = st.sidebar.slider("Poids du Risque (Endettement)", 0.0, 1.0, 0.6)
poids_fidelite = st.sidebar.slider("Poids de la Fidélité", 0.0, 1.0, 0.4)

# 2. Simulation de données (Logique de calcul)
def calculer_score(montant, revenu, anciennete, duree):
    endettement = (montant / duree) / revenu
    fidelite = anciennete / duree
    # Score composite (plus il est bas, meilleur c'est)
    score = (endettement * poids_endettement) - (fidelite * poids_fidelite)
    return score, endettement

# 3. Interface principale
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Simulateur Client")
    montant = st.number_input("Montant demandé (CFA)", 500000, 10000000, 2000000)
    revenu = st.number_input("Revenu Mensuel (CFA)", 150000, 5000000, 500000)
    anciennete = st.slider("Ancienneté (mois)", 1, 120, 24)
    duree = st.selectbox("Durée (mois)", [6, 12, 18, 24])

    if st.button("Calculer le Score"):
        score, endettement = calculer_score(montant, revenu, anciennete, duree)
        st.metric("Score de Risque", f"{score:.2f}")
        if score > 0.3:
            st.error("⚠️ Dossier à risque élevé.")
        else:
            st.success("✅ Dossier éligible.")

with col2:
    st.subheader("Analyse de Sensibilité")
    # Simulation d'un graphique interactif
    data = pd.DataFrame({
        'Revenu': np.linspace(200000, 2000000, 10),
        'Risque': [x * np.random.rand() for x in range(10)]
    })
    fig = px.line(data, x='Revenu', y='Risque', title="Évolution du risque selon le revenu")
    st.plotly_chart(fig, use_container_width=True)
