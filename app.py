import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import softmax

st.set_page_config(page_title="ScoreCredit", layout="centered")

st.title("ScoreCredit")
st.caption("Analyse de risque crédit · Modèle fairness-aware · Sans âge ni sexe")
st.divider()

# Modele multinomial fairness-aware
# Classes  : 0=EN COURS | 1=EN RETARD | 2=REMBOURSE
# Features : log_montant | taux | duree | charge_interet | cout_total_ratio
#            act_Artisanat | act_Commerce
#            reg_Kara | reg_Maritime | reg_Plateau | reg_Savane
#            cat_Moyen | cat_Petit
CLASSES   = ["EN COURS", "EN RETARD", "REMBOURSE"]
COEFS     = np.array([
    [-0.1279, -0.0875,  0.0845, -0.0335, -0.0335, -0.0391,  0.0647,  0.0115, -0.0813,  0.0322, -0.1659, -0.0124, -0.1817],
    [ 0.2095,  0.1478,  0.1292, -0.0952, -0.0952,  0.1505, -0.0210,  0.0328, -0.0887, -0.0251, -0.0271,  0.0714,  0.1386],
    [-0.0816, -0.0603, -0.2137,  0.1288,  0.1288, -0.1114, -0.0437, -0.0444,  0.1700, -0.0071,  0.1930, -0.0591,  0.0431],
])
INTERCEPT = np.array([0.0103, 0.0071, -0.0174])
MEAN      = np.array([12.289189, 0.112, 19.608, 0.18439, 1.18439, 0.35, 0.346, 0.112, 0.234, 0.238, 0.252, 0.51, 0.264])
SCALE     = np.array([1.012749, 0.025706, 10.39261, 0.111429, 0.111429, 0.47697, 0.475693, 0.315366, 0.423372, 0.425859, 0.434161, 0.4999, 0.440799])

def mensualite(montant, duree, taux):
    r = taux / 12
    if r < 1e-9: return montant / duree
    return montant * (r * (1+r)**duree) / ((1+r)**duree - 1)

def get_categorie(montant):
    if montant <= 100_000: return "Petit"
    elif montant <= 500_000: return "Moyen"
    else: return "Grand"

def predict(montant, duree, taux, activite, region):
    cat = get_categorie(montant)
    x = np.array([
        np.log(montant), taux, duree,
        taux,                    # charge_interet = taux annuel
        1 + taux,                # cout_total_ratio
        1 if activite == "Artisanat" else 0,
        1 if activite == "Commerce"  else 0,
        1 if region == "Kara"        else 0,
        1 if region == "Maritime"    else 0,
        1 if region == "Plateau"     else 0,
        1 if region == "Savane"      else 0,
        1 if cat == "Moyen"          else 0,
        1 if cat == "Petit"          else 0,
    ])
    x_sc = (x - MEAN) / SCALE
    return softmax(INTERCEPT + COEFS @ x_sc)

# Saisie
col1, col2 = st.columns(2)
with col1:
    montant  = st.number_input("Montant demandé (CFA)", 50_000, 1_000_000, 300_000, 25_000)
    taux_pct = st.slider("Taux d'intérêt annuel (%)", 8.0, 15.0, 12.0, 0.5)
    duree    = st.selectbox("Durée (mois)", [6, 12, 18, 24, 36])
with col2:
    activite = st.selectbox("Activité", ["Commerce", "Artisanat", "Agriculture"])
    region   = st.selectbox("Région", ["Maritime", "Savane", "Plateau", "Kara", "Centrale"])

# Calcul
taux        = taux_pct / 100
mens        = mensualite(montant, duree, taux)
probs       = predict(montant, duree, taux, activite, region)
p_cours, p_retard, p_rembourse = probs
classe      = CLASSES[np.argmax(probs)]

st.divider()

# Metriques
m1, m2, m3 = st.columns(3)
m1.metric("Statut prédit", classe)
m2.metric("Mensualité", f"{mens:,.0f} CFA")
m3.metric("Coût total", f"{mens*duree:,.0f} CFA")

# Verdict
if classe == "REMBOURSE":
    st.success(f"✅ Dossier ÉLIGIBLE — Probabilité de remboursement : {p_rembourse:.1%}")
elif classe == "EN COURS":
    st.info(f"🕐 Dossier EN COURS — À surveiller ({p_cours:.1%})")
else:
    st.error(f"⛔ Dossier À RISQUE — Probabilité de retard : {p_retard:.1%}")

st.divider()

# Graphique 3 courbes
montants = np.linspace(50_000, 1_000_000, 200)
p_ret = [predict(m, duree, taux, activite, region)[1] for m in montants]
p_rem = [predict(m, duree, taux, activite, region)[2] for m in montants]
p_cou = [predict(m, duree, taux, activite, region)[0] for m in montants]

fig = go.Figure()
fig.add_trace(go.Scatter(x=montants, y=p_ret, mode="lines", line=dict(color="#f85149", width=2), name="EN RETARD"))
fig.add_trace(go.Scatter(x=montants, y=p_rem, mode="lines", line=dict(color="#3fb950", width=2), name="REMBOURSE"))
fig.add_trace(go.Scatter(x=montants, y=p_cou, mode="lines", line=dict(color="#58a6ff", width=1.5, dash="dot"), name="EN COURS"))
fig.add_trace(go.Scatter(
    x=[montant], y=[probs[np.argmax(probs)]],
    mode="markers",
    marker=dict(size=12, color={"EN RETARD":"#f85149","REMBOURSE":"#3fb950","EN COURS":"#58a6ff"}[classe],
                symbol="circle", line=dict(color="white", width=2)),
    name="Dossier actuel"
))
fig.update_layout(
    title=f"Probabilités par statut selon le montant ({taux_pct:.1f}% · {duree} mois · {activite})",
    xaxis_title="Montant (CFA)",
    yaxis=dict(title="Probabilité", tickformat=".0%", range=[0, 1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=50, b=40), height=350
)
st.plotly_chart(fig, use_container_width=True)


