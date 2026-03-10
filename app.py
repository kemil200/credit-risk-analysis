import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import softmax

st.set_page_config(page_title="ScoreCredit", layout="centered")
st.title("ScoreCredit")
st.caption("Analyse risque et credit")
st.divider()

# ── Modele multinomial entraine sur Jeux_donnees.csv ─────────────────────────
# Classes : 0=EN COURS  |  1=EN RETARD  |  2=REMBOURSE
# Features: log_montant | Duree | taux | Age | sexe_bin | act_Artisanat | act_Commerce
CLASSES   = ["EN COURS", "EN RETARD", "REMBOURSE"]
COEFS     = np.array([
    [ 0.0003,  0.0278, -0.0944, -0.0446, -0.0079, -0.0426,  0.0532],  # EN COURS
    [ 0.1210, -0.0300,  0.0825,  0.1359,  0.0049,  0.1671, -0.0038],  # EN RETARD
    [-0.1213,  0.0022,  0.0119, -0.0913,  0.0030, -0.1245, -0.0494],  # REMBOURSE
])
INTERCEPT = np.array([0.0106, -0.0125, 0.0019])
MEAN      = np.array([12.289189, 19.608, 0.112, 44.198, 0.532, 0.35, 0.346])
SCALE     = np.array([1.012749, 10.39261, 0.025706, 14.442673, 0.498975, 0.47697, 0.475693])

def mensualite_reelle(montant, duree, taux_annuel):
    r = taux_annuel / 12
    if r < 1e-9:
        return montant / duree
    return montant * (r * (1 + r) ** duree) / ((1 + r) ** duree - 1)

def predict_multinomial(montant, duree, taux_annuel, age, sexe, activite):
    sexe_bin      = 1 if sexe == "Homme" else 0
    act_artisanat = 1 if activite == "Artisanat" else 0
    act_commerce  = 1 if activite == "Commerce" else 0

    x        = np.array([np.log(montant), duree, taux_annuel, age,
                          sexe_bin, act_artisanat, act_commerce])
    x_scaled = (x - MEAN) / SCALE
    logits   = INTERCEPT + COEFS @ x_scaled
    probs    = softmax(logits)          # softmax -> somme = 1
    return probs  # [P(EN COURS), P(EN RETARD), P(REMBOURSE)]

# ── Saisie ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    montant  = st.number_input("Montant demande (CFA)", 50_000, 10_000_000, 300_000, 25_000)
    taux_pct = st.slider("Taux d'interet annuel (%)", 5.0, 20.0, 12.0, 0.5)
    age      = st.slider("Age du client", 18, 80, 40)
with col2:
    duree    = st.selectbox("Duree du credit (mois)", [6, 12, 18, 24, 36])
    activite = st.selectbox("Activite", ["Commerce", "Artisanat", "Agriculture"])
    sexe     = st.selectbox("Sexe", ["Homme", "Femme"])

taux_annuel = taux_pct / 100
mens        = mensualite_reelle(montant, duree, taux_annuel)

# ── Calcul ────────────────────────────────────────────────────────────────────
probs         = predict_multinomial(montant, duree, taux_annuel, age, sexe, activite)
p_en_cours    = probs[0]
p_en_retard   = probs[1]
p_rembourse   = probs[2]
classe_predite = CLASSES[np.argmax(probs)]

st.divider()

# ── Metriques ─────────────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("Mensualite estimee", f"{mens:,.0f} CFA")
m2.metric("Cout total credit",  f"{mens * duree:,.0f} CFA")
m3.metric("Prediction",         classe_predite)

# ── Probabilites par classe ───────────────────────────────────────────────────
st.markdown("##### Probabilites par statut")
c1, c2, c3 = st.columns(3)
c1.metric("EN COURS",    f"{p_en_cours:.1%}",  help="Credit en cours de remboursement")
c2.metric("EN RETARD",   f"{p_en_retard:.1%}", help="Risque de retard de paiement")
c3.metric("REMBOURSE",   f"{p_rembourse:.1%}", help="Credit entierement rembourse")

# ── Verdict ───────────────────────────────────────────────────────────────────
if classe_predite == "REMBOURSE":
    st.success(f"Dossier ELIGIBLE — Le profil ressemble le plus a un client qui rembourse ({p_rembourse:.1%})")
elif classe_predite == "EN COURS":
    st.info(f"Dossier EN COURS — Remboursement probable mais non confirme ({p_en_cours:.1%})")
else:
    st.error(f"Dossier A RISQUE — Le profil ressemble le plus a un client en retard ({p_en_retard:.1%})")

st.divider()

# ── Graphique : P(EN RETARD) vs Montant ──────────────────────────────────────
montants   = np.linspace(50_000, 2_000_000, 300)
p_retards  = [predict_multinomial(m, duree, taux_annuel, age, sexe, activite)[1] for m in montants]
p_rembours = [predict_multinomial(m, duree, taux_annuel, age, sexe, activite)[2] for m in montants]
p_cours    = [predict_multinomial(m, duree, taux_annuel, age, sexe, activite)[0] for m in montants]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=montants, y=p_retards,
    mode="lines", line=dict(color="#f85149", width=2),
    name="P(EN RETARD)"
))
fig.add_trace(go.Scatter(
    x=montants, y=p_rembours,
    mode="lines", line=dict(color="#3fb950", width=2),
    name="P(REMBOURSE)"
))
fig.add_trace(go.Scatter(
    x=montants, y=p_cours,
    mode="lines", line=dict(color="#58a6ff", width=2, dash="dot"),
    name="P(EN COURS)"
))

# Point client actuel
colors = {"EN COURS": "#58a6ff", "EN RETARD": "#f85149", "REMBOURSE": "#3fb950"}
fig.add_trace(go.Scatter(
    x=[montant], y=[p_en_retard],
    mode="markers",
    marker=dict(color=colors[classe_predite], size=14, symbol="circle",
                line=dict(color="white", width=2)),
    name="Dossier actuel (P retard)"
))

fig.update_layout(
    title=f"Probabilites par statut selon le montant  ({taux_pct:.1f}% · {duree} mois · {activite})",
    xaxis_title="Montant demande (CFA)",
    yaxis=dict(title="Probabilite", tickformat=".0%", range=[0, 1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=60, b=40), height=400
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Modele : Regression Logistique Multinomiale (softmax) · "
    "Entraine sur Jeux_donnees.csv (500 dossiers) · "
    "Variables : montant, duree, taux, age, sexe, activite · Accuracy : 41%"
)
