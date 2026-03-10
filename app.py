import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="ScoreCredit", layout="centered")
st.title("ScoreCredit")
st.caption("Analyse risque et crédit")
st.divider()

# ── Modele entraine sur Jeux_donnees.csv ──────────────────────────────────────
# Features : log_montant | Duree | taux | age | sexe_bin | act_Artisanat | act_Commerce
# Cible    : EN RETARD = 1, REMBOURSE / EN COURS = 0
COEF      = np.array([0.1775, -0.0344, 0.1449, 0.2033, 0.0041, 0.2477, -0.0191])
INTERCEPT = -0.0249
MEAN      = np.array([12.289189, 19.608, 0.112, 44.198, 0.532, 0.35, 0.346])
SCALE     = np.array([1.012749, 10.39261, 0.025706, 14.442673, 0.498975, 0.47697, 0.475693])

def mensualite_reelle(montant, duree, taux_annuel):
    r = taux_annuel / 12
    if r < 1e-9:
        return montant / duree
    return montant * (r * (1 + r) ** duree) / ((1 + r) ** duree - 1)

def predict(montant, duree, taux_annuel, age, sexe, activite):
    sexe_bin      = 1 if sexe == "Homme" else 0
    act_artisanat = 1 if activite == "Artisanat" else 0
    act_commerce  = 1 if activite == "Commerce" else 0

    x        = np.array([np.log(montant), duree, taux_annuel, age,
                          sexe_bin, act_artisanat, act_commerce])
    x_scaled = (x - MEAN) / SCALE
    logit    = INTERCEPT + np.dot(COEF, x_scaled)
    prob     = 1 / (1 + np.exp(-logit))
    return float(prob)

# ── Saisie ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    montant   = st.number_input("Montant demande (CFA)", 50_000, 10_000_000, 300_000, 25_000)
    taux_pct  = st.slider("Taux d'interet annuel (%)", 5.0, 20.0, 12.0, 0.5)
    age       = st.slider("Age du client", 18, 80, 40)
with col2:
    duree     = st.selectbox("Duree du credit (mois)", [6, 12, 18, 24, 36])
    activite  = st.selectbox("Activite", ["Commerce", "Artisanat", "Agriculture"])
    sexe      = st.selectbox("Sexe", ["Homme", "Femme"])

taux_annuel = taux_pct / 100
mens        = mensualite_reelle(montant, duree, taux_annuel)

# ── Calcul ────────────────────────────────────────────────────────────────────
prob     = predict(montant, duree, taux_annuel, age, sexe, activite)
solvable = prob < 0.33   # seuil cale sur le taux de defaut observe (33%)

st.divider()

# ── Metriques ─────────────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("Probabilite de defaut", f"{prob:.1%}")
m2.metric("Mensualite estimee", f"{mens:,.0f} CFA",
          help="Amortissement constant avec le taux d'interet saisi")
m3.metric("Cout total du credit", f"{mens * duree:,.0f} CFA")

# ── Verdict ───────────────────────────────────────────────────────────────────
if solvable:
    st.success(f"Dossier ELIGIBLE — Risque de defaut faible ({prob:.1%})")
elif prob < 0.55:
    st.warning(f"Dossier a surveiller — Risque modere ({prob:.1%}), analyse complementaire recommandee.")
else:
    st.error(f"Dossier REJETE — Risque de defaut eleve ({prob:.1%})")

st.divider()

# ── Graphique : P(defaut) vs Montant ─────────────────────────────────────────
montants = np.linspace(50_000, 2_000_000, 300)
risques  = [predict(m, duree, taux_annuel, age, sexe, activite) for m in montants]

fig = go.Figure()

fig.add_hrect(y0=0,    y1=0.33, fillcolor="rgba(63,185,80,0.07)",  line_width=0)
fig.add_hrect(y0=0.33, y1=0.55, fillcolor="rgba(210,153,34,0.07)", line_width=0)
fig.add_hrect(y0=0.55, y1=1.0,  fillcolor="rgba(248,81,73,0.07)",  line_width=0)

fig.add_trace(go.Scatter(
    x=montants, y=risques,
    mode="lines", line=dict(color="#58a6ff", width=2.5),
    fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    name="P(Defaut)"
))

fig.add_hline(y=0.33, line_dash="dot", line_color="#3fb950",
              annotation_text="Seuil eligible (33%)", annotation_font_color="#3fb950",
              annotation_position="top right")
fig.add_hline(y=0.55, line_dash="dot", line_color="#d29922",
              annotation_text="Seuil rejet (55%)", annotation_font_color="#d29922",
              annotation_position="top right")

point_color = "#3fb950" if solvable else ("#d29922" if prob < 0.55 else "#f85149")
fig.add_trace(go.Scatter(
    x=[montant], y=[prob],
    mode="markers",
    marker=dict(color=point_color, size=14, symbol="circle",
                line=dict(color="white", width=2)),
    name="Dossier actuel"
))

fig.update_layout(
    title=f"Probabilite de defaut selon le montant  (taux {taux_pct:.1f}% · {duree} mois · {activite})",
    xaxis_title="Montant demande (CFA)",
    yaxis=dict(title="P(Defaut)", tickformat=".0%", range=[0, 1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=60, b=40), height=380,
    annotations=[
        dict(x=1_700_000, y=0.16,  text="Eligible", showarrow=False, font=dict(color="#3fb950", size=11)),
        dict(x=1_700_000, y=0.44,  text="Modere",   showarrow=False, font=dict(color="#d29922", size=11)),
        dict(x=1_700_000, y=0.75,  text="Rejete",   showarrow=False, font=dict(color="#f85149", size=11)),
    ]
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Modele : Regression Logistique · Entraine sur Jeux_donnees.csv (500 dossiers) · "
    "Variables : montant, duree, taux, age, sexe, activite"
)
