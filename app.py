import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import expit

st.set_page_config(page_title="ScoreCredit", layout="centered")
st.title("ScoreCredit")
st.caption("Analyse de risque crédit")
st.divider()

# ── Saisie ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    montant    = st.number_input("Montant demandé (CFA)", 500_000, 10_000_000, 2_000_000, 100_000)
    revenu     = st.number_input("Revenu mensuel (CFA)",  150_000,  5_000_000,   500_000,  50_000)
with col2:
    anciennete = st.slider("Ancienneté client (mois)", 1, 120, 24)
    duree      = st.selectbox("Durée du crédit (mois)", [6, 12, 18, 24, 36, 48, 60])

# ── Modèle logistique ─────────────────────────────────────────────────────────
def probabilite_defaut(montant, revenu, anciennete, duree):
    ratio_endettement = (montant / duree) / revenu
    ratio_fidelite    = anciennete / 120
    logit = -0.5 + (1.5 * ratio_endettement * 10) - (1.2 * ratio_fidelite * 5)
    return float(expit(logit))

prob = probabilite_defaut(montant, revenu, anciennete, duree)
solvable = prob < 0.5

# ── Verdict ───────────────────────────────────────────────────────────────────
st.divider()
if solvable:
    st.success(f"✅ Dossier ÉLIGIBLE — Probabilité de défaut : **{prob:.1%}**")
else:
    st.error(f"⛔ Dossier REJETÉ — Probabilité de défaut : **{prob:.1%}**")

# ── Graphique : courbe décroissante du risque selon le revenu ─────────────────
revenus   = np.linspace(50_000, 300_000_000, 200)
risques   = [probabilite_defaut(montant, r, anciennete, duree) for r in revenus]

fig = go.Figure()

# Zone colorée sous la courbe
fig.add_trace(go.Scatter(
    x=revenus, y=risques,
    fill="tozeroy",
    fillcolor="rgba(88,166,255,0.10)",
    line=dict(color="#58a6ff", width=2.5),
    name="P(Défaut)"
))

# Seuil de rejet
fig.add_hline(y=0.5, line_dash="dot", line_color="#f85149",
              annotation_text="Seuil de rejet (50%)",
              annotation_font_color="#f85149",
              annotation_position="top right")

# Position du client actuel
fig.add_trace(go.Scatter(
    x=[revenu], y=[prob],
    mode="markers",
    marker=dict(color="#3fb950" if solvable else "#f85149", size=14, symbol="circle"),
    name="Client actuel"
))

fig.update_layout(
    title="Risque de défaut selon le revenu (tendance décroissante)",
    xaxis_title="Revenu mensuel (CFA)",
    yaxis_title="Probabilité de défaut",
    yaxis=dict(tickformat=".0%", range=[0, 1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=60, b=40),
    height=380
)

st.plotly_chart(fig, use_container_width=True)
