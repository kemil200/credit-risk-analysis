import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="ScoreCredit", layout="centered")
st.title("ScoreCredit")
st.caption("Modele d'analyse de risque de credit")
st.divider()

# Parametres du modele entraine (StandardScaler + LogisticRegression)
# Features : taux_endettement | log_anciennete | nb_incidents | log_capacite_nette | ratio_montant_revenu
COEF       = np.array([-0.6608, -0.3763, 0.8385, -1.3757, 0.1453])
INTERCEPT  = -2.926
MEAN       = np.array([0.626798, 1.644405, 0.4122, 1.223135, 0.45271])
SCALE      = np.array([0.536309, 0.603517, 0.63142, 0.70906, 0.476419])

def predict_proba(montant, revenu, anciennete, duree, charges, nb_incidents):
    mensualite           = montant / duree
    taux_endettement     = (mensualite + charges) / revenu
    log_anciennete       = np.log1p(anciennete / 12)
    capacite_nette       = revenu - charges - mensualite
    log_capacite_nette   = np.log1p(max(capacite_nette, 0) / 100_000)
    ratio_montant_revenu = montant / (revenu * 12)

    x        = np.array([taux_endettement, log_anciennete, nb_incidents,
                          log_capacite_nette, ratio_montant_revenu])
    x_scaled = (x - MEAN) / SCALE
    logit    = INTERCEPT + np.dot(COEF, x_scaled)
    prob     = 1 / (1 + np.exp(-logit))
    return float(prob), float(taux_endettement), float(capacite_nette)

# Saisie
col1, col2 = st.columns(2)
with col1:
    montant      = st.number_input("Montant demande (CFA)", 200_000, 10_000_000, 2_000_000, 100_000)
    revenu       = st.number_input("Revenu mensuel net (CFA)", 150_000, 5_000_000, 500_000, 50_000)
    charges      = st.number_input("Charges fixes mensuelles (CFA)", 0, 3_000_000, 100_000, 10_000,
                                    help="Loyer, autres credits en cours, etc.")
with col2:
    anciennete   = st.slider("Anciennete client (mois)", 1, 120, 24)
    duree        = st.selectbox("Duree du credit (mois)", [6, 12, 18, 24, 36, 48, 60])
    nb_incidents = st.selectbox("Incidents de paiement passes", [0, 1, 2, 3, 4, 5],
                                 help="Retards ou defauts sur des credits precedents")

# Calcul
prob, dti, cap_nette = predict_proba(montant, revenu, anciennete, duree, charges, nb_incidents)

# Capacite nette negative = remboursement impossible, on force le rejet
if cap_nette < 0:
    prob = 1.0

solvable = prob < 0.20

st.divider()

# Metriques + Verdict
m1, m2, m3 = st.columns(3)
m1.metric("Probabilite de defaut", f"{prob:.1%}")
m2.metric("Taux d'endettement (DTI)", f"{dti:.1%}", help="(Mensualite + charges) / revenu. Critique > 40%")
m3.metric("Capacite nette mensuelle", f"{cap_nette:,.0f} CFA")

if cap_nette < 0:
    st.error("Dossier REJETE — La mensualite depasse le revenu disponible apres charges.")
elif solvable:
    st.success(f"Dossier ELIGIBLE — Risque de defaut faible ({prob:.1%})")
elif prob < 0.40:
    st.warning(f"Dossier a surveiller — Risque modere ({prob:.1%}), analyse complementaire recommandee.")
else:
    st.error(f"Dossier REJETE — Risque de defaut eleve ({prob:.1%})")

st.divider()

# Graphique
revenus = np.linspace(150_000, 3_000_000, 300)
risques = []
for r in revenus:
    p, _, cn = predict_proba(montant, r, anciennete, duree, charges, nb_incidents)
    risques.append(1.0 if cn < 0 else p)

fig = go.Figure()

fig.add_hrect(y0=0,    y1=0.20, fillcolor="rgba(63,185,80,0.07)",  line_width=0)
fig.add_hrect(y0=0.20, y1=0.40, fillcolor="rgba(210,153,34,0.07)", line_width=0)
fig.add_hrect(y0=0.40, y1=1.0,  fillcolor="rgba(248,81,73,0.07)",  line_width=0)

fig.add_trace(go.Scatter(
    x=revenus, y=risques,
    mode="lines", line=dict(color="#58a6ff", width=2.5),
    fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    name="P(Defaut)"
))

fig.add_hline(y=0.20, line_dash="dot", line_color="#3fb950",
              annotation_text="Seuil eligible (20%)", annotation_font_color="#3fb950",
              annotation_position="top right")
fig.add_hline(y=0.40, line_dash="dot", line_color="#d29922",
              annotation_text="Seuil rejet (40%)", annotation_font_color="#d29922",
              annotation_position="top right")

point_color = "#3fb950" if solvable else ("#d29922" if prob < 0.40 else "#f85149")
fig.add_trace(go.Scatter(
    x=[revenu], y=[prob],
    mode="markers",
    marker=dict(color=point_color, size=14, symbol="circle",
                line=dict(color="white", width=2)),
    name="Client actuel"
))

fig.update_layout(
    title="Probabilite de defaut selon le revenu mensuel",
    xaxis_title="Revenu mensuel (CFA)",
    yaxis=dict(title="P(Defaut)", tickformat=".0%", range=[0, 1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=60, b=40), height=380,
    annotations=[
        dict(x=250_000, y=0.10, text="Eligible", showarrow=False, font=dict(color="#3fb950", size=11)),
        dict(x=250_000, y=0.30, text="Modere",   showarrow=False, font=dict(color="#d29922", size=11)),
        dict(x=250_000, y=0.65, text="Rejete",   showarrow=False, font=dict(color="#f85149", size=11)),
    ]
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Modele : Regression Logistique (sklearn) · Dataset : 5 000 dossiers synthetiques · "
    "Variables : DTI, anciennete, incidents de paiement, capacite nette, ratio montant/revenu"
)
