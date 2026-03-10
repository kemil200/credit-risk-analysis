import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="ScoreCredit", layout="centered")
st.title("ScoreCredit")
st.caption("Analyse de risque crédit")
st.divider()

# ── Modèle entraîné ───────────────────────────────────────────────────────────
# Features : taux_endettement | log_anciennete | nb_incidents |
#            log_capacite_nette | ratio_montant_revenu | taux_annuel
COEF      = np.array([1.4291, -0.4675, 0.8437, -0.4201, 0.0776, 0.2854])
INTERCEPT = -3.7735
MEAN      = np.array([0.614103, 1.644405, 0.4122, 1.251553, 0.45271, 0.149104])
SCALE     = np.array([0.557874, 0.603517, 0.63142, 0.724844, 0.476419, 0.058073])

def mensualite_reelle(montant, duree, taux_annuel):
    """Formule d'amortissement standard (annuité constante)."""
    r = taux_annuel / 12
    if r < 1e-9:
        return montant / duree
    return montant * (r * (1 + r) ** duree) / ((1 + r) ** duree - 1)

def predict(montant, revenu, anciennete, duree, charges, nb_incidents, taux_annuel):
    mens                 = mensualite_reelle(montant, duree, taux_annuel)
    taux_endettement     = (mens + charges) / revenu
    log_anciennete       = np.log1p(anciennete / 12)
    capacite_nette       = revenu - charges - mens
    log_capacite_nette   = np.log1p(max(capacite_nette, 0) / 100_000)
    ratio_montant_revenu = montant / (revenu * 12)

    x        = np.array([taux_endettement, log_anciennete, nb_incidents,
                          log_capacite_nette, ratio_montant_revenu, taux_annuel])
    x_scaled = (x - MEAN) / SCALE
    logit    = INTERCEPT + np.dot(COEF, x_scaled)
    prob     = 1 / (1 + np.exp(-logit))
    return float(prob), float(mens), float(taux_endettement), float(capacite_nette)

# ── Saisie ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    montant      = st.number_input("Montant demandé (CFA)", 200_000, 10_000_000, 2_000_000, 100_000)
    revenu       = st.number_input("Revenu mensuel net (CFA)", 150_000, 5_000_000, 500_000, 50_000)
    charges      = st.number_input("Charges fixes mensuelles (CFA)", 0, 3_000_000, 100_000, 10_000,
                                    help="Loyer, autres crédits en cours, etc.")
with col2:
    taux_pct     = st.slider("Taux d'intérêt annuel (%)", 1.0, 30.0, 12.0, 0.5,
                              help="Taux annuel fixe — la mensualité est calculée par amortissement")
    anciennete   = st.slider("Ancienneté client (mois)", 1, 120, 24)
    duree        = st.selectbox("Durée du crédit (mois)", [6, 12, 18, 24, 36, 48, 60])
    nb_incidents = st.selectbox("Incidents de paiement passés", [0, 1, 2, 3, 4, 5],
                                 help="Retards ou défauts sur des crédits précédents")

taux_annuel = taux_pct / 100

# ── Calcul ────────────────────────────────────────────────────────────────────
prob, mens, dti, cap_nette = predict(montant, revenu, anciennete, duree,
                                      charges, nb_incidents, taux_annuel)
solvable = prob < 0.20

st.divider()

# ── Métriques ─────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Probabilité de défaut",    f"{prob:.1%}")
m2.metric("Mensualité réelle",        f"{mens:,.0f} CFA",
          help="Calculée par amortissement avec le taux d'intérêt")
m3.metric("Taux d'endettement (DTI)", f"{dti:.1%}",
          help="(Mensualité + charges) / revenu. Critique > 40%")
m4.metric("Capacité nette / mois",    f"{cap_nette:,.0f} CFA")

# ── Verdict ───────────────────────────────────────────────────────────────────
if cap_nette < 0:
    st.error("⛔ **Dossier REJETÉ** — La mensualité dépasse le revenu disponible après charges.")
elif solvable:
    st.success(f"✅ **Dossier ÉLIGIBLE** — Risque de défaut faible ({prob:.1%})")
elif prob < 0.40:
    st.warning(f"⚠️ **Dossier à surveiller** — Risque modéré ({prob:.1%}), analyse complémentaire recommandée.")
else:
    st.error(f"⛔ **Dossier REJETÉ** — Risque de défaut élevé ({prob:.1%})")

st.divider()

# ── Graphique : P(défaut) vs Revenu ──────────────────────────────────────────
revenus = np.linspace(150_000, 3_000_000, 300)
risques = [predict(montant, r, anciennete, duree, charges, nb_incidents, taux_annuel)[0]
           for r in revenus]

fig = go.Figure()

fig.add_hrect(y0=0,    y1=0.20, fillcolor="rgba(63,185,80,0.07)",  line_width=0)
fig.add_hrect(y0=0.20, y1=0.40, fillcolor="rgba(210,153,34,0.07)", line_width=0)
fig.add_hrect(y0=0.40, y1=1.0,  fillcolor="rgba(248,81,73,0.07)",  line_width=0)

fig.add_trace(go.Scatter(
    x=revenus, y=risques,
    mode="lines", line=dict(color="#58a6ff", width=2.5),
    fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    name="P(Défaut)"
))

fig.add_hline(y=0.20, line_dash="dot", line_color="#3fb950",
              annotation_text="Seuil éligible (20%)", annotation_font_color="#3fb950",
              annotation_position="top right")
fig.add_hline(y=0.40, line_dash="dot", line_color="#d29922",
              annotation_text="Seuil rejet (40%)", annotation_font_color="#d29922",
              annotation_position="top right")

fig.add_trace(go.Scatter(
    x=[revenu], y=[prob],
    mode="markers",
    marker=dict(
        color="#3fb950" if solvable else ("#d29922" if prob < 0.40 else "#f85149"),
        size=14, symbol="circle", line=dict(color="white", width=2)
    ),
    name="Client actuel"
))

fig.update_layout(
    title=f"Probabilité de défaut selon le revenu  (taux {taux_pct:.1f}% · durée {duree} mois)",
    xaxis_title="Revenu mensuel (CFA)",
    yaxis=dict(title="P(Défaut)", tickformat=".0%", range=[0, 1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=60, b=40), height=380,
    annotations=[
        dict(x=250_000, y=0.10, text="✅ Éligible", showarrow=False,
             font=dict(color="#3fb950", size=11)),
        dict(x=250_000, y=0.30, text="⚠️ Modéré",  showarrow=False,
             font=dict(color="#d29922", size=11)),
        dict(x=250_000, y=0.65, text="⛔ Rejeté",  showarrow=False,
             font=dict(color="#f85149", size=11)),
    ]
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Modèle : Régression Logistique (sklearn) · Dataset : 5 000 dossiers synthétiques · "
    "Variables : DTI, ancienneté, incidents, capacité nette, ratio montant/revenu, taux d'intérêt · "
    "Mensualité calculée par amortissement (annuité constante)"
)
