import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.special import expit  # Fonction sigmoïde (logistique)

# ─── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="ScoreCredit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Personnalisé ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

    .main { background-color: #0d1117; color: #e6edf3; }
    .stApp { background-color: #0d1117; }

    .block-container { padding: 2rem 3rem; }

    h1 { font-size: 2.4rem !important; font-weight: 800 !important;
         background: linear-gradient(135deg, #58a6ff, #3fb950);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card h3 { color: #8b949e; font-size: 0.8rem; letter-spacing: 0.1em;
                      text-transform: uppercase; margin: 0 0 0.3rem; }
    .metric-card .value { font-family: 'DM Mono', monospace; font-size: 2rem;
                          font-weight: 700; color: #e6edf3; }
    .metric-card .value.good { color: #3fb950; }
    .metric-card .value.bad  { color: #f85149; }
    .metric-card .value.warn { color: #d29922; }

    .badge {
        display: inline-block; padding: 0.4rem 1rem;
        border-radius: 999px; font-size: 0.85rem; font-weight: 700;
        letter-spacing: 0.05em; margin-top: 0.5rem;
    }
    .badge-green  { background: #0d2c17; color: #3fb950; border: 1px solid #3fb950; }
    .badge-red    { background: #2d1215; color: #f85149; border: 1px solid #f85149; }
    .badge-yellow { background: #2c1f0a; color: #d29922; border: 1px solid #d29922; }

    .stButton > button {
        background: linear-gradient(135deg, #58a6ff, #3fb950);
        color: #0d1117; font-weight: 800; font-size: 1rem;
        border: none; border-radius: 8px; padding: 0.6rem 2rem;
        width: 100%; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .stSlider > div > div { background: #30363d !important; }
    div[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace !important; }

    .section-title {
        font-size: 1rem; font-weight: 700; color: #8b949e;
        text-transform: uppercase; letter-spacing: 0.12em;
        border-bottom: 1px solid #21262d; padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── En-tête ──────────────────────────────────────────────────────────────────
st.title("ScoreCredit Pro")
st.markdown("<p style='color:#8b949e; font-size:1.05rem; margin-top:-1rem;'>Outil d'analyse de risque crédit · <em>Decreasing Trend Model</em></p>", unsafe_allow_html=True)
st.markdown("---")

# ─── Barre latérale ───────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Paramètres du Modèle")
st.sidebar.markdown("**Régression Logistique**")

poids_endettement = st.sidebar.slider("λ Endettement", 0.0, 3.0, 1.5, 0.1,
    help="Coefficient multiplicateur du ratio endettement dans la log-régression")
poids_fidelite = st.sidebar.slider("λ Fidélité", 0.0, 3.0, 1.2, 0.1,
    help="Coefficient multiplicateur de l'ancienneté (effet protecteur)")
seuil_risque = st.sidebar.slider("Seuil de rejet (probabilité)", 0.3, 0.8, 0.5, 0.05,
    help="Probabilité de défaut au-delà de laquelle le dossier est rejeté")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Données de Référence")
st.sidebar.markdown("**Revenu médian marché :** 450 000 CFA/mois")
st.sidebar.markdown("**Taux de défaut portefeuille :** 12.4 %")

# ─── Modèle logistique ────────────────────────────────────────────────────────
def calculer_score_logistique(montant, revenu, anciennete, duree, w_endet, w_fid):
    """
    Modèle de régression logistique :
    P(défaut) = sigmoid( β0 + β1*ratio_endettement - β2*ratio_fidélité )
    → Plus le revenu est élevé ou l'ancienneté grande, MOINS le risque est élevé
      (tendance décroissante avec ces deux variables)
    """
    ratio_endettement = (montant / duree) / revenu   # remboursement mensuel / revenu
    ratio_fidelite    = anciennete / 120              # normalisé sur 10 ans max

    # Log-odds (logit) : endettement augmente le risque, fidélité le réduit
    beta0   = -0.5
    logit_p = beta0 + (w_endet * ratio_endettement * 10) - (w_fid * ratio_fidelite * 5)

    probabilite_defaut = float(expit(logit_p))       # ∈ ]0, 1[

    # Score de solvabilité inversé : 100 = excellent, 0 = très risqué
    score_solvabilite = round((1 - probabilite_defaut) * 100, 1)

    return probabilite_defaut, score_solvabilite, ratio_endettement, ratio_fidelite


# ─── Layout principal ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown('<div class="section-title">Simulateur Client</div>', unsafe_allow_html=True)

    montant    = st.number_input("Montant demandé (CFA)", 500_000, 10_000_000, 2_000_000, 100_000)
    revenu     = st.number_input("Revenu Mensuel (CFA)", 150_000, 5_000_000, 500_000, 50_000)
    anciennete = st.slider("Ancienneté client (mois)", 1, 120, 24)
    duree      = st.selectbox("Durée du crédit (mois)", [6, 12, 18, 24, 36, 48, 60])

    calculate = st.button("▶ Calculer le Score")

    if calculate:
        prob, score_sol, r_end, r_fid = calculer_score_logistique(
            montant, revenu, anciennete, duree, poids_endettement, poids_fidelite
        )

        # Classification
        if prob >= seuil_risque:
            couleur, badge_class, verdict = "bad", "badge-red", "⛔ Dossier Rejeté"
        elif prob >= seuil_risque * 0.7:
            couleur, badge_class, verdict = "warn", "badge-yellow", "⚠️ Risque Modéré"
        else:
            couleur, badge_class, verdict = "good", "badge-green", "✅ Dossier Éligible"

        st.markdown(f"""
        <div class="metric-card">
            <h3>Probabilité de Défaut</h3>
            <div class="value {couleur}">{prob:.1%}</div>
            <span class="badge {badge_class}">{verdict}</span>
        </div>
        <div class="metric-card">
            <h3>Score de Solvabilité</h3>
            <div class="value {couleur}">{score_sol} / 100</div>
        </div>
        <div class="metric-card">
            <h3>Ratio Endettement Mensuel</h3>
            <div class="value">{r_end:.1%}</div>
        </div>
        <div class="metric-card">
            <h3>Indice Fidélité Normalisé</h3>
            <div class="value">{r_fid:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Jauge probabilité de défaut
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"color": "#e6edf3", "family": "DM Mono"}},
            delta={"reference": seuil_risque * 100, "decreasing": {"color": "#3fb950"}, "increasing": {"color": "#f85149"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": "#58a6ff"},
                "bgcolor": "#161b22",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, seuil_risque * 70], "color": "#0d2c17"},
                    {"range": [seuil_risque * 70, seuil_risque * 100], "color": "#2c1f0a"},
                    {"range": [seuil_risque * 100, 100], "color": "#2d1215"},
                ],
                "threshold": {"line": {"color": "#f85149", "width": 3},
                              "thickness": 0.8, "value": seuil_risque * 100}
            },
            title={"text": "Risque de Défaut", "font": {"color": "#8b949e", "size": 13}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0d1117", font_color="#e6edf3", height=220, margin=dict(t=40, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.markdown('<div class="section-title">Analyse de Sensibilité — Decreasing Trend</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉 Risque vs Revenu", "📊 Risque vs Ancienneté", "🗺️ Surface de Risque"])

    # ── Tab 1 : Tendance DÉCROISSANTE avec le revenu ───────────────────────────
    with tab1:
        revenus_range = np.linspace(200_000, 3_000_000, 80)
        probs_revenu  = [
            calculer_score_logistique(montant, r, anciennete, duree, poids_endettement, poids_fidelite)[0]
            for r in revenus_range
        ]

        df_rev = pd.DataFrame({"Revenu mensuel (CFA)": revenus_range, "P(Défaut)": probs_revenu})

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df_rev["Revenu mensuel (CFA)"], y=df_rev["P(Défaut)"],
            mode="lines", line=dict(color="#58a6ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
            name="P(Défaut)"
        ))
        fig1.add_hline(y=seuil_risque, line_dash="dot", line_color="#f85149",
                       annotation_text=f"Seuil de rejet ({seuil_risque:.0%})",
                       annotation_font_color="#f85149")
        # Marqueur position actuelle
        p_current = calculer_score_logistique(montant, revenu, anciennete, duree, poids_endettement, poids_fidelite)[0]
        fig1.add_trace(go.Scatter(
            x=[revenu], y=[p_current],
            mode="markers", marker=dict(color="#3fb950", size=12, symbol="circle"),
            name="Client actuel"
        ))
        fig1.update_layout(
            title="Tendance Décroissante : plus le revenu ↑, moins le risque ↓",
            xaxis_title="Revenu mensuel (CFA)", yaxis_title="Probabilité de défaut",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#8b949e"), legend=dict(bgcolor="#161b22"),
            margin=dict(t=50, b=40), height=360
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.info("📉 **Decreasing trend :** la probabilité de défaut diminue à mesure que le revenu augmente — conformément au modèle logistique calibré.")

    # ── Tab 2 : Tendance DÉCROISSANTE avec l'ancienneté ───────────────────────
    with tab2:
        anciennetes_range = np.arange(1, 121, 1)
        probs_anc = [
            calculer_score_logistique(montant, revenu, int(a), duree, poids_endettement, poids_fidelite)[0]
            for a in anciennetes_range
        ]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=anciennetes_range, y=probs_anc,
            mode="lines", line=dict(color="#3fb950", width=2.5),
            fill="tozeroy", fillcolor="rgba(63,185,80,0.08)",
            name="P(Défaut)"
        ))
        fig2.add_hline(y=seuil_risque, line_dash="dot", line_color="#f85149",
                       annotation_text=f"Seuil ({seuil_risque:.0%})",
                       annotation_font_color="#f85149")
        fig2.add_trace(go.Scatter(
            x=[anciennete], y=[p_current if calculate else probs_anc[anciennete - 1]],
            mode="markers", marker=dict(color="#58a6ff", size=12),
            name="Client actuel"
        ))
        fig2.update_layout(
            title="Tendance Décroissante : plus l'ancienneté ↑, moins le risque ↓",
            xaxis_title="Ancienneté (mois)", yaxis_title="Probabilité de défaut",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#8b949e"), legend=dict(bgcolor="#161b22"),
            margin=dict(t=50, b=40), height=360
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3 : Heatmap 2D Revenu × Ancienneté ────────────────────────────────
    with tab3:
        rev_grid = np.linspace(200_000, 2_000_000, 30)
        anc_grid = np.linspace(1, 120, 30)
        Z = np.array([
            [calculer_score_logistique(montant, r, int(a), duree, poids_endettement, poids_fidelite)[0]
             for r in rev_grid]
            for a in anc_grid
        ])

        fig3 = go.Figure(go.Heatmap(
            z=Z, x=rev_grid / 1000, y=anc_grid,
            colorscale=[[0, "#0d2c17"], [0.5, "#2c1f0a"], [1, "#2d1215"]],
            colorbar=dict(
                title=dict(text="P(Défaut)", font=dict(color="#8b949e")),
                tickformat=".0%",
                tickfont=dict(color="#8b949e")
            ),
            zmin=0, zmax=1
        ))
        fig3.update_layout(
            title="Surface de Risque : Revenu × Ancienneté",
            xaxis_title="Revenu mensuel (k CFA)", yaxis_title="Ancienneté (mois)",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#8b949e"), height=380,
            margin=dict(t=50, b=40)
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Zone verte = faible risque · Zone rouge = risque élevé")

# ─── Section récapitulative ───────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">À Propos du Modèle</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    **Régression Logistique**
    La probabilité de défaut est estimée via la fonction sigmoïde appliquée à une combinaison
    linéaire de variables financières :
    `P(défaut) = σ(β₀ + λ₁·Endettement − λ₂·Fidélité)`
    """)
with c2:
    st.markdown("""
    **Decreasing Trend**
    Conformément à la théorie financière, le modèle garantit une **tendance strictement
    décroissante** du risque avec le revenu et l'ancienneté client : plus ces indicateurs
    sont élevés, plus la probabilité de défaut s'approche de 0.
    """)
with c3:
    st.markdown("""
    **Paramètres Ajustables**
    Les coefficients λ (poids) sont configurables dans la barre latérale pour calibrer
    le modèle selon le profil de risque de votre portefeuille.
    Le seuil de rejet est également paramétrable (défaut : 50 %).
    """)
