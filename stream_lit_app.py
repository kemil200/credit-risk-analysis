import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import softmax

st.set_page_config(page_title="ScoreCredit", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
.block-container { padding: 2rem 3rem; max-width: 1100px; margin: auto; }

h1 { font-size: 2.2rem !important; font-weight: 800 !important;
     background: linear-gradient(135deg, #58a6ff, #3fb950);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

h3 { color: #e6edf3 !important; }

.section-label {
    font-size: 0.72rem; font-weight: 700; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 0.8rem;
}

.card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
}
.card-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase;
              letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.card-value { font-family: 'DM Mono', monospace; font-size: 1.8rem; font-weight: 700; }
.card-value.green  { color: #3fb950; }
.card-value.red    { color: #f85149; }
.card-value.yellow { color: #d29922; }
.card-value.blue   { color: #58a6ff; }

.badge { display: inline-block; padding: 0.35rem 1rem; border-radius: 999px;
         font-size: 0.8rem; font-weight: 700; margin-top: 0.5rem; }
.badge-green  { background: #0d2c17; color: #3fb950; border: 1px solid #3fb950; }
.badge-red    { background: #2d1215; color: #f85149; border: 1px solid #f85149; }
.badge-yellow { background: #2c1f0a; color: #d29922; border: 1px solid #d29922; }

.stButton > button {
    background: linear-gradient(135deg, #58a6ff, #3fb950);
    color: #0d1117; font-weight: 800; border: none;
    border-radius: 8px; padding: 0.55rem 1.8rem; width: 100%;
}
div[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace !important; }
.stSelectbox label, .stSlider label, .stNumberInput label { color: #8b949e !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("ScoreCredit")
st.markdown("<p style='color:#8b949e; margin-top:-1rem; font-size:1rem;'>Analyse de risque crédit · Régression logistique multinomiale · 500 dossiers réels</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#21262d; margin: 1rem 0 1.5rem;'>", unsafe_allow_html=True)

# ── Modèle multinomial entraîné sur Jeux_donnees.csv ─────────────────────────
# Classes : 0 = EN COURS | 1 = EN RETARD | 2 = REMBOURSE
# Features : log_montant | Duree | taux | Age | sexe_bin | act_Artisanat | act_Commerce
CLASSES   = ["EN COURS", "EN RETARD", "REMBOURSE"]
COEFS     = np.array([
    [ 0.0003,  0.0278, -0.0944, -0.0446, -0.0079, -0.0426,  0.0532],
    [ 0.1210, -0.0300,  0.0825,  0.1359,  0.0049,  0.1671, -0.0038],
    [-0.1213,  0.0022,  0.0119, -0.0913,  0.0030, -0.1245, -0.0494],
])
INTERCEPT = np.array([0.0106, -0.0125, 0.0019])
MEAN      = np.array([12.289189, 19.608, 0.112, 44.198, 0.532, 0.35, 0.346])
SCALE     = np.array([1.012749, 10.39261, 0.025706, 14.442673, 0.498975, 0.47697, 0.475693])

def mensualite_reelle(montant, duree, taux_annuel):
    r = taux_annuel / 12
    if r < 1e-9:
        return montant / duree
    return montant * (r * (1 + r) ** duree) / ((1 + r) ** duree - 1)

def predict(montant, duree, taux_annuel, age, sexe, activite):
    x = np.array([
        np.log(montant), duree, taux_annuel, age,
        1 if sexe == "Homme" else 0,
        1 if activite == "Artisanat" else 0,
        1 if activite == "Commerce" else 0,
    ])
    x_scaled = (x - MEAN) / SCALE
    logits   = INTERCEPT + COEFS @ x_scaled
    return softmax(logits)  # [P(EN COURS), P(EN RETARD), P(REMBOURSE)]

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.8], gap="large")

with left:
    st.markdown('<div class="section-label">Profil du dossier</div>', unsafe_allow_html=True)

    montant  = st.number_input("Montant demandé (CFA)", 50_000, 5_000_000, 300_000, 25_000)
    taux_pct = st.slider("Taux d'intérêt annuel (%)", 5.0, 20.0, 12.0, 0.5)
    duree    = st.selectbox("Durée du crédit (mois)", [6, 12, 18, 24, 36])
    age      = st.slider("Âge du client", 18, 80, 40)
    activite = st.selectbox("Activité", ["Commerce", "Artisanat", "Agriculture"])
    sexe     = st.selectbox("Sexe", ["Homme", "Femme"])

    st.markdown("<br>", unsafe_allow_html=True)
    calculer = st.button("▶ Analyser le dossier")

with right:
    taux_annuel = taux_pct / 100
    mens        = mensualite_reelle(montant, duree, taux_annuel)
    probs       = predict(montant, duree, taux_annuel, age, sexe, activite)
    p_cours, p_retard, p_rembourse = probs
    classe      = CLASSES[np.argmax(probs)]

    # Couleur et badge selon la classe prédite
    if classe == "REMBOURSE":
        couleur, badge_cls, verdict = "green",  "badge-green",  "✅ Dossier Éligible"
    elif classe == "EN COURS":
        couleur, badge_cls, verdict = "blue",   "badge-yellow", "🕐 En cours — surveiller"
    else:
        couleur, badge_cls, verdict = "red",    "badge-red",    "⛔ Risque de retard élevé"

    # Métriques principales
    st.markdown('<div class="section-label">Résultats</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Statut prédit</div>
            <div class="card-value {couleur}">{classe}</div>
            <span class="badge {badge_cls}">{verdict}</span>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Mensualité réelle</div>
            <div class="card-value blue">{mens:,.0f} <span style="font-size:1rem">CFA</span></div>
            <div style="color:#8b949e; font-size:0.8rem; margin-top:0.4rem">Coût total : {mens*duree:,.0f} CFA</div>
        </div>""", unsafe_allow_html=True)

    # Probabilités par classe — barres horizontales
    st.markdown('<div class="section-label" style="margin-top:0.5rem">Probabilités par statut</div>', unsafe_allow_html=True)

    for label, prob, color in [
        ("EN RETARD",  p_retard,   "#f85149"),
        ("EN COURS",   p_cours,    "#58a6ff"),
        ("REMBOURSE",  p_rembourse,"#3fb950"),
    ]:
        bar_pct = int(prob * 100)
        st.markdown(f"""
        <div style="margin-bottom:0.7rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#8b949e; margin-bottom:0.3rem;">
                <span>{label}</span><span style="font-family:'DM Mono',monospace; color:{color}; font-weight:700;">{prob:.1%}</span>
            </div>
            <div style="background:#21262d; border-radius:999px; height:8px;">
                <div style="background:{color}; width:{bar_pct}%; height:8px; border-radius:999px; transition:width 0.4s;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#21262d; margin: 1rem 0;'>", unsafe_allow_html=True)

    # Graphique : évolution des 3 probabilités selon le montant
    st.markdown('<div class="section-label">Sensibilité au montant</div>', unsafe_allow_html=True)

    montants   = np.linspace(50_000, 2_000_000, 250)
    p_r_curve  = [predict(m, duree, taux_annuel, age, sexe, activite)[1] for m in montants]
    p_rb_curve = [predict(m, duree, taux_annuel, age, sexe, activite)[2] for m in montants]
    p_c_curve  = [predict(m, duree, taux_annuel, age, sexe, activite)[0] for m in montants]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=montants, y=p_r_curve,  mode="lines",
                             line=dict(color="#f85149", width=2), name="EN RETARD"))
    fig.add_trace(go.Scatter(x=montants, y=p_rb_curve, mode="lines",
                             line=dict(color="#3fb950", width=2), name="REMBOURSE"))
    fig.add_trace(go.Scatter(x=montants, y=p_c_curve,  mode="lines",
                             line=dict(color="#58a6ff", width=1.5, dash="dot"), name="EN COURS"))
    fig.add_trace(go.Scatter(
        x=[montant], y=[p_retard], mode="markers",
        marker=dict(color="#f85149" if classe=="EN RETARD" else "#3fb950",
                    size=12, symbol="circle", line=dict(color="white", width=2)),
        name="Dossier actuel", showlegend=True
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", family="Syne"),
        xaxis=dict(title="Montant (CFA)", gridcolor="#21262d", tickformat=",.0f"),
        yaxis=dict(title="Probabilité", tickformat=".0%", range=[0, 1], gridcolor="#21262d"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=40, l=10, r=10), height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#21262d; margin: 1.5rem 0 0.5rem;'>", unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
f1.caption("**Modèle** : Régression Logistique Multinomiale (softmax)")
f2.caption("**Données** : 500 dossiers réels · Taux de défaut observé : 33.2%")
f3.caption("**Variables** : montant, durée, taux, âge, sexe, activité")
