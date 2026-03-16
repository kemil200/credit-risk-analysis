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
.section-label { font-size: 0.72rem; font-weight: 700; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.8rem; }
.card { background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem; }
.card-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.card-value { font-family: 'DM Mono', monospace; font-size: 1.8rem; font-weight: 700; }
.card-value.green  { color: #3fb950; }
.card-value.red    { color: #f85149; }
.card-value.blue   { color: #58a6ff; }
.card-value.yellow { color: #d29922; }
.badge { display: inline-block; padding: 0.35rem 1rem; border-radius: 999px;
    font-size: 0.8rem; font-weight: 700; margin-top: 0.5rem; }
.badge-green  { background: #0d2c17; color: #3fb950; border: 1px solid #3fb950; }
.badge-red    { background: #2d1215; color: #f85149; border: 1px solid #f85149; }
.badge-yellow { background: #2c1f0a; color: #d29922; border: 1px solid #d29922; }
.fair-tag { display:inline-block; background:#0d1f3c; color:#58a6ff;
    border:1px solid #1f4080; border-radius:6px; padding:0.2rem 0.7rem;
    font-size:0.72rem; font-weight:700; letter-spacing:0.08em; margin-left:0.5rem; }
.stButton > button { background: linear-gradient(135deg, #58a6ff, #3fb950);
    color: #0d1117; font-weight: 800; border: none;
    border-radius: 8px; padding: 0.55rem 1.8rem; width: 100%; }
.stSelectbox label, .stSlider label, .stNumberInput label
    { color: #8b949e !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("ScoreCredit")
st.markdown(
    "<p style='color:#8b949e; margin-top:-1rem; font-size:1rem;'>"
    "Analyse de risque crédit · Régression logistique multinomiale · "
    "<span class='fair-tag'>⚖️ FAIRNESS-AWARE</span></p>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border-color:#21262d; margin:1rem 0 1.5rem;'>", unsafe_allow_html=True)

# ── Modèle fairness-aware ─────────────────────────────────────────────────────
# Sans âge ni sexe — variables économiques uniquement
# Classes     : 0=EN COURS | 1=EN RETARD | 2=REMBOURSE
# Features    : log_montant | taux | duree | charge_interet | cout_total_ratio
#               act_Artisanat | act_Commerce
#               reg_Kara | reg_Maritime | reg_Plateau | reg_Savane
#               cat_Moyen | cat_Petit
CLASSES   = ["EN COURS", "EN RETARD", "REMBOURSE"]
COEFS     = np.array([
    [-0.1279, -0.0875,  0.0845, -0.0335, -0.0335, -0.0391,  0.0647,  0.0115, -0.0813,  0.0322, -0.1659, -0.0124, -0.1817],
    [ 0.2095,  0.1478,  0.1292, -0.0952, -0.0952,  0.1505, -0.0210,  0.0328, -0.0887, -0.0251, -0.0271,  0.0714,  0.1386],
    [-0.0816, -0.0603, -0.2137,  0.1288,  0.1288, -0.1114, -0.0437, -0.0444,  0.1700, -0.0071,  0.1930, -0.0591,  0.0431],
])
INTERCEPT = np.array([0.0103, 0.0071, -0.0174])
MEAN      = np.array([12.289189, 0.112, 19.608, 0.18439, 1.18439,
                       0.35, 0.346, 0.112, 0.234, 0.238, 0.252, 0.51, 0.264])
SCALE     = np.array([1.012749, 0.025706, 10.39261, 0.111429, 0.111429,
                       0.47697, 0.475693, 0.315366, 0.423372, 0.425859, 0.434161, 0.4999, 0.440799])

def mensualite_reelle(montant, duree, taux_annuel):
    r = taux_annuel / 12
    if r < 1e-9:
        return montant / duree
    return montant * (r * (1 + r) ** duree) / ((1 + r) ** duree - 1)

def predict(montant, duree, taux_annuel, activite, region, categorie):
    interets        = montant * taux_annuel
    charge_interet  = interets / montant           # = taux_annuel
    total_du        = montant + interets
    cout_total_ratio= total_du / montant

    x = np.array([
        np.log(montant),
        taux_annuel,
        duree,
        charge_interet,
        cout_total_ratio,
        1 if activite == "Artisanat"  else 0,
        1 if activite == "Commerce"   else 0,
        1 if region   == "Kara"       else 0,
        1 if region   == "Maritime"   else 0,
        1 if region   == "Plateau"    else 0,
        1 if region   == "Savane"     else 0,
        1 if categorie == "Moyen"     else 0,
        1 if categorie == "Petit"     else 0,
    ])
    x_scaled = (x - MEAN) / SCALE
    logits   = INTERCEPT + COEFS @ x_scaled
    return softmax(logits)

def get_categorie(montant):
    if montant <= 100_000:   return "Petit"
    elif montant <= 500_000: return "Moyen"
    else:                    return "Grand"

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.8], gap="large")

with left:
    st.markdown('<div class="section-label">Profil du dossier</div>', unsafe_allow_html=True)

    montant  = st.number_input("Montant demandé (CFA)", 50_000, 1_000_000, 300_000, 25_000)
    taux_pct = st.slider("Taux d'intérêt annuel (%)", 8.0, 15.0, 12.0, 0.5)
    duree    = st.selectbox("Durée du crédit (mois)", [6, 12, 18, 24, 36])
    activite = st.selectbox("Activité", ["Commerce", "Artisanat", "Agriculture"])
    region   = st.selectbox("Région", ["Maritime", "Savane", "Plateau", "Kara", "Centrale"])

    taux_annuel = taux_pct / 100
    categorie   = get_categorie(montant)
    mens        = mensualite_reelle(montant, duree, taux_annuel)

    st.markdown(f"""
    <div style='background:#161b22; border:1px solid #21262d; border-radius:8px;
                padding:0.8rem 1rem; margin-top:0.5rem; font-size:0.82rem; color:#8b949e;'>
        Catégorie détectée : <strong style='color:#58a6ff'>{categorie}</strong><br>
        Mensualité estimée : <strong style='color:#3fb950; font-family:DM Mono,monospace'>{mens:,.0f} CFA</strong>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d1f3c; border:1px solid #1f4080; border-radius:8px;
                padding:0.8rem 1rem; font-size:0.78rem; color:#8b949e; line-height:1.6;'>
        <strong style='color:#58a6ff'>⚖️ Modèle Fairness-Aware</strong><br>
        Âge et sexe exclus du modèle.<br>
        Seules les variables économiques sont utilisées :
        montant, taux, durée, activité, région.
    </div>""", unsafe_allow_html=True)

with right:
    probs             = predict(montant, duree, taux_annuel, activite, region, categorie)
    p_cours, p_retard, p_rembourse = probs
    classe            = CLASSES[np.argmax(probs)]

    if classe == "REMBOURSE":
        couleur, badge_cls, verdict = "green",  "badge-green",  "✅ Dossier Éligible"
    elif classe == "EN COURS":
        couleur, badge_cls, verdict = "blue",   "badge-yellow", "🕐 En cours — à surveiller"
    else:
        couleur, badge_cls, verdict = "red",    "badge-red",    "⛔ Risque de retard élevé"

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
        cout_total = mens * duree
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Coût total du crédit</div>
            <div class="card-value blue">{cout_total:,.0f} <span style='font-size:1rem'>CFA</span></div>
            <div style='color:#8b949e; font-size:0.8rem; margin-top:0.4rem'>
                Mensualité : {mens:,.0f} CFA / mois
            </div>
        </div>""", unsafe_allow_html=True)

    # Barres de probabilité
    st.markdown('<div class="section-label" style="margin-top:0.2rem">Probabilités par statut</div>', unsafe_allow_html=True)
    for label, prob, color in [
        ("EN RETARD",  p_retard,    "#f85149"),
        ("EN COURS",   p_cours,     "#58a6ff"),
        ("REMBOURSE",  p_rembourse, "#3fb950"),
    ]:
        st.markdown(f"""
        <div style='margin-bottom:0.7rem;'>
            <div style='display:flex; justify-content:space-between;
                        font-size:0.8rem; color:#8b949e; margin-bottom:0.3rem;'>
                <span>{label}</span>
                <span style='font-family:DM Mono,monospace; color:{color}; font-weight:700;'>{prob:.1%}</span>
            </div>
            <div style='background:#21262d; border-radius:999px; height:8px;'>
                <div style='background:{color}; width:{int(prob*100)}%; height:8px;
                            border-radius:999px;'></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#21262d; margin:1rem 0;'>", unsafe_allow_html=True)

    # Graphique 3 courbes vs montant
    st.markdown('<div class="section-label">Sensibilité au montant</div>', unsafe_allow_html=True)

    montants   = np.linspace(50_000, 1_000_000, 250)
    p_ret_c    = [predict(m, duree, taux_annuel, activite, region, get_categorie(m))[1] for m in montants]
    p_rem_c    = [predict(m, duree, taux_annuel, activite, region, get_categorie(m))[2] for m in montants]
    p_cou_c    = [predict(m, duree, taux_annuel, activite, region, get_categorie(m))[0] for m in montants]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=montants, y=p_ret_c, mode="lines",
                             line=dict(color="#f85149", width=2), name="EN RETARD"))
    fig.add_trace(go.Scatter(x=montants, y=p_rem_c, mode="lines",
                             line=dict(color="#3fb950", width=2), name="REMBOURSE"))
    fig.add_trace(go.Scatter(x=montants, y=p_cou_c, mode="lines",
                             line=dict(color="#58a6ff", width=1.5, dash="dot"), name="EN COURS"))
    point_color = {"EN COURS":"#58a6ff","EN RETARD":"#f85149","REMBOURSE":"#3fb950"}[classe]
    fig.add_trace(go.Scatter(
        x=[montant], y=[p_retard], mode="markers",
        marker=dict(color=point_color, size=12, symbol="circle",
                    line=dict(color="white", width=2)),
        name="Dossier actuel"
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", family="Syne"),
        xaxis=dict(title="Montant (CFA)", gridcolor="#21262d", tickformat=",.0f"),
        yaxis=dict(title="Probabilité", tickformat=".0%", range=[0,1], gridcolor="#21262d"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=40, l=10, r=10), height=280
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#21262d; margin:1.5rem 0 0.5rem;'>", unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
f1.caption("**Modèle** : Régression Logistique Multinomiale · Fairness-Aware (C=0.5)")
f2.caption("**Données** : 500 dossiers réels · Taux de défaut : 33.2%")
f3.caption("**Variables** : montant, taux, durée, activité, région, catégorie · Sans âge ni sexe")
