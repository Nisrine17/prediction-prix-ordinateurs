import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. CONFIG
# ==========================================
st.set_page_config(
    page_title="Laptop Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. STYLE GLOBAL
# ==========================================
st.markdown("""
<style>
    .stApp { background: #0a0c10; }

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    .card {
        background: #11151c;
        border-radius: 18px;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,0.05);
    }

    [data-testid="stMetric"] {
        background: #11151c;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
    }

    [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HEADER
# ==========================================
st.markdown("""
<div class="main-header">
    <h1 style="color:white;margin:0;">💻 Laptop Price Predictor</h1>
    <p style="color:rgba(255,255,255,0.8);margin-top:0.5rem;">
        Estimation intelligente par Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. PROFILS
# ==========================================
PROFILES = {
    "🎓 Étudiant": [14, 8, 2.2, 1.7, 1920, 1080],
    "💼 Bureautique": [15.6, 8, 2.5, 2.0, 1920, 1080],
    "🎮 Gamer": [15.6, 16, 3.5, 2.5, 1920, 1080],
    "⭐ Premium": [16, 32, 3.8, 1.8, 3840, 2160],
}

# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")

    profile = st.selectbox("Profil", ["✨ Personnalisé"] + list(PROFILES.keys()))
    taux = st.slider("INR → EUR", 80.0, 100.0, 91.5)

# ==========================================
# 6. VALEURS
# ==========================================
default = PROFILES[profile] if profile in PROFILES else [15.6, 8, 2.5, 2.0, 1920, 1080]

# ==========================================
# 7. INPUTS
# ==========================================
col1, col2 = st.columns(2)

with col1:
    inches = st.slider("📏 Écran", 10.0, 18.0, float(default[0]))
    ram = st.select_slider("💾 RAM", [2, 4, 8, 16, 32, 64], int(default[1]))
    cpu = st.slider("⚡ CPU", 1.0, 4.5, float(default[2]))

with col2:
    weight = st.slider("⚖️ Poids", 0.5, 3.5, float(default[3]))
    res_x = st.selectbox("Résolution X", [1366, 1920, 2560, 3840])
    res_y = st.selectbox("Résolution Y", [768, 1080, 1440, 2160])

# ==========================================
# 8. MODEL
# ==========================================
@st.cache_resource
def load_model():
    try:
        return joblib.load("random_forest_model.pkl")
    except:
        return None

model = load_model()

# ==========================================
# 9. PERFORMANCE FUNCTION
# ==========================================
def compute_performance(ram, cpu, res_x, weight):
    ram_s = min(ram / 64, 1) * 35
    cpu_s = min(cpu / 4.5, 1) * 35
    res_s = min(res_x / 3840, 1) * 20
    weight_s = max(0, 1 - weight / 3.5) * 10
    return int(ram_s + cpu_s + res_s + weight_s)

# ==========================================
# 10. PREDICTION
# ==========================================
input_df = pd.DataFrame([[inches, ram, cpu, weight, res_x, res_y]],
                         columns=["Inches", "Ram_GB", "Cpu_GHz", "Weight_value", "Resolution_x", "Resolution_y"])

price_inr = model.predict(input_df)[0] if model else ram * 4000 + cpu * 12000
price_eur = price_inr / taux

performance = compute_performance(ram, cpu, res_x, weight)
value_score = round(price_eur / max(performance, 1), 1)

# ==========================================
# 11. CATEGORY
# ==========================================
if price_eur < 500:
    category, color = "Entrée de gamme", "#10b981"
elif price_eur < 1000:
    category, color = "Milieu de gamme", "#3b82f6"
elif price_eur < 1800:
    category, color = "Haut de gamme", "#f59e0b"
else:
    category, color = "Premium", "#ef4444"

# ==========================================
# 12. INTERPRETATION SIMPLE
# ==========================================
if performance < 40:
    text = "Usage basique (navigation, bureautique)"
elif performance < 70:
    text = "Usage polyvalent (travail, études)"
else:
    text = "Usage intensif (gaming, création)"

# ==========================================
# 13. OUTPUT
# ==========================================
st.markdown("### 💰 Résultat")

st.markdown(f"""
<div class="card" style="text-align:center;">
    <div style="font-size:0.8rem;color:#888;">PRIX ESTIMÉ</div>
    <div style="font-size:3rem;font-weight:700;color:white;">
        {price_eur:,.0f} €
    </div>
    <div style="color:#777;">≈ {price_inr:,.0f} INR</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center;margin-top:1rem;">
    <span style="background:{color}22;color:{color};padding:6px 14px;border-radius:20px;">
        {category}
    </span>
</div>
""", unsafe_allow_html=True)

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("⚡ Perf", f"{performance}/100")
c2.metric("💎 €/Perf", value_score)
c3.metric("⚖️ Poids", f"{weight} kg")

# Insight
st.markdown(f"""
<div class="card" style="margin-top:1rem;">
    <b>Analyse :</b> {text}
</div>
""", unsafe_allow_html=True)

# ==========================================
# 14. GAUGE SIMPLE & CLEAN
# ==========================================
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=performance,
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#667eea"},
        "steps": [
            {"range": [0, 40], "color": "#ef4444"},
            {"range": [40, 70], "color": "#f59e0b"},
            {"range": [70, 100], "color": "#10b981"},
        ],
    }
))

fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10),
                  paper_bgcolor="rgba(0,0,0,0)")

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})



# ==========================================
# 14. DÉTAILS TECHNIQUES
# ==========================================
with st.expander("📋 Détails techniques"):

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Configuration saisie**")
        st.write(f"- Écran : {inches}″")
        st.write(f"- RAM : {ram} Go")
        st.write(f"- CPU : {cpu} GHz")
        st.write(f"- Poids : {weight} kg")
        st.write(f"- Résolution : {res_x} × {res_y}")

    with col_b:
        st.markdown("**Résultats du modèle**")
        st.write(f"- Prix estimé : {price_eur:,.0f} €")
        st.write(f"- Prix INR : {price_inr:,.0f} ₹")
        st.write(f"- Performance : {performance}/100")
        st.write(f"- Ratio €/perf : {value_score}")
        st.write(f"- Catégorie : {category}")

# ==========================================
# 15. FOOTER
# ==========================================
st.divider()
st.markdown("<div style='text-align:center;color:#444;'>💻 Laptop Predictor • ML Project</div>",
            unsafe_allow_html=True)
