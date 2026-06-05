genre je le remplace par ça import joblib
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
    
    .stSlider { padding-top: 0.5rem; }
    
    .chat-message {
        background: #1a1f2e;
        border-radius: 12px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
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
    
    st.divider()
    st.caption("📊 Modèle: Random Forest")
    st.caption("🎯 Précision: ~85%")

# ==========================================
# 6. VALEURS
# ==========================================
if profile != "✨ Personnalisé":
    default = PROFILES[profile]
else:
    default = [15.6, 8, 2.5, 2.0, 1920, 1080]

# ==========================================
# 7. INPUTS
# ==========================================
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown("### 📊 Configuration")
    
    col_a, col_b = st.columns(2)
    with col_a:
        inches = st.slider("📏 Écran (pouces)", 10.0, 18.0, float(default[0]), format="%.1f")
        ram = st.select_slider("💾 RAM (Go)", [2, 4, 8, 16, 32, 64], int(default[1]))
        cpu = st.slider("⚡ CPU (GHz)", 1.0, 4.5, float(default[2]), format="%.1f")
    with col_b:
        weight = st.slider("⚖️ Poids (kg)", 0.5, 3.5, float(default[3]), format="%.1f")
        res_x = st.selectbox("🖥️ Résolution X", [1366, 1920, 2560, 3840], index=1)
        res_y = st.selectbox("📐 Résolution Y", [768, 1080, 1440, 2160], index=1)
    
    # Guide rapide
    st.markdown("---")
    st.markdown("### 💡 Guide rapide")
    
    ram_temp = min(ram / 64, 1) * 35
    cpu_temp = min(cpu / 4.5, 1) * 35
    res_temp = min(res_x / 3840, 1) * 20
    weight_temp = max(0, 1 - weight / 3.5) * 10
    perf_preview = int(ram_temp + cpu_temp + res_temp + weight_temp)
    
    st.markdown(f"""
    <div style="margin: 0.5rem 0;">
        <div style="font-size: 0.7rem; color: #888;">Performance actuelle</div>
        <div style="background: #1f2937; border-radius: 20px; height: 6px; margin: 0.3rem 0;">
            <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: {perf_preview}%; height: 6px; border-radius: 20px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.6rem; color: #555;">
            <span>📱 Basique</span>
            <span>💻 Polyvalent</span>
            <span>🚀 Intensif</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("• **RAM**: 8Go minimum, 16Go recommandé")
    st.caption("• **CPU**: +3GHz pour gaming/création")
    st.caption("• **Poids**: <1.8kg pour ultraportable")

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

if model:
    price_inr = model.predict(input_df)[0]
else:
    price_inr = ram * 4000 + cpu * 12000

price_eur = price_inr / taux
performance = compute_performance(ram, cpu, res_x, weight)
value_score = round(price_eur / max(performance, 1), 1)

# ==========================================
# 11. CATEGORY
# ==========================================
if price_eur < 500:
    category, color = "📱 Entrée de gamme", "#10b981"
elif price_eur < 1000:
    category, color = "💻 Milieu de gamme", "#3b82f6"
elif price_eur < 1800:
    category, color = "⚡ Haut de gamme", "#f59e0b"
else:
    category, color = "💎 Premium", "#ef4444"

# ==========================================
# 12. INTERPRETATION
# ==========================================
if performance < 40:
    usage = "📱 Usage basique"
    detail = "Navigation, mails, bureautique légère"
elif performance < 70:
    usage = "💻 Usage polyvalent"
    detail = "Travail, études, multimédia"
else:
    usage = "🚀 Usage intensif"
    detail = "Gaming, création, développement"

# Équivalent marché
if price_eur < 500:
    market = "Chromebook / Notebook"
elif price_eur < 800:
    market = "Lenovo IdeaPad / HP Pavilion"
elif price_eur < 1200:
    market = "Dell XPS 13 / MacBook Air"
elif price_eur < 1800:
    market = "MacBook Pro 14 / Dell XPS 15"
else:
    market = "MacBook Pro 16 / Razer Blade"

# ==========================================
# 13. CHATBOT INTELLIGENT
# ==========================================
def smart_chatbot(question, ram, cpu, performance, price_eur, category):
    """Chatbot intelligent qui répond aux questions sur la configuration"""
    question = question.lower()
    
    # Dictionnaire des réponses
    reponses = {
        "ram": f"💾 **RAM** : {ram} Go. " + (
            "C'est parfait pour du multitâche et le gaming !" if ram >= 16 else
            "C'est bien pour un usage basique. Pour du gaming, passez à 16 Go." if ram >= 8 else
            "C'est un peu juste. Je recommande au moins 8 Go."
        ),
        "cpu": f"⚡ **CPU** : {cpu} GHz. " + (
            "Excellent pour les tâches exigeantes !" if cpu >= 3.5 else
            "Bon équilibre pour un usage polyvalent." if cpu >= 2.5 else
            "Suffisant pour la bureautique, mais limité pour le gaming."
        ),
        "poids": f"⚖️ **Poids** : {weight} kg. " + (
            "Ultraportable ! Facile à transporter." if weight < 1.5 else
            "Poids standard, bon compromis." if weight < 2.2 else
            "Assez lourd, plutôt pour un usage fixe."
        ),
        "performance": f"📊 **Score** : {performance}/100. " + (
            "Machine très puissante !" if performance >= 70 else
            "Bonnes performances pour un usage quotidien." if performance >= 40 else
            "Conçue pour les tâches essentielles."
        ),
        "prix": f"💰 **Prix** : {price_eur:,.0f} €. Catégorie : {category}.",
    }
    
    # Mots-clés et réponses
    if "ram" in question or "mémoire" in question:
        return reponses["ram"]
    elif "cpu" in question or "processeur" in question:
        return reponses["cpu"]
    elif "poids" in question:
        return reponses["poids"]
    elif "performance" in question or "perf" in question:
        return reponses["performance"]
    elif "prix" in question or "budget" in question or "coût" in question or "euro" in question:
        return reponses["prix"]
    
    # Recommandations personnalisées
    elif "recommandation" in question or "conseil" in question or "améliorer" in question:
        suggestions = []
        if ram < 16:
            suggestions.append("✅ Ajoutez plus de RAM (16 Go recommandé)")
        if cpu < 3.0:
            suggestions.append("✅ Choisissez un CPU plus rapide (+3.0 GHz)")
        if weight > 2.2:
            suggestions.append("✅ Optez pour un modèle plus léger si vous voyagez")
        
        if suggestions:
            return "💡 **Mes conseils pour améliorer cette config :**\n" + "\n".join(suggestions)
        else:
            return "⭐ **Excellente configuration !** Vous avez fait les bons choix."
    
    # Comparaison
    elif "comparer" in question or "macbook" in question or "dell" in question or "lenovo" in question:
        return f"🔍 **Comparaison** : Votre PC à {price_eur:,.0f} € (perf {performance}/100) se situe dans le {category}. Un MacBook Air M2 équivalent coûte ~1200€ (perf ~75/100)."
    
    # Aide générale
    elif "aide" in question or "que faire" in question:
        return "🤖 **Je peux vous aider sur :**\n• RAM, CPU, Poids, Performance, Prix\n• Recommandations pour améliorer la config\n• Comparaisons avec d'autres PC"
    
    elif "gaming" in question or "jeu" in question:
        if ram >= 16 and cpu >= 3.0:
            return "🎮 **Gaming** : Cette config est parfaite pour jouer ! Vous pourrez faire tourner la plupart des jeux récents."
        else:
            return "🎮 **Gaming** : Pour une meilleure expérience, je recommande au moins 16 Go de RAM et un CPU à 3.0 GHz+."
    
    elif "montage" in question or "vidéo" in question:
        if ram >= 16 and cpu >= 3.0:
            return "🎬 **Montage vidéo** : Cette config est adaptée pour du montage 1080p/4K léger !"
        else:
            return "🎬 **Montage vidéo** : Pour du montage confortable, visez 16 Go de RAM et un CPU rapide."
    
    elif "bureautique" in question or "travail" in question:
        return "💼 **Bureautique** : Cette config est parfaitement adaptée pour Word, Excel, navigation web et visioconférence."
    
    else:
        return "🤖 **Posez-moi une question sur :** RAM, CPU, poids, performance, prix, gaming, montage vidéo, ou demandez une recommandation !"

# ==========================================
# 14. OUTPUT COLONNE DROITE
# ==========================================
with col_right:
    st.markdown("### 💰 Résultat")
    
    # Carte prix
    st.markdown(f"""
    <div class="card" style="text-align:center;">
        <div style="font-size:0.8rem;color:#888;">PRIX ESTIMÉ</div>
        <div style="font-size:3rem;font-weight:700;background:linear-gradient(135deg, #667eea, #764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            {price_eur:,.0f} €
        </div>
        <div style="color:#777;">≈ {price_inr:,.0f} INR</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Badge catégorie
    st.markdown(f"""
    <div style="text-align:center;margin:1rem 0;">
        <span style="background:{color}22;color:{color};padding:6px 16px;border-radius:20px;font-size:0.85rem;">
            {category}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # 3 métriques
    c1, c2, c3 = st.columns(3)
    c1.metric("⚡ Perf", f"{performance}/100")
    c2.metric("💎 €/Perf", f"{value_score}")
    c3.metric("⚖️ Poids", f"{weight} kg")
    
    # Carte usage
    st.markdown(f"""
    <div class="card" style="margin:0.5rem 0;">
        <div style="font-size:0.7rem;color:#888;">🎯 RECOMMANDATION</div>
        <div style="font-size:1rem;font-weight:500;">{usage}</div>
        <div style="font-size:0.8rem;color:#aaa;margin-top:0.3rem;">{detail}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Carte équivalent marché
    st.markdown(f"""
    <div class="card" style="margin:0.5rem 0;">
        <div style="font-size:0.7rem;color:#888;">🏷️ ÉQUIVALENT MARCHÉ</div>
        <div style="font-size:0.9rem;">{market}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==========================================
    # 15. CHATBOT
    # ==========================================
    with st.expander("💬 Assistant PC - Posez votre question", expanded=False):
        st.markdown("""
        <div style="background:#1a1f2e; border-radius:12px; padding:0.8rem; margin-bottom:0.8rem;">
            <div style="font-size:0.8rem; color:#888;">🤖 Je peux vous aider sur :</div>
            <div style="font-size:0.75rem;">• RAM, CPU, Poids, Performance, Prix</div>
            <div style="font-size:0.75rem;">• Recommandations pour améliorer la config</div>
            <div style="font-size:0.75rem;">• Gaming, Montage vidéo, Bureautique</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialiser l'historique du chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Afficher l'historique
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Input utilisateur
        user_question = st.chat_input("Posez votre question...")
        
        if user_question:
            # Ajouter message utilisateur
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)
            
            # Générer réponse
            response = smart_chatbot(user_question, ram, cpu, performance, price_eur, category)
            
            # Ajouter réponse
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
    
    # Jauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=performance,
        title={"text": "Score Performance", "font": {"size": 12, "color": "#888"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555", "tickfont": {"color": "#888", "size": 10}},
            "bar": {"color": "#667eea", "thickness": 0.3},
            "bgcolor": "#1a1f2e",
            "steps": [
                {"range": [0, 40], "color": "#ef4444"},
                {"range": [40, 70], "color": "#f59e0b"},
                {"range": [70, 100], "color": "#10b981"},
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ==========================================
# 16. DETAILS TECHNIQUES
# ==========================================
with st.expander("📋 Détails techniques"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Configuration :**")
        st.write(f"- Écran : {inches}″")
        st.write(f"- RAM : {ram} Go")
        st.write(f"- CPU : {cpu} GHz")
        st.write(f"- Poids : {weight} kg")
        st.write(f"- Résolution : {res_x}×{res_y}")
    with col_b:
        st.write("**Résultats :**")
        st.write(f"- Prix : {price_eur:,.0f} €")
        st.write(f"- Performance : {performance}/100")
        st.write(f"- Catégorie : {category}")

# ==========================================
# 17. FOOTER
# ==========================================
st.divider()
st.markdown("<div style='text-align:center;color:#444;font-size:0.75rem;'>💻 Laptop Predictor • Random Forest • Data Science Project</div>", unsafe_allow_html=True)
