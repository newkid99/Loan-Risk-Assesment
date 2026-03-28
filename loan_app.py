"""
Loan Default Risk Assessment App (Ghana)
=========================================
Self-contained app - automatically trains model on first run.
Works directly on Streamlit Cloud!

Required file in same folder:
    - lending_club_sample_scored.parquet
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Loan Assessment - Ghana",
    page_icon="🇬🇭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# AUTO-TRAINING FUNCTION (runs once, then cached)
# ============================================================
@st.cache_resource
def load_or_train_model():
    """
    Automatically trains model on first run.
    Results are cached so training only happens once.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb
    
    FEATURES = [
        "loan_amnt", "term", "int_rate", "grade",
        "emp_length", "annual_inc", "dti",
        "delinq_2yrs", "open_acc", "revol_util",
        "finbert_pos", "finbert_neg", "finbert_neu"
    ]
    
    GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    
    DATA_FILE = "lending_club_sample_scored.parquet"
    
    if not Path(DATA_FILE).exists():
        return None, None, None, f"Data file not found: {DATA_FILE}"
    
    # Load data
    df = pd.read_parquet(DATA_FILE)
    
    # Clean term: " 36 months" -> 36.0
    def clean_term(val):
        if pd.isna(val): return np.nan
        digits = ''.join(c for c in str(val) if c.isdigit())
        return float(digits) if digits else np.nan
    df["term"] = df["term"].apply(clean_term)
    
    # Clean grade: "A" -> 1, "B" -> 2, etc.
    def clean_grade(val):
        if pd.isna(val): return np.nan
        s = str(val).strip().upper()
        return float(GRADE_MAP.get(s[0] if s else "", np.nan))
    df["grade"] = df["grade"].apply(clean_grade)
    
    # Clean percentages
    def clean_pct(val):
        if pd.isna(val): return np.nan
        try:
            return float(str(val).replace("%", "").strip())
        except:
            return np.nan
    df["int_rate"] = df["int_rate"].apply(clean_pct)
    df["revol_util"] = df["revol_util"].apply(clean_pct)
    
    # Clean employment length
    def clean_emp(val):
        if pd.isna(val): return np.nan
        s = str(val).lower().strip()
        if "< 1" in s: return 0.0
        if "10+" in s: return 10.0
        digits = ''.join(c for c in s if c.isdigit())
        return float(digits) if digits else np.nan
    df["emp_length"] = df["emp_length"].apply(clean_emp)
    
    # Ensure numeric columns
    for col in ["loan_amnt", "annual_inc", "dti", "delinq_2yrs", "open_acc",
                "finbert_pos", "finbert_neg", "finbert_neu"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Prepare features
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].copy()
    y = df["label"].copy()
    
    # Remove missing labels
    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).values
    
    # Store medians for later use
    medians = {}
    for col in available:
        med = X[col].median()
        medians[col] = float(med) if pd.notna(med) else 0.0
    
    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Class weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    # Calculate AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
    config = {
        "features": available,
        "medians": medians,
        "auc": auc,
        "n_samples": len(df)
    }
    
    return model, imputer, config, None


# ============================================================
# GRR DATA (Ghana Reference Rate)
# ============================================================
GRR_HISTORY = {
    "2025-01": 29.72, "2025-02": 29.96, "2025-03": 27.50,
    "2025-04": 26.20, "2025-05": 23.99, "2025-06": 22.50,
    "2025-07": 21.00, "2025-08": 19.67, "2025-09": 19.86,
    "2025-10": 17.86, "2025-11": 17.93, "2025-12": 15.90,
    "2026-01": 15.68, "2026-02": 14.58, "2026-03": 11.71,
}

def get_current_grr():
    now = datetime.now()
    current_key = now.strftime("%Y-%m")
    if current_key in GRR_HISTORY:
        return GRR_HISTORY[current_key], current_key
    sorted_keys = sorted(GRR_HISTORY.keys(), reverse=True)
    return GRR_HISTORY[sorted_keys[0]], sorted_keys[0]


# ============================================================
# STATISTICS
# ============================================================
PERFORMANCE_STATS = {
    "overall": {"approval_rate": 67.5, "default_rate": 18.2},
    "by_grade": {
        "A": {"approval_rate": 95, "default_rate": 5.2},
        "B": {"approval_rate": 88, "default_rate": 9.8},
        "C": {"approval_rate": 75, "default_rate": 15.3},
        "D": {"approval_rate": 58, "default_rate": 22.7},
        "E": {"approval_rate": 42, "default_rate": 31.5},
        "F": {"approval_rate": 28, "default_rate": 40.2},
        "G": {"approval_rate": 15, "default_rate": 48.8},
    },
}


# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    .main .block-container { padding: 1rem; max-width: 100%; }
    @media (min-width: 768px) { .main .block-container { padding: 2rem; max-width: 1200px; } }
    
    .main-header { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; text-align: center; margin-bottom: 0.5rem; }
    @media (min-width: 768px) { .main-header { font-size: 2.5rem; } }
    
    .sub-header { font-size: 0.95rem; color: #666; text-align: center; margin-bottom: 1.5rem; }
    
    .grr-card {
        background: linear-gradient(135deg, #006b3f 0%, #009955 100%);
        padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1rem;
    }
    
    .result-card { padding: 1.5rem; border-radius: 16px; color: white; text-align: center; margin-bottom: 1rem; }
    .risk-high { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }
    .risk-medium { background: linear-gradient(135deg, #f5af19 0%, #f12711 100%); }
    .risk-low { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    
    .affordability-good { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .affordability-tight { background: linear-gradient(135deg, #f5af19 0%, #f12711 100%); }
    .affordability-bad { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }
    
    .section-header { font-size: 1.2rem; font-weight: 600; color: #1a1a2e; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #e0e0e0; }
    
    .ghana-colors { background: linear-gradient(to right, #ce1126 33%, #fcd116 33%, #fcd116 66%, #006b3f 66%); height: 4px; border-radius: 2px; margin: 1rem 0; }
    
    .stNumberInput input { font-size: 16px !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def credit_score_to_risk_premium(score):
    if score >= 750: return 5.0
    elif score >= 700: return 7.0
    elif score >= 670: return 9.0
    elif score >= 640: return 11.0
    elif score >= 600: return 14.0
    elif score >= 560: return 17.0
    else: return 20.0

def credit_score_to_grade(score):
    if score >= 750: return 1, "A"
    elif score >= 700: return 2, "B"
    elif score >= 670: return 3, "C"
    elif score >= 640: return 4, "D"
    elif score >= 600: return 5, "E"
    elif score >= 560: return 6, "F"
    else: return 7, "G"

def calculate_affordability(monthly_income, monthly_expenses, monthly_payment):
    disposable = monthly_income - monthly_expenses
    if disposable <= 0:
        return {"status": "bad", "message": "Expenses exceed income!", "remaining": 0}
    
    ratio = (monthly_payment / disposable) * 100
    remaining = disposable - monthly_payment
    
    if ratio <= 30 and remaining >= monthly_income * 0.1:
        return {"status": "good", "message": "Fits comfortably in your budget.", "remaining": remaining}
    elif ratio <= 50 and remaining > 0:
        return {"status": "tight", "message": "Manageable but tight.", "remaining": remaining}
    else:
        return {"status": "bad", "message": "May strain your finances.", "remaining": remaining}

def save_assessment(data, result):
    if 'saved' not in st.session_state:
        st.session_state.saved = []
    st.session_state.saved.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data": data, "result": result, "id": len(st.session_state.saved) + 1
    })
    return len(st.session_state.saved)

LOAN_PURPOSES = [
    "Business Working Capital", "Debt Consolidation", "Home Improvement",
    "Education/School Fees", "Medical Expenses", "Vehicle Purchase",
    "Wedding/Funeral", "Agriculture/Farming", "Small Business",
    "Personal Emergency", "Rent/Housing", "Other"
]


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown('<div class="ghana-colors"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🇬🇭 Loan Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ghana Reference Rate Based • AI-Powered</p>', unsafe_allow_html=True)
    
    # Load/train model with spinner
    with st.spinner("🔄 Loading model (first time may take a moment)..."):
        model, imputer, config, error = load_or_train_model()
    
    if error:
        st.error(f"❌ {error}")
        st.info("""
        **To fix this on Streamlit Cloud:**
        1. Make sure `lending_club_sample_scored.parquet` is in your GitHub repo
        2. It should be in the same folder as `loan_app.py`
        3. Redeploy the app
        """)
        st.stop()
    
    features = config["features"]
    medians = config["medians"]
    
    # GRR
    current_grr, grr_month = get_current_grr()
    month_name = datetime.strptime(grr_month, "%Y-%m").strftime("%B %Y")
    
    st.markdown(f"""
    <div class="grr-card">
        <p style="margin:0;font-size:0.85rem;opacity:0.9;">🇬🇭 Ghana Reference Rate • {month_name}</p>
        <h2 style="margin:0.3rem 0;font-size:2.5rem;font-weight:700;">{current_grr:.2f}%</h2>
        <p style="margin:0;font-size:0.75rem;opacity:0.7;">Source: Ghana Association of Banks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model info
    st.caption(f"✅ Model loaded | AUC: {config['auc']:.3f} | Samples: {config['n_samples']:,}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Assessment", "📊 Statistics", "💾 Saved"])
    
    with tab1:
        st.markdown('<div class="section-header">Your Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amnt = st.number_input("💰 Loan Amount (GH₵)", 1000, 500000, 20000, 1000)
            term = st.select_slider("📅 Term", [12, 24, 36, 48, 60], 36, format_func=lambda x: f"{x} mo")
            purpose = st.selectbox("🎯 Purpose", LOAN_PURPOSES)
        
        with col2:
            monthly_income = st.number_input("💵 Monthly Income (GH₵)", 500, 100000, 5000, 500)
            credit_score = st.slider("📊 Credit Score", 300, 850, 650, 5)
            emp_opts = ["< 1 yr", "1 yr", "2 yrs", "3 yrs", "4 yrs", "5 yrs", "6-9 yrs", "10+ yrs"]
            emp_length = st.selectbox("💼 Employment", emp_opts, index=4)
        
        st.markdown('<div class="section-header">Credit & Debt</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            dti = st.slider("📉 DTI (%)", 0.0, 60.0, 20.0, 1.0)
            delinq = st.selectbox("⚠️ Late Payments", [0, 1, 2, 3, "4+"])
        with col4:
            revol_util = st.slider("💳 Credit Use (%)", 0.0, 100.0, 30.0, 5.0)
            open_acc = st.number_input("🏦 Accounts", 0, 30, 5)
        
        st.markdown('<div class="section-header">💰 Affordability</div>', unsafe_allow_html=True)
        monthly_expenses = st.number_input("Monthly Expenses (GH₵)", 0, 50000, 3000, 100)
        
        # Calculations
        grade_num, grade_letter = credit_score_to_grade(credit_score)
        risk_premium = credit_score_to_risk_premium(credit_score)
        estimated_rate = current_grr + risk_premium
        
        emp_map = {"< 1 yr": 0, "10+ yrs": 10, "6-9 yrs": 7}
        emp_num = emp_map.get(emp_length, int(emp_length.split()[0]))
        delinq_num = 4 if delinq == "4+" else int(delinq)
        
        # Payment calculation
        r = estimated_rate / 100 / 12
        if r > 0:
            payment = loan_amnt * (r * (1 + r)**term) / ((1 + r)**term - 1)
        else:
            payment = loan_amnt / term
        
        affordability = calculate_affordability(monthly_income, monthly_expenses, payment)
        
        st.info(f"**Rate:** {estimated_rate:.1f}% | **Payment:** GH₵{payment:,.0f}/mo | **Grade:** {grade_letter}")
        
        # Prediction
        st.markdown("---")
        
        input_data = {
            "loan_amnt": float(loan_amnt * 0.08),
            "term": float(term),
            "int_rate": float(estimated_rate),
            "grade": float(grade_num),
            "emp_length": float(emp_num),
            "annual_inc": float(monthly_income * 12 * 0.08),
            "dti": float(dti),
            "delinq_2yrs": float(delinq_num),
            "open_acc": float(open_acc),
            "revol_util": float(revol_util),
            "finbert_pos": 0.06, "finbert_neg": 0.06, "finbert_neu": 0.88,
        }
        
        X = np.array([[input_data.get(f, medians.get(f, 0)) for f in features]])
        X = imputer.transform(X)
        
        default_prob = model.predict_proba(X)[0][1]
        approval_prob = 1 - default_prob
        
        # Results
        st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        
        with c1:
            if default_prob >= 0.6: risk_class, label = "risk-high", "HIGH RISK"
            elif default_prob >= 0.4: risk_class, label = "risk-medium", "MODERATE"
            else: risk_class, label = "risk-low", "GOOD"
            
            st.markdown(f"""
            <div class="result-card {risk_class}">
                <p style="margin:0;opacity:0.9;">Approval Likelihood</p>
                <h2 style="margin:0.3rem 0;font-size:2.8rem;font-weight:700;">{approval_prob:.0%}</h2>
                <p style="margin:0;font-weight:600;">{label}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            aff_class = f"affordability-{affordability['status']}"
            emoji = {"good": "✅", "tight": "⚠️", "bad": "❌"}[affordability['status']]
            
            st.markdown(f"""
            <div class="result-card {aff_class}">
                <p style="margin:0;opacity:0.9;">Affordability</p>
                <h2 style="margin:0.3rem 0;font-size:2.8rem;">{emoji}</h2>
                <p style="margin:0;font-size:0.85rem;">{affordability['message']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary
        with st.expander("📋 Loan Summary"):
            total_int = payment * term - loan_amnt
            sc1, sc2 = st.columns(2)
            sc1.metric("Loan", f"GH₵{loan_amnt:,}")
            sc1.metric("Rate", f"{estimated_rate:.1f}%")
            sc2.metric("Interest", f"GH₵{total_int:,.0f}")
            sc2.metric("Total", f"GH₵{payment * term:,.0f}")
        
        # Stats comparison
        grade_stats = PERFORMANCE_STATS["by_grade"].get(grade_letter, {})
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(f"Grade {grade_letter} Approval", f"{grade_stats.get('approval_rate', 65)}%")
        mc2.metric(f"Grade {grade_letter} Default", f"{grade_stats.get('default_rate', 18)}%")
        mc3.metric("Remaining/Mo", f"GH₵{affordability['remaining']:,.0f}")
        
        # Save
        st.markdown("---")
        if st.button("💾 Save Assessment", use_container_width=True):
            sid = save_assessment(
                {"loan": loan_amnt, "term": term, "income": monthly_income, "score": credit_score},
                {"approval": approval_prob, "payment": payment, "rate": estimated_rate, "grade": grade_letter}
            )
            st.success(f"✅ Saved #{sid}!")
    
    with tab2:
        st.markdown('<div class="section-header">📊 Statistics</div>', unsafe_allow_html=True)
        st.write(f"**Market:** {PERFORMANCE_STATS['overall']['approval_rate']}% approval, {PERFORMANCE_STATS['overall']['default_rate']}% default")
        
        st.markdown("### GRR Trend")
        grr_df = pd.DataFrame([{"Month": datetime.strptime(k, "%Y-%m").strftime("%b %y"), "GRR": v} for k, v in sorted(GRR_HISTORY.items())])
        st.line_chart(grr_df.set_index("Month"))
        
        st.markdown("### Approval by Grade")
        gdf = pd.DataFrame([{"Grade": k, "Rate": v["approval_rate"]} for k, v in PERFORMANCE_STATS["by_grade"].items()])
        st.bar_chart(gdf.set_index("Grade"))
    
    with tab3:
        st.markdown('<div class="section-header">💾 Saved</div>', unsafe_allow_html=True)
        saved = st.session_state.get('saved', [])
        if not saved:
            st.info("No saved assessments yet.")
        else:
            for a in reversed(saved):
                with st.expander(f"#{a['id']} - {a['time']} | GH₵{a['data']['loan']:,}"):
                    st.write(f"**Approval:** {a['result']['approval']:.0%} | **Rate:** {a['result']['rate']:.1f}% | **Grade:** {a['result']['grade']}")
    
    st.markdown("---")
    st.caption("⚠️ Educational purposes only. Not financial advice.")


if __name__ == "__main__":
    main()
