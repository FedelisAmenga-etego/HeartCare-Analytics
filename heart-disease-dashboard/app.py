# app.py
# Professional HeartCare Analytics Dashboard
# Modern design inspired by professional SaaS platforms

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
from scipy import stats

st.set_page_config(page_title="HeartCare Analytics", layout="wide", page_icon="❤️", initial_sidebar_state="expanded")

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body, html {
        background: #F8F9FB;
        color: #1F2937;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    
    .main {
        background: #F8F9FB;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        color: white;
        padding: 40px 60px;
        border-radius: 12px;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    
    .header-container h1 {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .header-container p {
        font-size: 16px;
        opacity: 0.95;
        letter-spacing: 0.5px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #E5E7EB;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 2px solid #E5E7EB;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        background: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        color: #6B7280;
        font-weight: 600;
        padding: 12px 24px;
        font-size: 14px;
        border-radius: 0;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #667EEA;
        border-bottom-color: #667EEA;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #667EEA;
        border-bottom-color: #667EEA;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 12px 24px;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Text Styling */
    h1, h2, h3 {
        color: #1F2937;
        font-weight: 700;
    }
    
    h2 {
        margin-top: 32px;
        margin-bottom: 24px;
        font-size: 24px;
    }
    
    h3 {
        margin-top: 24px;
        margin-bottom: 16px;
        font-size: 18px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: white;
        border-right: 1px solid #E5E7EB;
    }
    
    .stSidebar [data-testid="stSidebarNav"] {
        background: white;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 800;
        color: #667EEA;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Info/Success/Error Boxes */
    .stInfo, .stSuccess, .stError, .stWarning {
        background: white;
        border-left: 4px solid #667EEA;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }
    
    .stSuccess {
        border-left-color: #10B981;
    }
    
    .stError {
        border-left-color: #EF4444;
    }
    
    .stWarning {
        border-left-color: #F59E0B;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: white;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Divider */
    .stDivider {
        border-color: #E5E7EB;
        margin: 32px 0;
    }
    
    /* Input Elements */
    .stNumberInput input, .stSelectbox select, .stSlider {
        border-radius: 8px;
        border: 1px solid #D1D5DB;
        padding: 10px 12px;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #667EEA;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Labels */
    .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #374151;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Card Container */
    .card-container {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #E5E7EB;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
DATA_PATH = os.path.join("data", "heart.csv")
MODEL_PATH = "heart_disease_model.pkl"

# ==================== UTILITY FUNCTIONS ====================
@st.cache_data(show_spinner=False)
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def basic_cleaning(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.strip()
            except:
                pass
    for col in df.columns:
        if col not in ['cp','sex','fbs','restecg','exang','slope','ca','thal','target'] and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'target' in df.columns:
        df = df.dropna(subset=['target'])
        df['target'] = df['target'].astype(int)
    return df

def create_features(df, use_tanaka=False):
    df = df.copy()
    df['chol_age_ratio'] = df['chol'] / (df['age'].replace(0, np.nan))
    df['predicted_max_hr'] = (208 - 0.7 * df['age']) if use_tanaka else (220 - df['age'])
    df['heart_rate_reserve'] = df['predicted_max_hr'] - df['thalach']
    df['age_trestbps_interaction'] = df['age'] * df['trestbps']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['chol_age_ratio'] = df['chol_age_ratio'].fillna(df['chol_age_ratio'].median())
    df['heart_rate_reserve'] = df['heart_rate_reserve'].fillna(df['heart_rate_reserve'].median())
    return df

def get_column_transformer(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

def build_and_train_pipeline(df, numeric_cols, categorical_cols, model_path=MODEL_PATH):
    X = df[numeric_cols + categorical_cols].copy()
    y = df['target'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = get_column_transformer(numeric_cols, categorical_cols)
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_clf)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    joblib.dump(pipeline, model_path)
    return pipeline, metrics, X_test, y_test, cm

# ==================== LOAD DATA ====================
df = load_data()
if df is None:
    st.error(f"Dataset not found at {DATA_PATH}")
    st.stop()

df = basic_cleaning(df)
st.session_state.setdefault('use_tanaka', False)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### HeartCare Analytics")
    st.divider()
    
    st.markdown("#### Filters")
    age_min, age_max = int(df['age'].min()), int(df['age'].max())
    age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))
    sex_filter = st.multiselect("Sex", ['Male','Female'], default=['Male','Female'])
    cp_filter = st.multiselect("Chest Pain Type", sorted(df['cp'].unique()), default=sorted(df['cp'].unique()))
    
    st.divider()
    st.caption("Built by Fedelis | HeartCare Analytics Platform")

# ==================== PREPARE DATA ====================
df = create_features(df, use_tanaka=st.session_state['use_tanaka'])
df['sex_str'] = df['sex'].map({1:'Male', 0:'Female'}) if df['sex'].dtype != object else df['sex']
df_filtered = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) & 
                 (df['sex_str'].isin(sex_filter)) & (df['cp'].isin(cp_filter))].copy()

numeric_cols = ['age','trestbps','chol','thalach','oldpeak','chol_age_ratio','heart_rate_reserve','age_trestbps_interaction']
categorical_cols = [c for c in ['sex','cp','fbs','restecg','exang','slope','ca','thal'] if c in df.columns]

# ==================== MODEL ====================
model_exists = os.path.exists(MODEL_PATH)
pipeline = metrics = X_test = y_test = cm = None

if model_exists:
    try:
        pipeline = joblib.load(MODEL_PATH)
    except:
        pipeline, metrics, X_test, y_test, cm = build_and_train_pipeline(df, numeric_cols, categorical_cols)
else:
    with st.spinner("Training model..."):
        pipeline, metrics, X_test, y_test, cm = build_and_train_pipeline(df, numeric_cols, categorical_cols)

if metrics is None and pipeline is not None:
    try:
        X = df[numeric_cols + categorical_cols].copy()
        y = df['target'].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:,1]
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
    except:
        metrics = {}

# ==================== HEADER ====================
st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); 
            color: white; padding: 32px 40px; border-radius: 12px; margin-bottom: 32px; 
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);'>
    <h1 style='font-size: 36px; font-weight: 800; margin: 0 0 8px 0;'>❤️ HeartCare Analytics</h1>
    <p style='font-size: 15px; opacity: 0.95; margin: 0; letter-spacing: 0.5px;'>Advanced Heart Disease Risk Assessment Platform</p>
</div>
""", unsafe_allow_html=True)

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Prediction", "Model Performance", "Analytics", "Insights"])

# ==================== TAB 1: DASHBOARD ====================
with tab1:
    total = len(df)
    cases = int(df['target'].sum())
    pct = cases / total * 100
    
    col1, col2, col3, col4 = st.columns(4, gap="large")
    with col1:
        st.metric("Total Records", f"{total:,}")
    with col2:
        st.metric("Disease Cases", f"{cases:,}", delta=f"{pct:.1f}%")
    with col3:
        st.metric("Healthy", f"{total - cases:,}", delta=f"{100-pct:.1f}%")
    with col4:
        disease_rate = int(df_filtered['target'].sum()) / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
        st.metric("Filtered Disease Rate", f"{disease_rate:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig = px.histogram(df_filtered, x='age', nbins=20, title="Age Distribution",
                          labels={'age': 'Age (years)', 'count': 'Count'}, template="plotly_white",
                          color_discrete_sequence=['#667EEA'])
        fig.update_layout(height=350, showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df_filtered, x='chol', nbins=25, title="Cholesterol Distribution",
                          labels={'chol': 'Cholesterol (mg/dl)', 'count': 'Count'}, template="plotly_white",
                          color_discrete_sequence=['#764BA2'])
        fig.update_layout(height=350, showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig = px.histogram(df_filtered, x='cp', color='target', title='Chest Pain Type vs Disease',
                          labels={'cp': 'Type', 'count': 'Count'}, template='plotly_white',
                          color_discrete_map={0: '#10B981', 1: '#EF4444'}, barmode='group')
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        sex_data = pd.crosstab(df['sex_str'], df['target'], normalize='index') * 100
        fig = px.bar(sex_data, title='Disease Prevalence by Sex', template='plotly_white',
                    color_discrete_sequence=['#10B981', '#EF4444'], barmode='group')
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: PREDICTION ====================
with tab2:
    st.markdown("### Patient Risk Assessment")
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### Patient Information")
        age = st.number_input("Age (years)", int(df['age'].min()), int(df['age'].max()), int(df['age'].median()))
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", sorted(df['cp'].unique()))
        trestbps = st.number_input("Resting BP (mmHg)", int(df['trestbps'].min()), int(df['trestbps'].max()), int(df['trestbps'].median()))
        chol = st.number_input("Cholesterol (mg/dl)", int(df['chol'].min()), int(df['chol'].max()), int(df['chol'].median()))
        fbs = st.selectbox("Fasting Blood Sugar > 120", sorted(df['fbs'].unique()))
    
    with col2:
        st.markdown("#### Clinical Measurements")
        restecg = st.selectbox("Resting ECG", sorted(df['restecg'].unique()))
        thalach = st.number_input("Max Heart Rate (bpm)", int(df['thalach'].min()), int(df['thalach'].max()), int(df['thalach'].median()))
        exang = st.selectbox("Exercise Induced Angina", sorted(df['exang'].unique()))
        oldpeak = st.number_input("ST Depression", float(df['oldpeak'].min()), float(df['oldpeak'].max()), float(df['oldpeak'].median()), step=0.1)
        slope = st.selectbox("Slope", sorted(df['slope'].unique()))
        ca = st.selectbox("Major Vessels", sorted(df['ca'].unique()))
        thal = st.selectbox("Thalassemia", sorted(df['thal'].unique()))
    
    sex_val = 1 if sex == "Male" else 0
    input_df = pd.DataFrame([{'age': age, 'sex': sex_val, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                              'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
                              'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}])
    
    predicted_max_hr = (208 - 0.7 * age) if st.session_state['use_tanaka'] else (220 - age)
    input_df['chol_age_ratio'] = chol / age
    input_df['predicted_max_hr'] = predicted_max_hr
    input_df['heart_rate_reserve'] = predicted_max_hr - thalach
    input_df['age_trestbps_interaction'] = age * trestbps
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 Analyze Risk", use_container_width=True):
            button_clicked = True
        else:
            button_clicked = False
    
    if button_clicked:
        if pipeline is None:
            st.error("Model not available")
        else:
            for col in numeric_cols + categorical_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            X_input = input_df[numeric_cols + categorical_cols]
            pred = pipeline.predict(X_input)[0]
            proba = pipeline.predict_proba(X_input)[0,1]
            
            # Risk Score Display - Semi-circle Donut
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if pred == 1:
                    color = "#EF4444"
                    status = "HIGH RISK"
                    icon = "⚠️"
                else:
                    color = "#10B981"
                    status = "LOW RISK"
                    icon = "✓"
                
                # Create semi-circle gauge chart
                percentage = proba * 100
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=percentage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'size': 20}},
                    number={'font': {'size': 40, 'color': color}, 'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#E5E7EB"},
                        'bar': {'color': color, 'thickness': 0.3},
                        'steps': [
                            {'range': [0, 30], 'color': "#D1FAE5", 'thickness': 0.4},
                            {'range': [30, 70], 'color': "#FEF3C7", 'thickness': 0.4},
                            {'range': [70, 100], 'color': "#FEE2E2", 'thickness': 0.4}
                        ],
                        'threshold': {
                            'line': {'color': color, 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=400, template='plotly_white', margin=dict(l=0,r=0,t=60,b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                <div style='text-align: center; margin-top: 16px;'>
                    <div style='font-size: 16px; font-weight: 700; color: {color};'>{status}</div>
                    <div style='font-size: 13px; color: #6B7280; margin-top: 4px;'>Probability of Heart Disease</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Risk Factors
            col1, col2 = st.columns([1.5, 1], gap="large")
            
            with col1:
                st.markdown("#### Risk Factors Detected")
                risks = []
                if age > 60: risks.append(("🔴", "Advanced age", f"{age} years"))
                if chol > 240: risks.append(("🔴", "High cholesterol", f"{chol} mg/dl"))
                elif chol > 200: risks.append(("🟡", "Borderline cholesterol", f"{chol} mg/dl"))
                if trestbps > 140: risks.append(("🔴", "High blood pressure", f"{trestbps} mmHg"))
                elif trestbps > 130: risks.append(("🟡", "Elevated blood pressure", f"{trestbps} mmHg"))
                if thalach < 100: risks.append(("🔴", "Low max heart rate", f"{thalach} bpm"))
                if oldpeak > 2.0: risks.append(("🔴", "High ST depression", f"{oldpeak}"))
                if exang == 1: risks.append(("🔴", "Exercise-induced angina", "Present"))
                
                if risks:
                    for icon, label, value in risks:
                        st.write(f"{icon} **{label}**: {value}")
                else:
                    st.success("✓ No significant risk factors detected")
            
            with col2:
                st.markdown("#### Patient Summary")
                summary_data = {
                    "Age": f"{age} yrs",
                    "Sex": sex,
                    "Cholesterol": f"{chol} mg/dl",
                    "Resting BP": f"{trestbps} mmHg",
                    "Max HR": f"{thalach} bpm"
                }
                for key, value in summary_data.items():
                    st.write(f"**{key}:** {value}")
            
            st.divider()
            
            # Feature Importance
            st.markdown("#### Top Contributing Factors")
            try:
                if X_test is not None and y_test is not None:
                    preprocessor = pipeline.named_steps['preprocessor']
                    feat_names = preprocessor.get_feature_names_out()
                    
                    r = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
                    
                    n_features = len(r.importances_mean)
                    feat_names = feat_names[:n_features]
                    
                    importances = pd.Series(r.importances_mean, index=feat_names).sort_values(ascending=True).tail(10)
                    
                    fig = px.bar(importances, x=importances.values, y=importances.index, orientation='h',
                                title='Feature Importance (Permutation Analysis)', template='plotly_white',
                                labels={'x': 'Importance Score', 'y': 'Feature'},
                                color=importances.values, color_continuous_scale='Purples')
                    fig.update_layout(height=400, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not compute feature importances")

# ==================== TAB 3: MODEL PERFORMANCE ====================
with tab3:
    col1, col2, col3, col4, col5 = st.columns(5, gap="large")
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    with col4:
        st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
    with col5:
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    
    st.divider()
    st.markdown("### Model Configuration")
    st.info(f"""
    **Algorithm:** Random Forest (200 trees) | **Training:** 80/20 Split  
    **Features:** {len(numeric_cols + categorical_cols)} total ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)  
    **Samples:** {len(df):,} training | Preprocessing: StandardScaler + OneHotEncoder
    """)
    
    if cm is not None:
        st.markdown("### Confusion Matrix")
        cm_df = pd.DataFrame(cm, index=['No Disease', 'Disease'], columns=['Pred: No Disease', 'Pred: Disease'])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix', template='plotly_white')
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: ANALYTICS ====================
with tab4:
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        diff = df[df['target']==1]['age'].mean() - df[df['target']==0]['age'].mean()
        st.metric("Age Difference", f"{diff:.1f} yrs", delta="Disease group higher")
    with col2:
        diff = df[df['target']==1]['chol'].mean() - df[df['target']==0]['chol'].mean()
        st.metric("Cholesterol Difference", f"{diff:.0f} mg/dl", delta="Disease group higher")
    with col3:
        diff = df[df['target']==1]['thalach'].mean() - df[df['target']==0]['thalach'].mean()
        st.metric("Max HR Difference", f"{diff:.0f} bpm", delta="Disease group lower")
    
    st.divider()
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        df['age_group'] = pd.cut(df['age'], bins=[0,30,40,50,60,100], labels=['<30','30-40','40-50','50-60','60+'])
        age_risk = df.groupby('age_group')['target'].agg(['sum','count'])
        age_risk['pct'] = age_risk['sum']/age_risk['count']*100
        fig = px.bar(age_risk.reset_index(), x='age_group', y='pct', title='Disease Rate by Age Group',
                    labels={'age_group': 'Age', 'pct': 'Rate (%)'}, template='plotly_white',
                    color='pct', color_continuous_scale='Reds')
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df_filtered, x='chol', y='thalach', color='target',
                        title='Cholesterol vs Heart Rate', labels={'chol': 'Chol', 'thalach': 'Max HR'},
                        template='plotly_white', color_discrete_map={0: '#10B981', 1: '#EF4444'})
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: INSIGHTS ====================
with tab5:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        t_chol, p_chol = stats.ttest_ind(df[df['target']==1]['chol'].dropna(),
                                         df[df['target']==0]['chol'].dropna(), equal_var=False)
        st.metric("Cholesterol p-value", f"{p_chol:.5f}", 
                 delta="Significant ✓" if p_chol < 0.05 else "Not significant")
    
    with col2:
        t_bp, p_bp = stats.ttest_ind(df[df['target']==1]['trestbps'].dropna(),
                                     df[df['target']==0]['trestbps'].dropna(), equal_var=False)
        st.metric("Resting BP p-value", f"{p_bp:.5f}",
                 delta="Significant ✓" if p_bp < 0.05 else "Not significant")
    
    st.divider()
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig = px.box(df_filtered, x='target', y='heart_rate_reserve', color='target',
                    title='Heart Rate Reserve Distribution', template='plotly_white',
                    color_discrete_map={0: '#10B981', 1: '#EF4444'})
        fig.update_layout(height=350, showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        corr_cols = ['age','chol','trestbps','thalach','oldpeak']
        corr = df[corr_cols + ['target']].corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='Viridis',
                       title='Feature Correlation Matrix', template='plotly_white')
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("""
<div style='text-align: center; padding: 24px; color: #6B7280;'>
    <p style='font-weight: 600; margin-bottom: 8px;'>HeartCare Analytics Platform</p>
    <p style='font-size: 12px;'>Built by Fedelis | Advanced Medical Intelligence</p>
    <p style='font-size: 11px; margin-top: 12px;'>⚠️ For educational purposes only - Not a substitute for professional medical advice</p>
</div>
""", unsafe_allow_html=True)