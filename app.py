import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

# 1. Set Streamlit Page Config to 'wide' as requested
st.set_page_config(
    page_title="Customer Churn Analytics Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphism & High-End Typography
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Apply font across app */
        html, body, [class*="css"], .stText {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Gradient Header Banner */
        .header-container {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2.5rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .header-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .header-subtitle {
            font-size: 1.1rem;
            font-weight: 300;
            opacity: 0.9;
            margin-top: 0.5rem;
        }
        
        /* KPI Metric Cards styling */
        .kpi-card {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(209, 213, 219, 0.3);
            box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px 0 rgba(31, 38, 135, 0.1);
        }
        .kpi-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #6b7280;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 800;
            color: #1e3c72;
        }
        .kpi-delta {
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }
        .delta-up { color: #dc2626; }
        .delta-down { color: #16a34a; }
        
        /* Section styling */
        .section-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1f2937;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #1e3c72;
            padding-left: 0.75rem;
        }
    </style>
""", unsafe_allow_html=True)

# 2. Loading and Preprocessing the dataset with caching
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("customer_churn_data.csv")
    
    # Preprocess: Handle missing values in 'TotalCharges'
    if "TotalCharges" in df.columns:
        # Convert to string and strip whitespace
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
        # Convert empty strings to NaN
        df["TotalCharges"] = df["TotalCharges"].replace("", pd.NA)
        # Convert to numeric, setting invalid parsing as NaN
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Compute median of TotalCharges
        median_val = df["TotalCharges"].median()
        # Fill missing values with median
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
        
    return df

# 3. Loading the trained ML model, categorical encoders, and scaler with resource caching
@st.cache_resource
def load_ml_artifacts():
    model_data = joblib.load("mlp_churn_model.pkl")
    scaler_obj = joblib.load("churn_scaler.pkl")
    return model_data["model"], model_data["encoders"], scaler_obj

try:
    df = load_and_preprocess_data()
    data_loaded_successfully = True
except Exception as e:
    data_loaded_successfully = False
    error_message = str(e)

try:
    model, encoders, scaler = load_ml_artifacts()
    ml_loaded_successfully = True
except Exception as e:
    ml_loaded_successfully = False
    ml_error_message = str(e)

# Sidebar with Logo/Branding & Interactive Filters
with st.sidebar:
    st.markdown("<div style='text-align: center; padding: 1rem 0;'><h2 style='color:#1e3c72; font-weight:800; margin:0;'>TeleSphere</h2><p style='color:#6b7280; font-size:0.9rem;'>Churn Intelligence Platform</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔍 Interactive Filters")
    st.write("Refine the metrics and segments analyzed across the dashboard:")
    
    if data_loaded_successfully:
        # Multiselect filter for InternetService
        internet_options = sorted(list(df["InternetService"].unique()))
        selected_internet = st.multiselect(
            "Internet Service Provider",
            options=internet_options,
            default=internet_options,
            help="Select one or more internet services to filter the results."
        )
        
        # Multiselect filter for ContractType
        contract_options = sorted(list(df["ContractType"].unique()))
        selected_contract = st.multiselect(
            "Subscription Contract Type",
            options=contract_options,
            default=contract_options,
            help="Select one or more contract types to filter the results."
        )
        
        st.markdown("---")
    
    st.info("💡 **Tip:** Streamlit is running in *wide mode* for maximum analytical real estate.")

# Main Dashboard layout
# Hero Banner
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📊 Customer Churn Intelligence Portal</h1>
        <p class="header-subtitle">Real-time analytical interface for customer retention, contract tenures, and monthly charges.</p>
    </div>
""", unsafe_allow_html=True)

if data_loaded_successfully:
    # Apply filters dynamically
    filtered_df = df[
        df["InternetService"].isin(selected_internet) & 
        df["ContractType"].isin(selected_contract)
    ]
    
    if filtered_df.empty:
        st.warning("⚠️ No subscriber records match your current filter selections. Please select at least one filter in the sidebar.")
    else:
        # Calculate Metrics based on filtered data
        total_customers = len(filtered_df)
        churn_count = len(filtered_df[filtered_df["Churn"] == "Yes"])
        churn_rate = (churn_count / total_customers) * 100 if total_customers > 0 else 0.0
        avg_monthly = filtered_df["MonthlyCharges"].mean() if total_customers > 0 else 0.0
        
        # 3. Key Metrics at the top using st.metric()
        st.markdown('<div class="section-header">📈 Core Platform Metrics</div>', unsafe_allow_html=True)
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(label="👥 Total Customers", value=f"{total_customers:,}")
        with metric_col2:
            st.metric(label="📉 Overall Churn Rate", value=f"{churn_rate:.2f}%")
        with metric_col3:
            st.metric(label="💰 Average Monthly Charges", value=f"${avg_monthly:.2f}")

        # 4. Exploratory Data Analysis Section
        st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
        st.write("Explore dataset distributions, customer attributes, and factors influencing customer attrition below:")
        
        # Sub-section for Dataset Preview
        st.markdown('### 📄 Customer Dataset Preview (First 5 Rows)')
        st.dataframe(filtered_df.head(5), use_container_width=True)
        st.success("✅ Dataset successfully loaded, preprocessed, and cached!")
        
        # Sub-section for Visual Analysis
        st.markdown('### 📊 Visual Distributions & Insights')
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Pie chart showing the proportion of churned vs. retained customers
            churn_counts = filtered_df["Churn"].value_counts().reset_index()
            churn_counts.columns = ["Churn Status", "Count"]
            
            # Human-readable labels
            churn_counts["Churn Status"] = churn_counts["Churn Status"].map({"Yes": "Churned", "No": "Retained"})
            
            fig_pie = px.pie(
                churn_counts,
                values="Count",
                names="Churn Status",
                title="Proportion of Churned vs. Retained Customers",
                color="Churn Status",
                color_discrete_map={"Churned": "#e63946", "Retained": "#1d3557"},
                hole=0.4,
                template="plotly_white"
            )
            fig_pie.update_traces(
                textinfo="percent+label", 
                pull=[0.05, 0],
                textfont=dict(size=12, family="Outfit")
            )
            fig_pie.update_layout(
                font_family="Outfit",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                margin=dict(t=50, b=50, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with chart_col2:
            # Histogram showing distribution of customer tenure, color-coded by churn status
            fig_hist = px.histogram(
                filtered_df,
                x="Tenure (months)",
                color="Churn",
                title="Customer Tenure Distribution (Months) by Churn Status",
                color_discrete_map={"Yes": "#e63946", "No": "#1d3557"},
                barmode="overlay",
                nbins=35,
                template="plotly_white"
            )
            # Tweak aesthetics
            fig_hist.update_traces(opacity=0.75)
            fig_hist.update_layout(
                font_family="Outfit",
                hovermode="x unified",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                margin=dict(t=50, b=50, l=20, r=20)
            )
            # Remove gridlines
            fig_hist.update_xaxes(showgrid=False)
            fig_hist.update_yaxes(showgrid=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        # Full-width analysis of churn rate based on 'ContractType'
        st.markdown('### 📄 Contract Type Churn Rate Analysis')
        
        # Calculate actual churn percentage per Contract Type
        contract_stats = filtered_df.groupby("ContractType").agg(
            Total=("CustomerID", "count"),
            Churned=("Churn", lambda x: (x == "Yes").sum())
        ).reset_index()
        contract_stats["Churn Rate (%)"] = (contract_stats["Churned"] / contract_stats["Total"]) * 100
        
        fig_contract = px.bar(
            contract_stats,
            x="ContractType",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            title="Churn Rate by Contract Type (%)",
            color="ContractType",
            color_discrete_sequence=["#1d3557", "#457b9d", "#a8dadc"],
            template="plotly_white"
        )
        
        fig_contract.update_traces(
            texttemplate='%{text:.1f}%', 
            textposition='outside',
            textfont=dict(size=12, family="Outfit", color="#1f2937"),
            marker_line_color='rgba(0,0,0,0)',
            marker_line_width=0,
            opacity=0.9
        )
        
        fig_contract.update_layout(
            font_family="Outfit",
            showlegend=False,
            margin=dict(t=50, b=30, l=20, r=20),
            yaxis=dict(
                ticksuffix="%",
                range=[0, max(contract_stats["Churn Rate (%)"]) * 1.15]
            )
        )
        
        # Remove unnecessary gridlines
        fig_contract.update_xaxes(showgrid=False)
        fig_contract.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig_contract, use_container_width=True)

        # 5. Predictive High-Risk Customer Segments Section using ML
        st.markdown('<div class="section-header">🚨 Predictive High-Risk Customer Segments</div>', unsafe_allow_html=True)
        
        if ml_loaded_successfully:
            # Preprocess the currently filtered dataframe on the fly
            pred_df = filtered_df.copy()
            
            categorical_cols = ["Gender", "Partner", "Dependents", "PhoneService", "InternetService", "ContractType"]
            for col in categorical_cols:
                le = encoders[col]
                # Ensure no unseen labels are parsed
                pred_df[col] = pred_df[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
                pred_df[col] = le.transform(pred_df[col])
                
            numerical_cols = ["Tenure (months)", "MonthlyCharges", "TotalCharges"]
            pred_df[numerical_cols] = scaler.transform(pred_df[numerical_cols])
            
            # Select feature vector exactly in same order as trained
            X_pred = pred_df[categorical_cols + numerical_cols]
            
            # Compute predictive probabilities
            churn_probs = model.predict_proba(X_pred)[:, 1] * 100
            
            # Add Churn Probability (%) to the display dataframe
            filtered_df["Churn Probability (%)"] = churn_probs.round(2)
            
            # Extract high-risk profiles: churn probability >= 50%
            high_risk_df = filtered_df[filtered_df["Churn Probability (%)"] >= 50.0]
            # Sort the table so the customers with the highest churn probability appear at the top
            high_risk_df = high_risk_df.sort_values(by="Churn Probability (%)", ascending=False)
            
            col_risk_text, col_risk_data = st.columns([2, 3])
            
            with col_risk_text:
                st.markdown(f"""
                    ### 🔍 ML Churn Prediction Analysis
                    We have upgraded the risk assessment model to use a **Multi-Layer Perceptron (MLP) Neural Network** classifier. 
                    
                    **Predictive Classification Model:**
                    * **Algorithm:** Multi-Layer Perceptron (Artificial Neural Network)
                    * **High-Risk Threshold:** $\ge 50.0\%$ predicted probability of churn.
                    * **Scale & Encoders:** Preprocessed dynamically using pre-trained `StandardScaler` and `LabelEncoder` parameters.
                    
                    ### ⚠️ Why is this model superior?
                    1. **Non-linear Relationship Modeling:** The neural network looks at the joint combinations of factors (such as short tenure, high charges, and lack of contract commitments) rather than one single factor.
                    2. **Exact Probability Scores:** Rather than a simple flag, each customer receives an exact churn probability. This allows retention teams to prioritize their outreach.
                    3. **Optimized Resource Allocation:** Focus incentives on customers with 80%+ risk scores first, maximizing the return on retention spending.
                    
                    *💡 **Action Item:** Target this list of **{len(high_risk_df)} highly vulnerable subscribers** sorted by risk severity!*
                """)
                
            with col_risk_data:
                st.markdown("### 📋 Predictive High-Risk Customer Profiles")
                if not high_risk_df.empty:
                    # Select key columns to display including Churn Probability (%)
                    st.dataframe(
                        high_risk_df[[
                            "CustomerID", "Churn Probability (%)", "ContractType", "InternetService", "MonthlyCharges", "Tenure (months)", "Churn"
                        ]],
                        use_container_width=True,
                        height=350
                    )
                    st.info(f"Showing {len(high_risk_df)} high-risk profiles sorted by churn probability.")
                else:
                    st.success("🎉 No customers are predicted to have a churn probability >= 50% in the currently selected filters!")
        else:
            st.error(f"❌ Could not run machine learning model. Error: {ml_error_message}")

else:
    st.error(f"❌ Error loading customer churn dataset: {error_message}")
