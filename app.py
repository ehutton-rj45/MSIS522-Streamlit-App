import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import altair as alt
from isohyet_map import plot_isohyet
import torch
import torch.nn as nn
from modeling import TwoStageRainModel

class TorchMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Data Science Workflow", layout="wide")

st.sidebar.title("Model Selection")
st.sidebar.info("Select a primary model to dynamically update the plots in the Performance and Explainability tabs:")
selected_model_name = st.sidebar.radio("Primary Model:", ["LightGBM", "Random Forest", "Two-Stage LGBM + Tweedie"])

model_key_map = {
    "LightGBM": "LGB",
    "Random Forest": "RF",
    "Two-Stage LGBM + Tweedie": "TS"
}
selected_key = model_key_map[selected_model_name]

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "unified_rain_data.csv"))
    df['Date'] = pd.to_datetime(df['Date'])
    
    metrics = pd.read_csv(os.path.join(BASE_DIR, "hw1_metrics.csv"))
    test_preds = pd.read_csv(os.path.join(BASE_DIR, "hw1_test_predictions.csv"))
    
    with open(os.path.join(BASE_DIR, "hw1_hyperparams.pkl"), "rb") as f:
        hyperparams = pickle.load(f)
        
    ts_metrics = pd.read_csv(os.path.join(BASE_DIR, "model_metrics_unified.csv")).rename(columns={'Unnamed: 0': 'Model'})
    ts_test_sample = pd.read_csv(os.path.join(BASE_DIR, "test_sample_for_app_unified.csv"))
        
    return df, metrics, test_preds, hyperparams, ts_metrics, ts_test_sample

@st.cache_resource
def load_models(_test_preds):
    with open(os.path.join(BASE_DIR, 'hw1_lr.pkl'), 'rb') as f:
        lr = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'hw1_dt.pkl'), 'rb') as f:
        dt = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'hw1_rf.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'hw1_lgb.pkl'), 'rb') as f:
        lgb = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'hw1_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
        
    mlp = TorchMLP(input_dim=len(scaler.mean_))
    mlp.load_state_dict(torch.load(os.path.join(BASE_DIR, "hw1_mlp.pth")))
    mlp.eval()
    
    with open(os.path.join(BASE_DIR, 'hw1_shap_explainer.pkl'), 'rb') as f:
        lgb_explainer = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'hw1_shap_values.pkl'), 'rb') as f:
        lgb_shap = pickle.load(f)
        
    features_list = scaler.get_feature_names_out()
    X_test_scaled = _test_preds[features_list].sample(min(1000, len(_test_preds)), random_state=42)
    rf_explainer = shap.TreeExplainer(rf)
    rf_shap = rf_explainer(X_test_scaled)
    
    with open(os.path.join(BASE_DIR, 'best_model_unified.pkl'), 'rb') as f:
        ts_model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'shap_explainer_unified.pkl'), 'rb') as f:
        ts_explainer = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'shap_values_unified.pkl'), 'rb') as f:
        ts_shap = pickle.load(f)
        
    explainers = {"LGB": lgb_explainer, "RF": rf_explainer, "TS": ts_explainer}
    shap_vals = {"LGB": lgb_shap, "RF": rf_shap, "TS": ts_shap}

    return {"LR": lr, "DT": dt, "RF": rf, "LGB": lgb, "MLP": mlp, "TS": ts_model}, scaler, explainers, shap_vals

try:
    df, metrics, test_preds, hyperparams, ts_metrics, ts_test_sample = load_data()
    models, scaler, explainers, shap_vals = load_models(test_preds)
except Exception as e:
    st.warning(f"Models and data are still initializing. Please wait. Error: {e}")
    st.stop()

st.title("Data Science Workflow - Kitsap Rainfall")
st.markdown("Developed as part of the MSIS 522 Data Science Workflow Homework.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Tab 1 — Executive Summary", 
    "Tab 2 — Descriptive Analytics", 
    "Tab 3 — Model Performance", 
    "Tab 4 — Explainability & Interactive Prediction",
    "Tab 5 — Isohyetal Map (Alternative Dataset feature)"
])

with tab1:
    st.header("Executive Summary")
    st.subheader("Dataset Description")
    st.write("We are analyzing a region-wide unified rainfall dataset for Kitsap County. It originates from daily rainfall gauge reads across multiple stations, combined to find the average rainfall volume across the region daily.")
    st.write("**Prediction Target:** `Target_Average_Rain_Next_Day`. This continuous target measures the average rainfall tomorrow across all tracking locations.")
    
    st.subheader("Why This problem matters?")
    st.write("Forecasting rainfall using simple lags and historical time series isn't just about weather—it supports emergency flood risk mitigation and resource allocation. By accurately forecasting massive rainfall spikes, authorities can issue warnings, dispatch sandbags, and allocate emergency management personnel efficiently within the county.")
    
    st.subheader("Approach and Key Findings")
    st.write("We approached this regression problem by implementing a five-model ML pipeline. The models include Linear Regression (Baseline), Decision Tree, Random Forest, LightGBM, and an Artificial Neural Network (MLP implemented in PyTorch). All hyperparameters were tuned using grid-search to avoid overfitting.")
    st.write("Key finding: Random Forest and LightGBM models handled the extremely skewed nature of rainfall data significantly better than simple linear combinations, achieving solid R-squared metrics over baseline.")

with tab2:
    st.header("Descriptive Analytics")
    
    st.write("### Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['Target_Average_Rain_Next_Day'], bins=50, kde=True, ax=ax, color='blue')
    ax.set_title("Distribution of Daily Rainfall (Continuous)")
    st.pyplot(fig)
    st.write("**Commentary:** The dataset is highly zero-inflated and positively skewed. The vast majority of days see zero or near-zero rainfall, while a few severe storm events account for substantial outliers. We did not transform it for modeling, keeping interpretations raw scale.")
    
    st.write("### Feature Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='Rain_Lag_1', y='Target_Average_Rain_Next_Day', alpha=0.3, ax=ax)
        ax.set_title("Target vs. Rain Lag 1")
        st.pyplot(fig)
        st.write("**Commentary:** We see a moderate positive scatter, indicating consecutive rainy days.")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='Month_sin', y='Target_Average_Rain_Next_Day', ax=ax)
        ax.set_title("Rainfall by Monthly Sine Transformation")
        st.pyplot(fig)
        st.write("**Commentary:** Rainfall is highly dependent on cyclical season metrics, confirming that winter months see heavier storms.")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Rain_Rolling_Sum_7'], bins=30, ax=ax, color='green')
        ax.set_title("Distribution of 7-Day Rolling Rainfall")
        st.pyplot(fig)
        st.write("**Commentary:** A deeply heavy-tailed distribution matching the target, serving as an effective long-term momentum indicator.")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='Rain_Std_7', y='Target_Average_Rain_Next_Day', alpha=0.3, ax=ax, color='purple')
        ax.set_title("Target vs. 7-Day Rain Standard Deviation")
        st.pyplot(fig)
        st.write("**Commentary:** Higher localized volatility strongly signals an active weather system, correlating well with larger rainfall events.")
        
    st.write("### Correlation Heatmap")
    cols = ['Target_Average_Rain_Next_Day', 'Rain_Lag_1', 'Rain_Lag_7', 'Rain_Rolling_Sum_3', 'Rain_Std_7', 'Month_sin', 'Month_cos']
    corr_matrix = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
    ax.set_title("Features vs. Target Correlation Heatmap")
    st.pyplot(fig)
    st.write("**Commentary:** The highest correlations to tomorrow's rain are `Rain_Lag_1` and `Rain_Rolling_Sum_3`. These provide an inertial snapshot of the moving weather system.")

with tab3:
    st.header("Model Performance")
    
    ts_row = pd.DataFrame([{"Model": "Two-Stage LGBM + Tweedie", "MAE": ts_metrics.iloc[0]['MAE'], "RMSE": ts_metrics.iloc[0]['RMSE'], "R2": ts_metrics.iloc[0]['R2']}])
    all_metrics = pd.concat([metrics, ts_row], ignore_index=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Comparison Table")
        st.dataframe(all_metrics.style.highlight_max(subset=['R2'], color='lightgreen').highlight_min(subset=['MAE'], color='lightgreen'))
    
    with col2:
        st.subheader("Model R-Squared Comparison")
        fig, ax = plt.subplots()
        sns.barplot(data=all_metrics, x='Model', y='R2', palette='viridis', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    st.write("### Predicted vs. Actual Output")
    st.write(f"Currently displaying: **{selected_model_name}**")
    
    if selected_key == "TS":
        actuals = ts_test_sample['Actual_Average_Rain']
        ts_features = [c for c in ts_test_sample.columns if c not in ['Date', 'Actual_Average_Rain']]
        preds = models['TS'].predict(ts_test_sample[ts_features].fillna(0))
    else:
        actuals = test_preds['Actual']
        preds = test_preds[f'Pred_{selected_key}']
        
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=actuals, y=preds, alpha=0.3, ax=ax)
    max_val = max(actuals.max(), preds.max())
    ax.plot([0, max_val], [0, max_val], 'r--')
    ax.set_title(f"Predicted vs. Actual ({selected_model_name})")
    ax.set_xlabel("Actual Rain")
    ax.set_ylabel("Predicted Rain")
    st.pyplot(fig)
    
    with st.expander("Best Hyperparameters via CV"):
        st.json(hyperparams)

    st.write("### MLP Training History")
    try:
        st.image("mlp_history.png", caption="Training Loss (Epochs) for PyTorch MLP")
    except:
        pass

with tab4:
    st.header("Explainability & Interactive Prediction")
    
    st.subheader("Global Explainability (SHAP)")
    cur_shap = shap_vals[selected_key]
    if selected_key == "TS":
        feat_names = [c for c in ts_test_sample.columns if c not in ['Date', 'Actual_Average_Rain']]
    else:
        feat_names = scaler.get_feature_names_out()
        
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Summary Beeswarm Plot**")
        fig, ax = plt.subplots()
        shap.summary_plot(cur_shap, feature_names=feat_names, show=False)
        st.pyplot(fig)
        
    with col2:
        st.write("**Summary Bar Plot**")
        fig, ax = plt.subplots()
        shap.summary_plot(cur_shap, feature_names=feat_names, plot_type='bar', show=False)
        st.pyplot(fig)
        
    st.write("**Interpretation:** Features having the strongest impact rely heavily on short-term lags (`Rain_Lag_1`, `Rain_Rolling_Sum_3`). They positively drive tomorrow's forecast up. This helps a decision-maker quickly assess momentum when a storm is already active.")
    
    st.markdown("---")
    st.subheader("Interactive Prediction Simulator")
    
    inf_col1, inf_col2 = st.columns(2)
    with inf_col1:
        st.write(f"**Current Model:** {selected_model_name}")
        lag1 = st.slider("Rain Lag 1 (inches)", min_value=0.0, max_value=5.0, value=1.0)
        roll3 = st.slider("Rain Rolling Sum 3 (inches)", min_value=0.0, max_value=8.0, value=2.0)
        
    with inf_col2:
        features = [
            'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'DayOfWeek', 'IsWeekend', 
            'Rain_Lag_1', 'Rain_Lag_2', 'Rain_Lag_3', 'Rain_Lag_4', 'Rain_Lag_5', 'Rain_Lag_6', 'Rain_Lag_7', 'Rain_Lag_365',
            'Rain_Diff_1', 'Rain_Diff_2', 'Rain_Diff_3',
            'Rain_Rolling_Sum_3', 'Rain_Rolling_Sum_7',
            'Rain_Rolling_3', 'Rain_Rolling_7', 'Rain_Rolling_14', 'Rain_Rolling_30',
            'Rain_Std_3', 'Rain_Std_7', 'Rain_Std_14', 'Rain_Std_30'
        ]
        
        mock_input = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        mock_input['Rain_Lag_1'] = lag1
        mock_input['Rain_Rolling_Sum_3'] = roll3
        
        if selected_key == 'TS':
            pred = models['TS'].predict(mock_input)[0]
        else:
            scaled_in = scaler.transform(mock_input)
            pred = models[selected_key].predict(scaled_in)[0]
            
        pred = max(pred, 0)
        
        st.metric(label=f"Predicted Rain Tomorrow", value=f"{pred:.4f} inches")
        
        st.write("SHAP Waterfall for this input:")
        if selected_key == 'TS':
            sv_single = explainers['TS'](mock_input)
            sv_single.feature_names = features
        else:
            scaled_in = scaler.transform(mock_input)
            sv_single = explainers[selected_key](scaled_in)
            sv_single.feature_names = features
            
        fig, ax = plt.subplots()
        shap.plots.waterfall(sv_single[0], show=False)
        st.pyplot(fig)

with tab5:
    st.header("Isohyetal Contour Map")
    st.write("Bonus Dataset feature to explore raw historical spatial contours.")
    
    @st.cache_data
    def load_stat_data():
        df_hist = pd.read_csv(os.path.join(BASE_DIR, "processed_rain_data.csv"))
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        try:
            df_future = pd.read_csv(os.path.join(BASE_DIR, "future_forecast_stations.csv"))
            df_future['Date'] = pd.to_datetime(df_future['Date'])
            full_df = pd.concat([df_hist, df_future], ignore_index=True)
        except:
            full_df = df_hist
        return full_df
        
    try:
        stat_df = load_stat_data()
        
        available_years = sorted(stat_df['Date'].dt.year.unique())
        default_idx = available_years.index(2026) if 2026 in available_years else 0
        selected_year = st.selectbox("Select Year", available_years, index=default_idx)
        
        year_df = stat_df[stat_df['Date'].dt.year == selected_year]
        daily_rain = year_df.groupby(year_df['Date'].dt.date)['Rain (inches)'].sum().to_dict()
        year_dates = sorted(list(daily_rain.keys()))
        
        if len(year_dates) > 0:
            st.write(f"**Total Daily Rainfall for {selected_year}**")
            rain_series = pd.Series(daily_rain)
            rain_series.index = pd.to_datetime(rain_series.index)
            
            def format_date_slider(dt):
                r = daily_rain.get(dt, 0)
                return f"{dt.strftime('%b %d')} ({r:.1f}\")"
                
            selected_date = st.select_slider(
                "Select Day (Total Rain):", 
                options=year_dates,
                format_func=format_date_slider
            )

            source = pd.DataFrame({
                'Date': rain_series.index,
                'Rain': rain_series.values
            })

            base_bar = alt.Chart(source).mark_bar(color='steelblue').encode(
                x=alt.X('Date:T', title='', scale=alt.Scale(domain=[pd.Timestamp(year_dates[0]), pd.Timestamp(year_dates[-1])])),
                y=alt.Y('Rain:Q', title='Total Rain (inches)')
            ).properties(height=250)
            
            selected_rule = alt.Chart(pd.DataFrame({'Date': [pd.to_datetime(selected_date)]})).mark_rule(
                color='orange', strokeWidth=2, opacity=1.0
            ).encode(
                x='Date:T'
            )

            chart = base_bar + selected_rule
            
            if selected_year == 2026:
                barrier_rule = alt.Chart(pd.DataFrame({'Date': [pd.to_datetime("2026-03-10 12:00:00")]})).mark_rule(
                    color='red', strokeDash=[5, 5], strokeWidth=2
                ).encode(
                    x='Date:T'
                )
                barrier_text = alt.Chart(pd.DataFrame({
                    'Date': [pd.to_datetime("2026-03-10 12:00:00")], 
                    'Rain': [source['Rain'].max() * 0.95]
                })).mark_text(
                    align='left', dx=5, color='red', text='Collected vs Predicted Data', fontWeight='bold'
                ).encode(
                    x='Date:T',
                    y='Rain:Q'
                )
                chart = chart + barrier_rule + barrier_text
                
            st.altair_chart(chart, use_container_width=True)
            
            st.write(f"**Total rainfall across reporting stations on {selected_date}:** {daily_rain.get(selected_date, 0):.2f} inches")
            
            if st.button("Generate Regional Map"):
                fig, msg = plot_isohyet(str(selected_date), df=stat_df)
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.error(msg)
        else:
            st.warning("No data available for this year.")
    except Exception as e:
        st.write(f"Error loading historical/predictive map data: {e}")
