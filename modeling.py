import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import shap

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    percentile_95 = np.percentile(y_true, 95)
    true_spikes = y_true >= percentile_95
    pred_spikes = y_pred >= percentile_95
    spike_recall = np.sum(true_spikes & pred_spikes) / np.sum(true_spikes) if np.sum(true_spikes) > 0 else 0
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name} - MAE: {mae:.4f}, SpikeRecall@95: {spike_recall:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {"MAE": mae, "Spike Recall 95th": spike_recall, "RMSE": rmse, "R2": r2}

class TwoStageRainModel:
    def __init__(self):
        self.clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=10, num_leaves=64, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        self.reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=10, num_leaves=64, objective='tweedie', tweedie_variance_power=1.8, random_state=42, n_jobs=-1, verbose=-1)

    def fit(self, X, y):
        is_rain = (y > 0).astype(int)
        self.clf.fit(X, is_rain)
        
        rain_mask = y > 0
        X_rain = X[rain_mask]
        y_rain = y[rain_mask]
        
        if len(y_rain) > 0:
            self.reg.fit(X_rain, y_rain)
            
        return self

    def predict(self, X):
        p_rain = self.clf.predict_proba(X)[:, 1]
        pred_amount = self.reg.predict(X)
        pred_amount = np.clip(pred_amount, 0, None)
        return p_rain * pred_amount

def get_ensemble_model():
    return TwoStageRainModel()

def run_per_station_modeling(data_path):
    print("\n--- Running Phase 3 Per-Station Modeling ---")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)

    features = [
        'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'DayOfWeek', 'IsWeekend', 
        'Station_Target_Encoded',
        'Rain_Lag_1', 'Rain_Lag_2', 'Rain_Lag_3', 'Rain_Lag_4', 'Rain_Lag_5', 'Rain_Lag_6', 'Rain_Lag_7', 'Rain_Lag_365',
        'Rain_Diff_1', 'Rain_Diff_2', 'Rain_Diff_3',
        'Rain_Rolling_Sum_3', 'Rain_Rolling_Sum_7',
        'Rain_Rolling_3', 'Rain_Rolling_7', 'Rain_Rolling_14', 'Rain_Rolling_30',
        'Rain_Std_3', 'Rain_Std_7', 'Rain_Std_14', 'Rain_Std_30'
    ]
    target = 'Target_Rain_Next_Day'

    train_df = df[df['Year'] < 2024]
    test_df = df[df['Year'] >= 2024]

    X_train = train_df[features].fillna(0)
    y_train = train_df[target]
    X_test = test_df[features].fillna(0)
    y_test = test_df[target]

    print(f"Per-Station Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("Training Two-Stage Rain Model (Classifier + Regressor)...")
    ensemble = get_ensemble_model()
    ensemble.fit(X_train, y_train)

    preds = ensemble.predict(X_test)
    preds = np.maximum(preds, 0)
    
    metrics = evaluate_model(y_test, preds, "Two-Stage Model (Per Station)")
    pd.DataFrame({"Two-Stage Model": metrics}).T.to_csv('model_metrics.csv')

    print("Generating SHAP values (using Regressor component)...")
    lgb_fitted = ensemble.reg
    explainer = shap.TreeExplainer(lgb_fitted)
    X_test_sample = X_test.sample(1000, random_state=42)
    shap_values = explainer(X_test_sample)

    with open('best_model.pkl', 'wb') as f: pickle.dump(ensemble, f)
    with open('shap_explainer.pkl', 'wb') as f: pickle.dump(explainer, f)
    with open('shap_values.pkl', 'wb') as f: pickle.dump(shap_values, f)
        
    X_test_sample['Actual_Rain'] = y_test.loc[X_test_sample.index]
    context_df = test_df.loc[X_test_sample.index, ['Station', 'Date']]
    X_test_sample = pd.concat([context_df, X_test_sample], axis=1)
    X_test_sample.to_csv('test_sample_for_app.csv', index=False)
    
    return metrics


def run_unified_modeling(data_path):
    print("\n--- Running Phase 3 Unified Regional Modeling ---")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)

    features = [
        'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'DayOfWeek', 'IsWeekend', 
        'Rain_Lag_1', 'Rain_Lag_2', 'Rain_Lag_3', 'Rain_Lag_4', 'Rain_Lag_5', 'Rain_Lag_6', 'Rain_Lag_7', 'Rain_Lag_365',
        'Rain_Diff_1', 'Rain_Diff_2', 'Rain_Diff_3',
        'Rain_Rolling_Sum_3', 'Rain_Rolling_Sum_7',
        'Rain_Rolling_3', 'Rain_Rolling_7', 'Rain_Rolling_14', 'Rain_Rolling_30',
        'Rain_Std_3', 'Rain_Std_7', 'Rain_Std_14', 'Rain_Std_30'
    ]
    target = 'Target_Average_Rain_Next_Day'

    train_df = df[df['Year'] < 2024]
    test_df = df[df['Year'] >= 2024]

    X_train = train_df[features].fillna(0)
    y_train = train_df[target]
    X_test = test_df[features].fillna(0)
    y_test = test_df[target]

    print(f"Unified Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("Training Unified Two-Stage Rain Model...")
    ensemble = get_ensemble_model()
    ensemble.fit(X_train, y_train)

    preds = ensemble.predict(X_test)
    preds = np.maximum(preds, 0)
    
    metrics = evaluate_model(y_test, preds, "Unified Two-Stage Model")
    pd.DataFrame({"Unified Two-Stage": metrics}).T.to_csv('model_metrics_unified.csv')

    print("Generating SHAP values (using Regressor component) for unified model...")
    lgb_fitted = ensemble.reg
    explainer = shap.TreeExplainer(lgb_fitted)
    shap_values = explainer(X_test)

    with open('best_model_unified.pkl', 'wb') as f: pickle.dump(ensemble, f)
    with open('shap_explainer_unified.pkl', 'wb') as f: pickle.dump(explainer, f)
    with open('shap_values_unified.pkl', 'wb') as f: pickle.dump(shap_values, f)
        
    X_test_with_actual = X_test.copy()
    X_test_with_actual['Actual_Average_Rain'] = y_test
    context_df = test_df[['Date']]
    X_test_with_actual = pd.concat([context_df, X_test_with_actual], axis=1)
    X_test_with_actual.to_csv('test_sample_for_app_unified.csv', index=False)
    
    return metrics


if __name__ == "__main__":
    per_station_data = r"c:\Users\gtmmp\Desktop\isohyetal contours\processed_rain_data.csv"
    unified_data = r"c:\Users\gtmmp\Desktop\isohyetal contours\unified_rain_data.csv"
    
    run_per_station_modeling(per_station_data)
    run_unified_modeling(unified_data)
    
    print("\nAll Phase 3 modeling workflows complete.")
