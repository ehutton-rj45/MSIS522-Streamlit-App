import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}

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

def run_hw1_pipeline(data_path="unified_rain_data.csv"):
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    features = [
        'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'DayOfWeek', 'IsWeekend', 
        'Rain_Lag_1', 'Rain_Lag_2', 'Rain_Lag_3', 'Rain_Lag_4', 'Rain_Lag_5', 'Rain_Lag_6', 'Rain_Lag_7', 'Rain_Lag_365',
        'Rain_Diff_1', 'Rain_Diff_2', 'Rain_Diff_3',
        'Rain_Rolling_Sum_3', 'Rain_Rolling_Sum_7',
        'Rain_Rolling_3', 'Rain_Rolling_7', 'Rain_Rolling_14', 'Rain_Rolling_30',
        'Rain_Std_3', 'Rain_Std_7', 'Rain_Std_14', 'Rain_Std_30'
    ]
    target = 'Target_Average_Rain_Next_Day'
    
    X = df[features].fillna(0)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    with open("hw1_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    metrics_list = []
    hyperparams = {}
    
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_preds = np.maximum(lr.predict(X_test_scaled), 0)
    metrics_list.append(evaluate_model(y_test, lr_preds, "Linear Regression"))
    with open("hw1_lr.pkl", "wb") as f: pickle.dump(lr, f)
    
    print("Training Decision Tree...")
    dt_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_leaf': [5, 10, 20, 50]}
    dt_cv = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    dt_cv.fit(X_train_scaled, y_train)
    dt_best = dt_cv.best_estimator_
    dt_preds = np.maximum(dt_best.predict(X_test_scaled), 0)
    metrics_list.append(evaluate_model(y_test, dt_preds, "Decision Tree (CART)"))
    hyperparams["Decision Tree"] = dt_cv.best_params_
    with open("hw1_dt.pkl", "wb") as f: pickle.dump(dt_best, f)
    
    print("Training Random Forest...")
    rf_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 8]}
    rf_cv = GridSearchCV(RandomForestRegressor(random_state=42), rf_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_cv.fit(X_train_scaled, y_train)
    rf_best = rf_cv.best_estimator_
    rf_preds = np.maximum(rf_best.predict(X_test_scaled), 0)
    metrics_list.append(evaluate_model(y_test, rf_preds, "Random Forest"))
    hyperparams["Random Forest"] = rf_cv.best_params_
    with open("hw1_rf.pkl", "wb") as f: pickle.dump(rf_best, f)
    
    print("Training LightGBM...")
    lgb_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 4, 5, 6], 'learning_rate': [0.01, 0.05, 0.1]}
    lgb_cv = GridSearchCV(LGBMRegressor(random_state=42), lgb_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    lgb_cv.fit(X_train_scaled, y_train)
    lgb_best = lgb_cv.best_estimator_
    lgb_preds = np.maximum(lgb_best.predict(X_test_scaled), 0)
    metrics_list.append(evaluate_model(y_test, lgb_preds, "LightGBM"))
    hyperparams["LightGBM"] = lgb_cv.best_params_
    with open("hw1_lgb.pkl", "wb") as f: pickle.dump(lgb_best, f)
        
    print("Generating SHAP for LightGBM (Best Model)...")
    explainer = shap.TreeExplainer(lgb_best)
    shap_sample = pd.DataFrame(X_test_scaled, columns=features).sample(min(1000, len(X_test)), random_state=42)
    shap_values = explainer(shap_sample)
    with open("hw1_shap_explainer.pkl", "wb") as f: pickle.dump(explainer, f)
    with open("hw1_shap_values.pkl", "wb") as f: pickle.dump(shap_values, f)
    
    print("Training PyTorch MLP...")
    device = torch.device('cpu')
    mlp = TorchMLP(X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.005)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train.values))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    epochs = 30
    train_losses = []
    
    for epoch in range(epochs):
        mlp.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = mlp(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss (MSE)')
    plt.title("MLP Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("mlp_history.png")
    
    mlp.eval()
    with torch.no_grad():
        mlp_preds = mlp(torch.FloatTensor(X_test_scaled)).numpy()
        mlp_preds = np.maximum(mlp_preds, 0)
        
    metrics_list.append(evaluate_model(y_test, mlp_preds, "PyTorch MLP"))
    torch.save(mlp.state_dict(), "hw1_mlp.pth")
    
    test_df_export = pd.DataFrame(X_test_scaled, columns=features)
    test_df_export['Actual'] = y_test.values
    test_df_export['Pred_LR'] = lr_preds
    test_df_export['Pred_DT'] = dt_preds
    test_df_export['Pred_RF'] = rf_preds
    test_df_export['Pred_LGB'] = lgb_preds
    test_df_export['Pred_MLP'] = mlp_preds
    test_df_export.to_csv("hw1_test_predictions.csv", index=False)
    
    res_df = pd.DataFrame(metrics_list)
    res_df.to_csv("hw1_metrics.csv", index=False)
    
    with open("hw1_hyperparams.pkl", "wb") as f: pickle.dump(hyperparams, f)
    
    print("Done! All Phase 2 homework criteria fulfilled.")

if __name__ == "__main__":
    run_hw1_pipeline()
