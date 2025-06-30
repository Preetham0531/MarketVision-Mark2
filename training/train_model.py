import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, classification_report
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    data_path = os.path.join('..', 'data', 'final_training_data.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")
    
    return df

def prepare_features_and_targets(df):
    feature_columns = [col for col in df.columns if col not in [
        'date', 'symbol', 'next_day_return', 'next_5_day_return', 
        'next_week_return', 'next_month_return', 'price_increase_next_day',
        'price_increase_next_5_days', 'price_increase_next_week', 
        'price_increase_next_month', 'volatility_next_day', 'volatility_next_5_days'
    ]]
    
    regression_targets = [
        'next_day_return', 'next_5_day_return', 'next_week_return', 'next_month_return',
        'volatility_next_day', 'volatility_next_5_days'
    ]
    
    classification_targets = [
        'price_increase_next_day', 'price_increase_next_5_days', 
        'price_increase_next_week', 'price_increase_next_month'
    ]
    
    X = df[feature_columns].fillna(0)
    y_regression = df[regression_targets].fillna(0)
    y_classification = df[classification_targets].fillna(0)
    
    return X, y_regression, y_classification, feature_columns, regression_targets, classification_targets

def train_lightgbm_model(X, y_regression, y_classification, feature_names, regression_labels, classification_labels):
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42
    )
    
    regression_models = {}
    classification_models = {}
    
    for i, target in enumerate(regression_labels):
        print(f"Training regression model for {target}...")
        
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_train, y_reg_train.iloc[:, i],
            eval_set=[(X_test, y_reg_test.iloc[:, i])],
            early_stopping_rounds=50,
            verbose=False
        )
        
        regression_models[target] = model
    
    for i, target in enumerate(classification_labels):
        print(f"Training classification model for {target}...")
        
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_train, y_clf_train.iloc[:, i],
            eval_set=[(X_test, y_clf_test.iloc[:, i])],
            early_stopping_rounds=50,
            verbose=False
        )
        
        classification_models[target] = model
    
    return regression_models, classification_models, X_test, y_reg_test, y_clf_test

def evaluate_models(regression_models, classification_models, X_test, y_reg_test, y_clf_test):
    regression_results = {}
    classification_results = {}
    
    for target, model in regression_models.items():
        y_pred = model.predict(X_test)
        y_true = y_reg_test[target]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        regression_results[target] = {
            'rmse': rmse,
            'mae': mae
        }
        
        print(f"{target} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    for target, model in classification_models.items():
        y_pred = model.predict(X_test)
        y_true = y_clf_test[target]
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        classification_results[target] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"{target} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    return regression_results, classification_results

def save_model_and_metadata(regression_models, classification_models, feature_names, 
                           regression_labels, classification_labels, regression_results, 
                           classification_results):
    
    models_dir = os.path.join('..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_data = {
        'regression_models': regression_models,
        'classification_models': classification_models,
        'feature_names': feature_names,
        'regression_labels': regression_labels,
        'classification_labels': classification_labels
    }
    
    model_path = os.path.join(models_dir, 'multioutput_lightgbm_model.pkl')
    joblib.dump(model_data, model_path)
    
    metadata = {
        'model_type': 'LightGBM Multi-Output',
        'training_date': datetime.now().isoformat(),
        'test_size': 0.2,
        'random_state': 42,
        'feature_names': feature_names,
        'regression_labels': regression_labels,
        'classification_labels': classification_labels,
        'regression_results': regression_results,
        'classification_results': classification_results
    }
    
    metadata_path = os.path.join(models_dir, 'lightgbm_model_info.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

def main():
    print("Starting model training...")
    
    df = load_and_prepare_data()
    if df is None:
        return
    
    X, y_regression, y_classification, feature_names, regression_labels, classification_labels = prepare_features_and_targets(df)
    
    print(f"Features: {len(feature_names)}")
    print(f"Regression targets: {len(regression_labels)}")
    print(f"Classification targets: {len(classification_labels)}")
    
    regression_models, classification_models, X_test, y_reg_test, y_clf_test = train_lightgbm_model(
        X, y_regression, y_classification, feature_names, regression_labels, classification_labels
    )
    
    regression_results, classification_results = evaluate_models(
        regression_models, classification_models, X_test, y_reg_test, y_clf_test
    )
    
    save_model_and_metadata(
        regression_models, classification_models, feature_names,
        regression_labels, classification_labels, regression_results, classification_results
    )
    
    print("Model training completed successfully!")

if __name__ == "__main__":
    main() 