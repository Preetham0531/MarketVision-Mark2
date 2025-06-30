import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
import lightgbm as lgb
import xgboost as xgb
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

FINAL_DATA_PATH = os.path.join('..', 'data', 'final_training_data.csv')
MODELS_DIR = os.path.join('..', 'models')
LOGS_DIR = os.path.join('..', 'logs')
TUNING_DIR = os.path.join(LOGS_DIR, 'hyperparameter_tuning')
os.makedirs(TUNING_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 3  
N_ITER = 20   

def load_and_prepare_data():
    print("[INFO] Loading and preparing data for hyperparameter tuning...")
    
    df = pd.read_csv(FINAL_DATA_PATH)
    print(f"[INFO] Dataset shape: {df.shape}")
    
    regression_labels = [
        'predicted_close_price_t_plus_1',
        'predicted_close_price_t_plus_5', 
        'predicted_close_price_t_plus_20',
        'predicted_daily_return_t_plus_1',
        'predicted_weekly_return_t_plus_5',
        'predicted_volatility_next_20_days'
    ]
    
    classification_labels = [
        'predicted_trend_direction',
        'buy_sell_hold_recommendation',
        'predicted_volatility_regime'
    ]
    
    feature_cols = [col for col in df.columns if col not in regression_labels + classification_labels + ['date']]
    X = df[feature_cols].copy()
    X = X.fillna(X.mean())
    
    y_reg = df[regression_labels].copy()
    y_reg = y_reg.fillna(y_reg.mean())
    
    y_clf = df[classification_labels].copy()
    
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    y_clf_encoded = pd.DataFrame()
    for col in classification_labels:
        le = LabelEncoder()
        y_clf_encoded[col] = le.fit_transform(y_clf[col].fillna('neutral'))
        label_encoders[col] = le
    
    return X, y_reg, y_clf_encoded, feature_cols, label_encoders

def define_parameter_grids():
    print("[INFO] Defining parameter grids...")
    
    lgb_reg_params = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'estimator__max_depth': [3, 5, 7, 9],
        'estimator__num_leaves': [31, 50, 100],
        'estimator__min_child_samples': [20, 50, 100],
        'estimator__subsample': [0.8, 0.9, 1.0],
        'estimator__colsample_bytree': [0.8, 0.9, 1.0],
        'estimator__reg_alpha': [0, 0.1, 0.5],
        'estimator__reg_lambda': [0, 0.1, 0.5]
    }
    
    lgb_clf_params = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'estimator__max_depth': [3, 5, 7, 9],
        'estimator__num_leaves': [31, 50, 100],
        'estimator__min_child_samples': [20, 50, 100],
        'estimator__subsample': [0.8, 0.9, 1.0],
        'estimator__colsample_bytree': [0.8, 0.9, 1.0],
        'estimator__reg_alpha': [0, 0.1, 0.5],
        'estimator__reg_lambda': [0, 0.1, 0.5]
    }
    
    xgb_reg_params = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'estimator__max_depth': [3, 5, 7, 9],
        'estimator__min_child_weight': [1, 3, 5],
        'estimator__subsample': [0.8, 0.9, 1.0],
        'estimator__colsample_bytree': [0.8, 0.9, 1.0],
        'estimator__reg_alpha': [0, 0.1, 0.5],
        'estimator__reg_lambda': [0, 0.1, 0.5]
    }
    
    xgb_clf_params = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'estimator__max_depth': [3, 5, 7, 9],
        'estimator__min_child_weight': [1, 3, 5],
        'estimator__subsample': [0.8, 0.9, 1.0],
        'estimator__colsample_bytree': [0.8, 0.9, 1.0],
        'estimator__reg_alpha': [0, 0.1, 0.5],
        'estimator__reg_lambda': [0, 0.1, 0.5]
    }
    
    return {
        'lightgbm': {'regression': lgb_reg_params, 'classification': lgb_clf_params},
        'xgboost': {'regression': xgb_reg_params, 'classification': xgb_clf_params}
    }

def create_models():
    print("[INFO] Creating base models...")
    
    lgb_reg = MultiOutputRegressor(
        lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)
    )
    
    lgb_clf = MultiOutputClassifier(
        lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    )
    
    xgb_reg = MultiOutputRegressor(
        xgb.XGBRegressor(random_state=RANDOM_STATE)
    )
    
    xgb_clf = MultiOutputClassifier(
        xgb.XGBClassifier(random_state=RANDOM_STATE)
    )
    
    return {
        'lightgbm': {'regression': lgb_reg, 'classification': lgb_clf},
        'xgboost': {'regression': xgb_reg, 'classification': xgb_clf}
    }

def tune_regression_model(model, param_grid, X, y, model_name):
    print(f"[INFO] Tuning {model_name} regression model...")
    
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    
    rmse_scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
    
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=N_ITER,
        cv=tscv,
        scoring=rmse_scorer,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    search.fit(X, y)
    
    print(f"[INFO] Best {model_name} regression parameters: {search.best_params_}")
    print(f"[INFO] Best {model_name} regression score: {-search.best_score_:.4f}")
    
    return search

def tune_classification_model(model, param_grid, X, y, model_name):
    print(f"[INFO] Tuning {model_name} classification model...")
    
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    
    accuracy_scorer = make_scorer(accuracy_score)
    
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=N_ITER,  
        cv=tscv,
        scoring=accuracy_scorer,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    

    search.fit(X, y)
    
    print(f"[INFO] Best {model_name} classification parameters: {search.best_params_}")
    print(f"[INFO] Best {model_name} classification score: {search.best_score_:.4f}")
    
    return search

def save_tuning_results(results, model_name):
    print(f"[INFO] Saving {model_name} tuning results...")
    
    results_path = os.path.join(TUNING_DIR, f'{model_name}_tuning_results.json')
    
    serializable_results = {}
    for task, search in results.items():
        serializable_results[task] = {
            'best_params': search.best_params_,
            'best_score': float(search.best_score_),
            'cv_results': {
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist(),
                'params': search.cv_results_['params']
            }
        }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"[INFO] {model_name} tuning results saved to {results_path}")

def save_optimized_models(results, model_name, label_encoders): 
    print(f"[INFO] Saving optimized {model_name} models...")
    
    reg_model_path = os.path.join(MODELS_DIR, f'optimized_{model_name}_regression_model.pkl')
    joblib.dump(results['regression'].best_estimator_, reg_model_path)
    
    clf_model_path = os.path.join(MODELS_DIR, f'optimized_{model_name}_classification_model.pkl')
    joblib.dump(results['classification'].best_estimator_, clf_model_path)
    
    for col, le in label_encoders.items():
        encoder_path = os.path.join(MODELS_DIR, f'label_encoder_{col}.pkl')
        joblib.dump(le, encoder_path)
    
    metadata = {
        'model_name': model_name,
        'tuning_date': datetime.now().isoformat(),
        'regression_best_params': results['regression'].best_params_,
        'regression_best_score': float(results['regression'].best_score_),
        'classification_best_params': results['classification'].best_params_,
        'classification_best_score': float(results['classification'].best_score_),
        'cv_folds': CV_FOLDS,
        'n_iter': N_ITER
    }
    
    metadata_path = os.path.join(MODELS_DIR, f'optimized_{model_name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[INFO] Optimized {model_name} models saved to {MODELS_DIR}")

def compare_models(tuning_results):
    print("[INFO] Creating model comparison summary...")
    
    comparison_data = []
    for model_name, results in tuning_results.items():
        comparison_data.append({
            'model': model_name,
            'regression_score': -results['regression'].best_score_,  
            'classification_score': results['classification'].best_score_,
            'regression_params': results['regression'].best_params_,
            'classification_params': results['classification'].best_params_
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    comparison_path = os.path.join(TUNING_DIR, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print("\nRegression Performance (RMSE - lower is better):")
    for _, row in comparison_df.iterrows():
        print(f"{row['model']}: {row['regression_score']:.4f}")
    
    print("\nClassification Performance (Accuracy - higher is better):")
    for _, row in comparison_df.iterrows():
        print(f"{row['model']}: {row['classification_score']:.4f}")
    
    best_regression = comparison_df.loc[comparison_df['regression_score'].idxmin(), 'model']
    best_classification = comparison_df.loc[comparison_df['classification_score'].idxmax(), 'model']
    
    print(f"\nBest Regression Model: {best_regression}")
    print(f"Best Classification Model: {best_classification}")
    
    return comparison_df

def main():     
    print("=" * 60)
    print("HYPERPARAMETER TUNING PIPELINE")
    print("=" * 60)
    
    try:
        X, y_reg, y_clf, feature_cols, label_encoders = load_and_prepare_data()
        
        param_grids = define_parameter_grids()
        
        models = create_models()
        
        tuning_results = {}
        
        for model_name in ['lightgbm', 'xgboost']:
            print(f"\n{'='*20} TUNING {model_name.upper()} {'='*20}")
            
            reg_search = tune_regression_model(
                models[model_name]['regression'],
                param_grids[model_name]['regression'],
                X, y_reg, model_name
            )
            
            clf_search = tune_classification_model(
                models[model_name]['classification'],
                param_grids[model_name]['classification'],
                X, y_clf, model_name
            )
            
            tuning_results[model_name] = {
                'regression': reg_search,
                'classification': clf_search
            }
            
            save_tuning_results(tuning_results[model_name], model_name)
            save_optimized_models(tuning_results[model_name], model_name, label_encoders)
        
        comparison_df = compare_models(tuning_results)
        
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {TUNING_DIR}")
        print(f"Optimized models saved to: {MODELS_DIR}")
        
    except Exception as e:
        print(f"[ERROR] Hyperparameter tuning failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 