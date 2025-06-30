import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

FINAL_DATA_PATH = os.path.join('..', 'data', 'final_training_data.csv')
MODELS_DIR = os.path.join('..', 'models')
LOGS_DIR = os.path.join('..', 'logs')
EVALUATION_DIR = os.path.join(LOGS_DIR, 'evaluation')
os.makedirs(EVALUATION_DIR, exist_ok=True)

CV_FOLDS = 5
RANDOM_STATE = 42

def load_data_and_model():
    print("[INFO] Loading dataset and model...")
    
    df = pd.read_csv(FINAL_DATA_PATH)
    print(f"[INFO] Dataset shape: {df.shape}")
    
    model_path = os.path.join(MODELS_DIR, 'multioutput_lightgbm_model.pkl')
    model_data = joblib.load(model_path)
    
    metadata_path = os.path.join(MODELS_DIR, 'lightgbm_model_info.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return df, model_data, metadata

def prepare_evaluation_data(df, metadata):
    print("[INFO] Preparing evaluation data...")
    
    feature_cols = metadata['feature_names']
    regression_labels = metadata['regression_labels']
    classification_labels = metadata['classification_labels']
    
    X = df[feature_cols].copy()
    X = X.fillna(X.mean())
    
    y_reg = df[regression_labels].copy()
    y_reg = y_reg.fillna(y_reg.mean())
    
    y_clf = df[classification_labels].copy()
    
    label_encoders = {}
    y_clf_encoded = pd.DataFrame()
    for col in classification_labels:
        le = joblib.load(os.path.join(MODELS_DIR, f'label_encoder_{col}.pkl'))
        y_clf_encoded[col] = le.transform(y_clf[col].fillna('neutral'))
        label_encoders[col] = le
    
    return X, y_reg, y_clf_encoded, label_encoders

def perform_cross_validation(model_data, X, y_reg, y_clf, cv_folds=5):
    print(f"[INFO] Performing {cv_folds}-fold time series cross-validation...")
    
    reg_model = model_data['regression_model']
    clf_model = model_data['classification_model']
    
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    reg_scores = {}
    for i, label in enumerate(y_reg.columns):
        scores = cross_val_score(reg_model, X, y_reg.iloc[:, i], 
                               cv=tscv, scoring='neg_mean_squared_error')
        reg_scores[label] = {
            'rmse_scores': np.sqrt(-scores),
            'mean_rmse': np.sqrt(-scores).mean(),
            'std_rmse': np.sqrt(-scores).std()
        }
        print(f"[{label}] Mean RMSE: {reg_scores[label]['mean_rmse']:.4f} ± {reg_scores[label]['std_rmse']:.4f}")
    
    clf_scores = {}
    for i, label in enumerate(y_clf.columns):
        scores = cross_val_score(clf_model, X, y_clf.iloc[:, i], 
                               cv=tscv, scoring='accuracy')
        clf_scores[label] = {
            'accuracy_scores': scores,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }
        print(f"[{label}] Mean Accuracy: {clf_scores[label]['mean_accuracy']:.4f} ± {clf_scores[label]['std_accuracy']:.4f}")
    
    return reg_scores, clf_scores

def detailed_regression_analysis(model_data, X, y_reg, test_size=0.2):
    print("[INFO] Performing detailed regression analysis...")
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=test_size, random_state=RANDOM_STATE, shuffle=False
    )
    
    reg_model = model_data['regression_model']
    reg_model.fit(X_train, y_train)
    
    y_pred = reg_model.predict(X_test)
    
    results = {}
    for i, label in enumerate(y_reg.columns):
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        
        results[label] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'actual': y_test.iloc[:, i].values,
            'predicted': y_pred[:, i]
        }
        
        print(f"[{label}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return results

def detailed_classification_analysis(model_data, X, y_clf, label_encoders, test_size=0.2):
    print("[INFO] Performing detailed classification analysis...")
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=test_size, random_state=RANDOM_STATE, shuffle=False
    )
    
    clf_model = model_data['classification_model']
    clf_model.fit(X_train, y_train)
    
    y_pred = clf_model.predict(X_test)
    
    results = {}
    for i, label in enumerate(y_clf.columns):
        accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
        f1 = f1_score(y_test.iloc[:, i], y_pred[:, i], average='weighted')
        precision = precision_score(y_test.iloc[:, i], y_pred[:, i], average='weighted', zero_division=0)
        recall = recall_score(y_test.iloc[:, i], y_pred[:, i], average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_test.iloc[:, i], y_pred[:, i])
        
        le = label_encoders[label]
        class_names = le.classes_
        
        results[label] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'class_names': class_names,
            'actual': y_test.iloc[:, i].values,
            'predicted': y_pred[:, i]
        }
        
        print(f"[{label}] Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    return results

def create_regression_plots(reg_results):
    print("[INFO] Creating regression plots...")
    
    n_targets = len(reg_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (label, results) in enumerate(reg_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        ax.scatter(results['actual'], results['predicted'], alpha=0.6)
        
        min_val = min(results['actual'].min(), results['predicted'].min())
        max_val = max(results['actual'].max(), results['predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{label}\nR² = {results["r2"]:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, 'regression_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Regression plots saved to {plot_path}")

def create_classification_plots(clf_results):
    print("[INFO] Creating classification plots...")
    
    n_targets = len(clf_results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (label, results) in enumerate(clf_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        cm = results['confusion_matrix']
        class_names = results['class_names']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{label}\nAccuracy: {results["accuracy"]:.3f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, 'classification_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Classification plots saved to {plot_path}")

def create_performance_summary(reg_results, clf_results, reg_cv_scores, clf_cv_scores):
    print("[INFO] Creating performance summary...")
    
    reg_metrics = []
    for label, results in reg_results.items():
        reg_metrics.append({
            'target': label,
            'rmse': results['rmse'],
            'mae': results['mae'],
            'r2': results['r2'],
            'cv_rmse': reg_cv_scores[label]['mean_rmse']
        })
    
    reg_df = pd.DataFrame(reg_metrics)
    
    clf_metrics = []
    for label, results in clf_results.items():
        clf_metrics.append({
            'target': label,
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'precision': results['precision'],
            'recall': results['recall'],
            'cv_accuracy': clf_cv_scores[label]['mean_accuracy']
        })
    
    clf_df = pd.DataFrame(clf_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    reg_df.plot(x='target', y=['rmse', 'cv_rmse'], kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Regression RMSE Comparison')
    axes[0,0].set_ylabel('RMSE')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    reg_df.plot(x='target', y='r2', kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Regression R² Scores')
    axes[0,1].set_ylabel('R²')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    clf_df.plot(x='target', y=['accuracy', 'cv_accuracy'], kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Classification Accuracy Comparison')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    clf_df.plot(x='target', y=['f1_score', 'precision', 'recall'], kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Classification Metrics')
    axes[1,1].set_ylabel('Score')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, 'performance_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Performance summary saved to {plot_path}")
    
    return reg_df, clf_df

def save_evaluation_report(reg_results, clf_results, reg_cv_scores, clf_cv_scores, reg_df, clf_df):
    print("[INFO] Saving evaluation report...")
    
    report_path = os.path.join(EVALUATION_DIR, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().isoformat()}\n")
        f.write(f"Cross-Validation Folds: {CV_FOLDS}\n\n")

        f.write("REGRESSION MODEL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        for label, results in reg_results.items():
            f.write(f"\n{label}:\n")
            f.write(f"  RMSE: {results['rmse']:.4f}\n")
            f.write(f"  MAE: {results['mae']:.4f}\n")
            f.write(f"  R²: {results['r2']:.4f}\n")
            f.write(f"  CV RMSE: {reg_cv_scores[label]['mean_rmse']:.4f} ± {reg_cv_scores[label]['std_rmse']:.4f}\n")
        
        f.write("\n\nCLASSIFICATION MODEL PERFORMANCE\n")
        f.write("-" * 35 + "\n")
        for label, results in clf_results.items():
            f.write(f"\n{label}:\n")
            f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"  F1-Score: {results['f1_score']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall: {results['recall']:.4f}\n")
            f.write(f"  CV Accuracy: {clf_cv_scores[label]['mean_accuracy']:.4f} ± {clf_cv_scores[label]['std_accuracy']:.4f}\n")
        
        f.write("\n\nPERFORMANCE SUMMARY TABLES\n")
        f.write("-" * 25 + "\n")
        f.write("\nRegression Metrics:\n")
        f.write(reg_df.to_string(index=False))
        f.write("\n\nClassification Metrics:\n")
        f.write(clf_df.to_string(index=False))
    
    print(f"[INFO] Evaluation report saved to {report_path}")

def main():
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    try:
        df, model_data, metadata = load_data_and_model()
        
        X, y_reg, y_clf, label_encoders = prepare_evaluation_data(df, metadata)
        
        reg_cv_scores, clf_cv_scores = perform_cross_validation(model_data, X, y_reg, y_clf, CV_FOLDS)
        
        reg_results = detailed_regression_analysis(model_data, X, y_reg)
        clf_results = detailed_classification_analysis(model_data, X, y_clf, label_encoders)
        
        create_regression_plots(reg_results)
        create_classification_plots(clf_results)
        reg_df, clf_df = create_performance_summary(reg_results, clf_results, reg_cv_scores, clf_cv_scores)
            
        save_evaluation_report(reg_results, clf_results, reg_cv_scores, clf_cv_scores, reg_df, clf_df)
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {EVALUATION_DIR}")
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 