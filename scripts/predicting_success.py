"""
Predicting Student Academic Success or Dropout
===============================================
This script applies Logistic Regression classification to predict whether a student
will graduate or dropout from their studies.

Research Question: "¿Es posible predecir el abandono académico o éxito de un estudiante 
utilizando técnicas estadísticas y de clasificación?"

Author: Statistical Analysis Project
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "prepared_data", "dataset_prepared.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "prediction_results")
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_data(filepath):
    """
    Load the prepared dataset and filter for binary classification.
    We exclude 'Enrolled' students and keep only 'Dropout' vs 'Graduate'.
    """
    print("=" * 70)
    print("LOADING AND PREPARING DATA FOR CLASSIFICATION")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    print(f"\n📊 Dataset original shape: {df.shape}")
    
    # Show distribution of target variable
    print("\n📈 Distribución original de la variable Target:")
    target_counts = df['Target'].value_counts()
    for target, count in target_counts.items():
        percentage = count / len(df) * 100
        print(f"   - {target}: {count} ({percentage:.1f}%)")
    
    # Filter only Dropout and Graduate (exclude Enrolled students)
    # This is because we want to predict final outcomes
    df_filtered = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
    
    print(f"\n📊 Dataset filtrado (solo Dropout y Graduate): {df_filtered.shape}")
    print("\n📈 Distribución del dataset filtrado:")
    target_counts_filtered = df_filtered['Target'].value_counts()
    for target, count in target_counts_filtered.items():
        percentage = count / len(df_filtered) * 100
        print(f"   - {target}: {count} ({percentage:.1f}%)")
    
    # Create binary target: 1 = Graduate (Success), 0 = Dropout
    df_filtered['Target_binary'] = (df_filtered['Target'] == 'Graduate').astype(int)
    
    return df_filtered

def select_features(df):
    """
    Select relevant features for the classification model.
    Exclude derived columns (z-scores, outliers) and use original features.
    """
    # Original feature columns (excluding target and derived columns)
    exclude_patterns = ['_zscore', '_outlier', 'Target', 'Target_encoded', 'Target_binary']
    
    feature_columns = [col for col in df.columns 
                       if not any(pattern in col for pattern in exclude_patterns)]
    
    print(f"\n🔧 Features seleccionadas para el modelo: {len(feature_columns)}")
    
    return feature_columns

def prepare_train_test_split(df, feature_columns):
    """
    Prepare training and testing sets with stratification.
    """
    X = df[feature_columns]
    y = df['Target_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n📊 División de datos:")
    print(f"   - Conjunto de entrenamiento: {len(X_train)} ({100-TEST_SIZE*100:.0f}%)")
    print(f"   - Conjunto de prueba: {len(X_test)} ({TEST_SIZE*100:.0f}%)")
    
    # Verify stratification
    print(f"\n📊 Distribución en entrenamiento:")
    print(f"   - Graduados: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"   - Deserción: {len(y_train)-sum(y_train)} ({(len(y_train)-sum(y_train))/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Standardize features for logistic regression.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def train_logistic_regression(X_train, y_train):
    """
    Train logistic regression model with regularization.
    """
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO DEL MODELO DE REGRESIÓN LOGÍSTICA")
    print("=" * 70)
    
    # Train model with L2 regularization
    model = LogisticRegression(
        penalty='l2',
        C=1.0,  # Inverse of regularization strength
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    print("\n✅ Modelo entrenado exitosamente")
    print(f"   - Regularización: L2 (Ridge)")
    print(f"   - Solver: LBFGS")
    print(f"   - Class weight: Balanced")
    
    return model

def perform_cross_validation(model, X_train, y_train):
    """
    Perform k-fold cross-validation to assess model stability.
    """
    print("\n" + "=" * 70)
    print("VALIDACIÓN CRUZADA (5-FOLD)")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision')
    cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    print("\n📊 Resultados de Validación Cruzada (5 folds):")
    print(f"\n   {'Métrica':<15} {'Media':>10} {'Desv. Est.':>12} {'Min':>10} {'Max':>10}")
    print("   " + "-" * 57)
    print(f"   {'Accuracy':<15} {cv_accuracy.mean():>10.4f} {cv_accuracy.std():>12.4f} {cv_accuracy.min():>10.4f} {cv_accuracy.max():>10.4f}")
    print(f"   {'Precision':<15} {cv_precision.mean():>10.4f} {cv_precision.std():>12.4f} {cv_precision.min():>10.4f} {cv_precision.max():>10.4f}")
    print(f"   {'Recall':<15} {cv_recall.mean():>10.4f} {cv_recall.std():>12.4f} {cv_recall.min():>10.4f} {cv_recall.max():>10.4f}")
    print(f"   {'F1-Score':<15} {cv_f1.mean():>10.4f} {cv_f1.std():>12.4f} {cv_f1.min():>10.4f} {cv_f1.max():>10.4f}")
    print(f"   {'ROC-AUC':<15} {cv_roc_auc.mean():>10.4f} {cv_roc_auc.std():>12.4f} {cv_roc_auc.min():>10.4f} {cv_roc_auc.max():>10.4f}")
    
    cv_results = {
        'accuracy': {'mean': cv_accuracy.mean(), 'std': cv_accuracy.std(), 'values': cv_accuracy.tolist()},
        'precision': {'mean': cv_precision.mean(), 'std': cv_precision.std(), 'values': cv_precision.tolist()},
        'recall': {'mean': cv_recall.mean(), 'std': cv_recall.std(), 'values': cv_recall.tolist()},
        'f1': {'mean': cv_f1.mean(), 'std': cv_f1.std(), 'values': cv_f1.tolist()},
        'roc_auc': {'mean': cv_roc_auc.mean(), 'std': cv_roc_auc.std(), 'values': cv_roc_auc.tolist()}
    }
    
    return cv_results

def evaluate_model(model, X_test, y_test, feature_names):
    """
    Comprehensive model evaluation on test set.
    """
    print("\n" + "=" * 70)
    print("EVALUACIÓN DEL MODELO EN CONJUNTO DE PRUEBA")
    print("=" * 70)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    print("\n📊 Métricas de Rendimiento:")
    print(f"\n   {'Métrica':<25} {'Valor':>10}")
    print("   " + "-" * 35)
    print(f"   {'Accuracy (Exactitud)':<25} {accuracy:>10.4f}")
    print(f"   {'Precision':<25} {precision:>10.4f}")
    print(f"   {'Recall (Sensibilidad)':<25} {recall:>10.4f}")
    print(f"   {'F1-Score':<25} {f1:>10.4f}")
    print(f"   {'ROC-AUC':<25} {roc_auc:>10.4f}")
    print(f"   {'Average Precision':<25} {avg_precision:>10.4f}")
    
    # Classification Report
    print("\n📊 Reporte de Clasificación Detallado:")
    print("\n" + classification_report(y_test, y_pred, 
                                       target_names=['Dropout (0)', 'Graduate (1)']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Matriz de Confusión:")
    print(f"\n                    Predicho")
    print(f"                 Dropout  Graduate")
    print(f"   Real Dropout    {cm[0][0]:>5}    {cm[0][1]:>5}")
    print(f"   Real Graduate   {cm[1][0]:>5}    {cm[1][1]:>5}")
    
    # Interpretation
    tn, fp, fn, tp = cm.ravel()
    print(f"\n   ✅ Verdaderos Negativos (Dropout correctamente predicho): {tn}")
    print(f"   ❌ Falsos Positivos (Dropout predicho como Graduate): {fp}")
    print(f"   ❌ Falsos Negativos (Graduate predicho como Dropout): {fn}")
    print(f"   ✅ Verdaderos Positivos (Graduate correctamente predicho): {tp}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    return metrics, y_pred, y_pred_proba

def analyze_feature_importance(model, feature_names):
    """
    Analyze and rank feature importance based on coefficients.
    """
    print("\n" + "=" * 70)
    print("ANÁLISIS DE IMPORTANCIA DE VARIABLES")
    print("=" * 70)
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients),
        'Odds_Ratio': np.exp(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\n📊 Top 15 Variables más Importantes (por magnitud del coeficiente):")
    print("\n   " + "-" * 75)
    print(f"   {'#':<3} {'Variable':<45} {'Coef.':>10} {'Odds Ratio':>12}")
    print("   " + "-" * 75)
    
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
        effect = "↑ Graduate" if row['Coefficient'] > 0 else "↓ Dropout"
        print(f"   {i+1:<3} {row['Feature']:<45} {row['Coefficient']:>10.4f} {row['Odds_Ratio']:>12.4f}")
    
    print("\n📖 Interpretación de Odds Ratio:")
    print("   - Odds Ratio > 1: Mayor probabilidad de GRADUARSE")
    print("   - Odds Ratio < 1: Mayor probabilidad de ABANDONAR")
    print("   - Odds Ratio = 1: Variable no tiene efecto")
    
    # Key findings
    print("\n🔍 Hallazgos Clave:")
    
    # Top positive factors (increase graduation probability)
    positive_factors = feature_importance[feature_importance['Coefficient'] > 0].head(5)
    print("\n   📈 Factores que AUMENTAN la probabilidad de graduarse:")
    for _, row in positive_factors.iterrows():
        print(f"      • {row['Feature']}: OR = {row['Odds_Ratio']:.3f}")
    
    # Top negative factors (increase dropout probability)
    negative_factors = feature_importance[feature_importance['Coefficient'] < 0].head(5)
    print("\n   📉 Factores que AUMENTAN la probabilidad de abandono:")
    for _, row in negative_factors.iterrows():
        print(f"      • {row['Feature']}: OR = {row['Odds_Ratio']:.3f}")
    
    return feature_importance

def create_visualizations(model, X_test, y_test, y_pred, y_pred_proba, 
                          feature_importance, metrics, output_dir):
    """
    Create comprehensive visualizations for the classification results.
    """
    print("\n" + "=" * 70)
    print("GENERACIÓN DE VISUALIZACIONES")
    print("=" * 70)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Confusion Matrix Heatmap
    ax1 = fig.add_subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Dropout', 'Graduate'],
                yticklabels=['Dropout', 'Graduate'],
                ax=ax1)
    ax1.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicción')
    ax1.set_ylabel('Valor Real')
    
    # 2. ROC Curve
    ax2 = fig.add_subplot(2, 3, 2)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Clasificador aleatorio')
    ax2.fill_between(fpr, tpr, alpha=0.3)
    ax2.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    ax2.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    ax2.set_title('Curva ROC', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = fig.add_subplot(2, 3, 3)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    ax3.plot(recall_curve, precision_curve, 'g-', linewidth=2, 
             label=f'AP = {metrics["average_precision"]:.3f}')
    ax3.axhline(y=sum(y_test)/len(y_test), color='r', linestyle='--', 
                label=f'Baseline = {sum(y_test)/len(y_test):.3f}')
    ax3.fill_between(recall_curve, precision_curve, alpha=0.3, color='green')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature Importance (Top 15)
    ax4 = fig.add_subplot(2, 3, 4)
    top_features = feature_importance.head(15)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features['Coefficient']]
    bars = ax4.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['Feature'], fontsize=9)
    ax4.axvline(x=0, color='black', linewidth=0.5)
    ax4.set_xlabel('Coeficiente de Regresión Logística')
    ax4.set_title('Top 15 Variables más Importantes', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Favorece Graduación'),
                       Patch(facecolor='#e74c3c', label='Favorece Abandono')]
    ax4.legend(handles=legend_elements, loc='lower right')
    
    # 5. Probability Distribution
    ax5 = fig.add_subplot(2, 3, 5)
    prob_dropout = y_pred_proba[y_test == 0]
    prob_graduate = y_pred_proba[y_test == 1]
    ax5.hist(prob_dropout, bins=30, alpha=0.6, label='Dropout', color='#e74c3c', density=True)
    ax5.hist(prob_graduate, bins=30, alpha=0.6, label='Graduate', color='#2ecc71', density=True)
    ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Umbral (0.5)')
    ax5.set_xlabel('Probabilidad Predicha de Graduarse')
    ax5.set_ylabel('Densidad')
    ax5.set_title('Distribución de Probabilidades por Clase Real', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Metrics Summary
    ax6 = fig.add_subplot(2, 3, 6)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                      metrics['f1_score'], metrics['roc_auc']]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics_names)))
    bars = ax6.bar(metrics_names, metrics_values, color=colors)
    ax6.set_ylim(0, 1)
    ax6.set_ylabel('Valor')
    ax6.set_title('Resumen de Métricas de Clasificación', fontsize=14, fontweight='bold')
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    ax6.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Umbral 0.8')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, 'classification_results.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualizaciones guardadas en: {fig_path}")
    
    # Create additional visualization: Odds Ratio plot
    fig2, ax = plt.subplots(figsize=(12, 10))
    top_20 = feature_importance.head(20).copy()
    top_20 = top_20.sort_values('Odds_Ratio', ascending=True)
    
    colors = ['#2ecc71' if x > 1 else '#e74c3c' for x in top_20['Odds_Ratio']]
    bars = ax.barh(range(len(top_20)), top_20['Odds_Ratio'], color=colors)
    ax.axvline(x=1, color='black', linewidth=2, label='Sin efecto (OR=1)')
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['Feature'], fontsize=10)
    ax.set_xlabel('Odds Ratio', fontsize=12)
    ax.set_title('Odds Ratio de las Top 20 Variables\n(Efecto sobre la Probabilidad de Graduarse)', 
                 fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [Patch(facecolor='#2ecc71', label='OR > 1: Favorece Graduación'),
                       Patch(facecolor='#e74c3c', label='OR < 1: Favorece Abandono')]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    odds_path = os.path.join(output_dir, 'odds_ratio_analysis.png')
    plt.savefig(odds_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráfico de Odds Ratio guardado en: {odds_path}")
    
    return [fig_path, odds_path]

def save_results(metrics, cv_results, feature_importance, output_dir):
    """
    Save all results to JSON file.
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'Logistic Regression',
        'parameters': {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'lbfgs',
            'class_weight': 'balanced',
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE
        },
        'cross_validation': cv_results,
        'test_metrics': metrics,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    results_path = os.path.join(output_dir, 'classification_report.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save feature importance to CSV
    fi_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance.to_csv(fi_path, index=False)
    
    print(f"\n✅ Resultados guardados en: {results_path}")
    print(f"✅ Importancia de variables guardada en: {fi_path}")
    
    return results_path, fi_path

def generate_conclusions(metrics, cv_results, feature_importance):
    """
    Generate final conclusions and answer the research question.
    """
    print("\n" + "=" * 70)
    print("CONCLUSIONES Y RESPUESTA A LA PREGUNTA DE INVESTIGACIÓN")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│  PREGUNTA DE INVESTIGACIÓN                                               │
│  "¿Es posible predecir el abandono académico o éxito de un estudiante   │
│   utilizando técnicas estadísticas y de clasificación?"                  │
└─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Determine answer based on metrics
    accuracy = metrics['accuracy']
    roc_auc = metrics['roc_auc']
    f1 = metrics['f1_score']
    
    if roc_auc >= 0.85:
        performance_level = "EXCELENTE"
        answer = "SÍ, definitivamente"
    elif roc_auc >= 0.75:
        performance_level = "BUENO"
        answer = "SÍ, con buena precisión"
    elif roc_auc >= 0.65:
        performance_level = "MODERADO"
        answer = "SÍ, con precisión moderada"
    else:
        performance_level = "LIMITADO"
        answer = "Parcialmente, con limitaciones"
    
    print(f"""
📊 RESPUESTA: {answer}

El modelo de Regresión Logística demuestra un rendimiento {performance_level} 
en la predicción del abandono académico y éxito estudiantil:

   📈 Métricas de Rendimiento:
   ┌────────────────────────────────────────────────────────────────────┐
   │ • Accuracy (Exactitud):        {accuracy:.1%}                           │
   │ • ROC-AUC:                     {roc_auc:.1%}                           │
   │ • F1-Score:                    {f1:.1%}                           │
   │ • Cross-Validation Accuracy:   {cv_results['accuracy']['mean']:.1%} (±{cv_results['accuracy']['std']:.1%})     │
   └────────────────────────────────────────────────────────────────────┘
    """)
    
    # Key predictive factors
    top_positive = feature_importance[feature_importance['Coefficient'] > 0].head(3)
    top_negative = feature_importance[feature_importance['Coefficient'] < 0].head(3)
    
    print("""
🔍 FACTORES PREDICTIVOS CLAVE:

   ✅ Factores que PREDICEN ÉXITO (Graduación):""")
    for _, row in top_positive.iterrows():
        print(f"      • {row['Feature']}")
    
    print("""
   ❌ Factores que PREDICEN ABANDONO:""")
    for _, row in top_negative.iterrows():
        print(f"      • {row['Feature']}")
    
    print("""
💡 IMPLICACIONES PRÁCTICAS:

   1. DETECCIÓN TEMPRANA: El modelo puede identificar estudiantes en riesgo
      de abandono al inicio de su carrera académica.
   
   2. INTERVENCIÓN FOCALIZADA: Las variables identificadas pueden guiar
      programas de apoyo estudiantil hacia factores modificables.
   
   3. ASIGNACIÓN DE RECURSOS: Permite priorizar recursos de apoyo hacia
      estudiantes con mayor probabilidad de abandono.

📋 LIMITACIONES Y CONSIDERACIONES:

   • El modelo se basa en datos históricos y patrones pasados
   • Factores externos no medidos pueden influir en el resultado
   • Las predicciones deben usarse como herramienta de apoyo, no como
     determinantes absolutos del futuro de un estudiante
   • Se recomienda combinar con evaluación cualitativa

🎯 RECOMENDACIÓN FINAL:

   El análisis confirma que ES POSIBLE predecir con """ + performance_level.lower() + """ precisión
   el abandono académico utilizando técnicas de regresión logística. Este modelo
   puede ser una herramienta valiosa para instituciones educativas en la
   identificación temprana de estudiantes que requieren apoyo adicional.
    """)

def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 70)
    print("  PREDICCIÓN DE ÉXITO/ABANDONO ACADÉMICO ESTUDIANTIL")
    print("  Modelo: Regresión Logística")
    print("=" * 70)
    print(f"\n📅 Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and prepare data
    df = load_and_prepare_data(INPUT_PATH)
    
    # Select features
    feature_columns = select_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = prepare_train_test_split(df, feature_columns)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model
    model = train_logistic_regression(X_train_scaled, y_train)
    
    # Cross-validation
    cv_results = perform_cross_validation(model, X_train_scaled, y_train)
    
    # Evaluate on test set
    metrics, y_pred, y_pred_proba = evaluate_model(
        model, X_test_scaled, y_test, feature_columns
    )
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, feature_columns)
    
    # Create visualizations
    viz_paths = create_visualizations(
        model, X_test_scaled, y_test, y_pred, y_pred_proba,
        feature_importance, metrics, OUTPUT_DIR
    )
    
    # Save results
    save_results(metrics, cv_results, feature_importance, OUTPUT_DIR)
    
    # Generate conclusions
    generate_conclusions(metrics, cv_results, feature_importance)
    
    print("\n" + "=" * 70)
    print("  ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"\n📁 Resultados guardados en: {OUTPUT_DIR}/")
    print("   • classification_results.png - Visualizaciones principales")
    print("   • odds_ratio_analysis.png - Análisis de Odds Ratio")
    print("   • classification_report.json - Métricas detalladas")
    print("   • feature_importance.csv - Importancia de variables")
    print("\n")

if __name__ == "__main__":
    main()
