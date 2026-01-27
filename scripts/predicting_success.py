"""
Este script aplica un modelo de Regresión Logística para predecir si un 
estudiante se graduará o abandonará sus estudios.

Pregunta de Investigación: 
"¿Es posible predecir el abandono académico o éxito de un estudiante 
utilizando técnicas estadísticas y de clasificación?"

Modelo utilizado: Regresión Logística
- Ideal para clasificación binaria (Dropout vs Graduate)
"""
import pandas as pd                                      # Manipulación de datos tabulares
import numpy as np                                       # Operaciones numéricas y arrays
import matplotlib.pyplot as plt                          # Creación de gráficos
import seaborn as sns                                    # Visualizaciones estadísticas
import warnings                                          # Manejo de advertencias
import json                                              # Guardado/carga de resultados
import os                                                # Operaciones del sistema
from datetime import datetime                            # Timestamps
from scipy.stats import zscore                           # Cálculo de z-scores
import statsmodels.api as sm                             # Modelos estadísticos
from statsmodels.formula.api import logit                # Regresión logística con fórmulas
from statsmodels.tools.tools import add_constant         # Agregar constante al modelo
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Cálculo de VIF
from statsmodels.tools.sm_exceptions import ConvergenceWarning              # Manejo de warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  # División y CV
from sklearn.linear_model import LogisticRegression                                     # Modelo de clasificación
from sklearn.preprocessing import StandardScaler                                        # Estandarización de features
from sklearn.feature_selection import SelectKBest, f_classif, RFE   # Métodos de selección

# Métricas de evaluación del modelo:
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

# Suprimir advertencias para salida más limpia
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# CONFIGURACIÓN VISUAL
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)  # Tamaño por defecto de figuras
plt.rcParams['font.size'] = 12            # Tamaño de fuente legible

# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Rutas de entrada y salida
INPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "prepared_data", "dataset_prepared.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "prediction_results")

# Parámetros del modelo - importantes para reproducibilidad
RANDOM_STATE = 42   # Semilla para reproducibilidad de resultados
TEST_SIZE = 0.20    # 20% de datos para prueba, 80% para entrenamiento

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_data(filepath):
    """
    Cargar el dataset preparado y filtrar para clasificación binaria.
    Filtramos solo 'Dropout' y 'Graduate'
    
    La variable objetivo se codifica como:
    - 1 = Graduate (Graduado - éxito)
    - 0 = Dropout (Abandono)
    
    El dataset resultante cumple los dos supuestos fundamentales para la regresión logística:
    1. Supuesto de variable dependiente binaria: la variable objetivo tiene solo dos categorías (0 = Dropout, 1 = Graduate).
    2. Supuesto de independencia de las observaciones: se eliminan filas duplicadas, asegurando que cada fila representa una observación única e independiente.
    """
    print("=" * 70)
    print("CARGA Y PREPARACIÓN DE DATOS (VARIABLE OBJ. BINARIA E INDEPENDENCIA)")
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
    df_filtered = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
    
    print(f"\n📊 Dataset filtrado (solo Dropout y Graduate): {df_filtered.shape}")
    print("\n📈 Distribución del dataset filtrado:")
    target_counts_filtered = df_filtered['Target'].value_counts()
    for target, count in target_counts_filtered.items():
        percentage = count / len(df_filtered) * 100
        print(f"   - {target}: {count} ({percentage:.1f}%)")
    
    # Create binary target: 1 = Graduate (Success), 0 = Dropout
    df_filtered['Target_binary'] = (df_filtered['Target'] == 'Graduate').astype(int)

    print("Supuesto de variable dependiente binaria (se cumple): la variable objetivo tiene solo dos categorías (0 = Dropout, 1 = Graduate).")

    # Comprobar independencia de las observaciones (filas duplicadas)
    duplicated_rows = df_filtered.duplicated()
    num_duplicates = duplicated_rows.sum()
    if num_duplicates > 0:
        print(f"Advertencia: Se encontraron {num_duplicates} filas no independientes (duplicadas). Serán eliminadas.")
        df_filtered = df_filtered[~duplicated_rows].copy()
        print(f"Supuesto de independencia de las observaciones (se cumple): se eliminaron {num_duplicates} filas duplicadas. Luego de las transformaciones se logró la independencia de las observaciones.\nCada fila del dataset es única.")
    else:
        print("Supuesto de independencia de las observaciones (se cumple): se eliminan filas duplicadas, asegurando que cada fila representa una observación única e independiente.")
    return df_filtered

def select_features(df):
    """
    Seleccionar características relevantes para el modelo de clasificación.
    
    Excluimos:
    - Columnas derivadas (z-scores, marcadores de outliers) que podrían
      causar fuga de datos o redundancia
    - Variable objetivo y sus variantes
    
    Mantenemos solo las características originales del dataset para
    que el modelo aprenda de información disponible al momento de
    la predicción.
    """
    # Patrones a excluir de las features
    exclude_patterns = ['_zscore', '_outlier', 'Target','Age','Target_encoded', 'Target_binary', '2nd sem', 'grade','enrolled', 'approved']
    exclude_exact = ['Nacionality', "Mother's occupation", 'Unemployment rate', 'Curricular units 1st sem (evaluations)','Tasa_aprobacion_1sem']

    # Filtrar columnas que no contengan ninguno de los patrones excluidos ni sean exactamente 'Nationality'
    feature_columns = [col for col in df.columns
                       if not any(pattern in col for pattern in exclude_patterns)
                       and col not in exclude_exact]

    print(f"\n🔧 Features seleccionadas para el modelo: {len(feature_columns)}")

    return feature_columns

def verificar_linealidad_logit(df, features, target_col='Target_binary'):
    """
    Verifica la linealidad del logit para variables continuas usando Box-Tidwell y gráficos.
    - Cada predictor continuo debe relacionarse linealmente con el logit de la probabilidad.
    - Si no se cumple, se recomienda transformar la variable.
    """

    print("\n" + "="*70)
    print("VERIFICACIÓN DE LINEALIDAD DEL LOGIT (Box-Tidwell)")
    print("="*70)

    # Seleccionar variables continuas de la lista predefinida
    from sklearn.preprocessing import PowerTransformer
    lista_numericas = [
        'Age at enrollment',
        'GDP',
        'Inflation rate',
        'Unemployment rate',
        'Curricular units 1st sem (grade)',
        'Curricular units 1st sem (credited)',
        'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (approved)', 
        'Curricular units 1st sem (without evaluations)', 
        'Tasa_aprobacion_1sem'
    ]
    continuous_vars = [col for col in features if col in lista_numericas and pd.api.types.is_numeric_dtype(df[col])]
    if not continuous_vars:
        print("No se encontraron variables continuas (de la lista definida y presentes en features) para verificar linealidad del logit.")
        return df
    df_num = df[continuous_vars].copy()

    # Box-Tidwell: agregar término de interacción variable*log(variable)
    results = {}
    for var in continuous_vars:
        x = df_num[var].copy()
        x = pd.to_numeric(x, errors='coerce')
        n_nulos = x.isnull().sum()
        if n_nulos > 0:
            print(f"   ⚠️ {var}: {n_nulos} valores nulos/no numéricos eliminados para Box-Tidwell.")
        x = x.dropna()
        x = x.apply(lambda v: v if v > 0 else 1e-6)
        safe_var = f"var_{continuous_vars.index(var)}"
        safe_log = f"{safe_var}_log"
        df_bt = pd.DataFrame({
            safe_var: x,
            safe_log: x.apply(np.log),
            target_col: df.loc[x.index, target_col]
        })
        formula = f"{target_col} ~ {safe_var} + {safe_log}"
        try:
            model = logit(formula, data=df_bt).fit(disp=0)
            p_value = model.pvalues.get(safe_log, np.nan)
            results[var] = p_value
        except Exception as e:
            print(f"   ⚠️ No se pudo ajustar Box-Tidwell para {var}: {e}")
            results[var] = np.nan

    print("\nResultados Box-Tidwell (p-valor para término log):\n" + "-"*55)
    no_lineales = []
    for var, pval in results.items():
        if np.isnan(pval):
            print(f"   • {var:<40} | No calculado")
        elif pval < 0.05:
            print(f"   • {var:<40} | ❌ p = {pval:.4f} | NO lineal, se intentará transformar")
            no_lineales.append(var)
        else:
            print(f"   • {var:<40} | ✅ p = {pval:.4f} | Linealidad aceptable")
    print("-"*55 + "\n")

    # Intentar transformar variables no lineales
    transformaciones = {
        'log': lambda x: np.log(np.where(x > 0, x, 1e-6)),
        'sqrt': lambda x: np.sqrt(np.where(x >= 0, x, 0)),
        'square': lambda x: np.power(x, 2),
        'inverse': lambda x: 1.0 / np.where(x != 0, x, 1e-6),
        'yeo-johnson': None  # Usaremos PowerTransformer
    }
    cambios = {}
    for var in no_lineales:
        x_orig = pd.to_numeric(df[var], errors='coerce').fillna(1e-6)
        mejor_pval = None
        mejor_nombre = None
        mejor_x = None
        for nombre, func in transformaciones.items():
            if nombre == 'yeo-johnson':
                try:
                    pt = PowerTransformer(method='yeo-johnson')
                    x_tr = pt.fit_transform(x_orig.values.reshape(-1,1)).flatten()
                except Exception:
                    continue
            else:
                try:
                    x_tr = func(x_orig)
                except Exception:
                    continue
            # Box-Tidwell con variable transformada
            safe_var = 'var_tr'
            safe_log = 'var_tr_log'
            df_bt = pd.DataFrame({
                safe_var: x_tr,
                safe_log: np.log(np.where(x_tr > 0, x_tr, 1e-6)),
                target_col: df[target_col]
            })
            formula = f"{target_col} ~ {safe_var} + {safe_log}"
            try:
                model = logit(formula, data=df_bt).fit(disp=0)
                pval = model.pvalues.get(safe_log, np.nan)
                if not np.isnan(pval) and (mejor_pval is None or pval > mejor_pval):
                    mejor_pval = pval
                    mejor_nombre = nombre
                    mejor_x = x_tr
            except Exception:
                continue
        if mejor_pval is not None and mejor_pval >= 0.05:
            print(f"   ✅ {var:<40} | Transformada con {mejor_nombre:<10} | p = {mejor_pval:.4f}\n")
            df[var] = mejor_x
            cambios[var] = mejor_nombre
        else:
            print(f"   ⚠️ {var:<40} | No pudo ser transformada para cumplir linealidad\n")

    if cambios:
        print("Variables transformadas para cumplir linealidad:\n" + "-"*55)
        for var, trans in cambios.items():
            print(f"   • {var:<40} | {trans}")
        print("-"*55 + "\n")
    else:
        print("No se realizaron transformaciones automáticas.\n")

    # Inspección gráfica
    print("\nInspección gráfica de la relación logit vs variable continua:")
    n_vars = len(continuous_vars)
    if n_vars == 0:
        print("No hay variables continuas para graficar.")
        return df
    ncols = 2
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows))
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for idx, var in enumerate(continuous_vars):
        ax = axes[idx]
        try:
            X = sm.add_constant(df[var])
            y = df[target_col]
            model = sm.Logit(y, X).fit(disp=0)
            logit_pred = model.predict(X)
            ax.scatter(df[var], logit_pred, alpha=0.3)
            ax.set_xlabel(var)
            ax.set_ylabel('Logit estimado')
            ax.set_title(f'Logit vs {var}')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"No se pudo graficar\n{var}\n{e}", ha='center', va='center', fontsize=10)
            print(f"   ⚠️ No se pudo graficar {var}: {e}")
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout(pad=1.0, h_pad=1.0)
    plt.show()

    print("\nSi alguna variable muestra relación no lineal, considerar transformar (log, raíz, polinomio) o categorizar.")
    return df

def verificar_multicolinealidad(df, features, threshold=10.0):
    """
    Verifica la ausencia de multicolinealidad fuerte entre predictores.
    - Calcula VIF (Variance Inflation Factor) para cada variable numérica.
    - Si VIF > threshold, hay multicolinealidad fuerte.
    """

    print("\n" + "="*70)
    print("VERIFICACIÓN DE MULTICOLINEALIDAD (VIF y correlación)")
    print("="*70)

    # Seleccionar solo variables numéricas
    df_num = df[features].select_dtypes(include=[np.number]).copy()
    if df_num.shape[1] < 2:
        print("No hay suficientes variables numéricas para analizar multicolinealidad.")
        return

    # Calcular VIF
    vif_data = pd.DataFrame()
    vif_data['Variable'] = df_num.columns
    vif_data['VIF'] = [variance_inflation_factor(df_num.values, i) for i in range(df_num.shape[1])]

    print("\nValores de VIF:")
    for _, row in vif_data.iterrows():
        if row['VIF'] > threshold:
            print(f"   - {row['Variable']}: ❌ VIF = {row['VIF']:.2f} (multicolinealidad fuerte)")
        else:
            print(f"   - {row['Variable']}: ✅ VIF = {row['VIF']:.2f}")

    # Calcular porcentajes de variables en rangos de VIF
    total_vars = len(vif_data)
    count_1_2 = ((vif_data['VIF'] >= 1) & (vif_data['VIF'] < 2)).sum()
    count_2_5 = ((vif_data['VIF'] >= 2) & (vif_data['VIF'] < 5)).sum()
    count_5_10 = ((vif_data['VIF'] >= 5) & (vif_data['VIF'] < 10)).sum()
    pct_1_2 = count_1_2 / total_vars * 100 if total_vars > 0 else 0
    pct_2_5 = count_2_5 / total_vars * 100 if total_vars > 0 else 0
    pct_5_10 = count_5_10 / total_vars * 100 if total_vars > 0 else 0
    print("\nDistribución de VIF en variables predictoras:")
    print(f"   - VIF entre 1 y 2:    {count_1_2} variables ({pct_1_2:.1f}%)")
    print(f"   - VIF entre 2 y 5:    {count_2_5} variables ({pct_2_5:.1f}%)")
    print(f"   - VIF entre 5 y 10:   {count_5_10} variables ({pct_5_10:.1f}%)")

def verificar_tamanio_muestra_epv(df, features, target_col='Target_binary', epv_min=10):
    """
    Verifica si el tamaño de muestra es adecuado para regresión logística según la regla clásica:
    - EPV (eventos por predictor) ≥ 10
    - EPV = min(n_eventos_clase_1, n_eventos_clase_0) / n_predictors
    """
    print("\n" + "="*70)
    print("VERIFICACIÓN DE TAMAÑO DE MUESTRA (Regla clásica EPV ≥ 10)")
    print("="*70)
    
    n_obs = len(df)
    n_vars = len(features)
    n_eventos_1 = (df[target_col] == 1).sum()
    n_eventos_0 = (df[target_col] == 0).sum()
    epv = min(n_eventos_1, n_eventos_0) / n_vars if n_vars > 0 else 0
    print(f"Total de observaciones: {n_obs}")
    print(f"Variables predictoras: {n_vars}")
    print(f"Eventos clase 1 (Graduate): {n_eventos_1}")
    print(f"Eventos clase 0 (Dropout): {n_eventos_0}")
    print(f"EPV (eventos por predictor): {epv:.2f}")
    print(f"Mínimo recomendado (EPV ≥ {epv_min})")
    if epv >= epv_min:
        print("\n✅ Tamaño de muestra adecuado según la regla clásica EPV ≥ 10.")
    else:
        print("\n⚠️ Tamaño de muestra POTENCIALMENTE INSUFICIENTE para la cantidad de predictores. Considere reducir el número de variables o recolectar más datos.")
    def verificar_tamanio_muestra(df, features, target_col='Target_binary', min_per_variable=10):
        """
        Verifica si el tamaño de muestra es adecuado para la regresión logística.
        Regla común: al menos 10 casos por variable predictora para cada clase de la variable objetivo.
        """
        print("\n==============================")
        print("VERIFICACIÓN DE TAMAÑO DE MUESTRA ADECUADO")
        print("==============================")
        n_obs = len(df)
        n_vars = len(features)
        n_success = (df[target_col] == 1).sum()
        n_failure = (df[target_col] == 0).sum()
        min_class = min(n_success, n_failure)
        min_required = n_vars * min_per_variable
        print(f"Total de observaciones: {n_obs}")
        print(f"Variables predictoras: {n_vars}")
        print(f"Casos clase 1 (éxito): {n_success}")
        print(f"Casos clase 0 (abandono): {n_failure}")
        print(f"Casos mínimos por clase: {min_class}")
        print(f"Mínimo recomendado (10 por variable): {min_required}")
        if min_class >= min_required:
            print("\n✅ Tamaño de muestra adecuado para la regresión logística.")
        else:
            print("\n⚠️ Tamaño de muestra POTENCIALMENTE INSUFICIENTE para la cantidad de variables. Considere reducir el número de predictores o recolectar más datos.")
        print("Nota: Esta es una regla empírica. Si el modelo converge y los resultados son estables, puede ser aceptable con menos casos, pero aumenta el riesgo de sobreajuste.")

def prepare_train_test_split(df, feature_columns):
    """
    Preparar conjuntos de entrenamiento y prueba con estratificación.
    División típica: 80% entrenamiento, 20% prueba
    """
    X = df[feature_columns]   # Variables predictoras (features)
    y = df['Target_binary']   # Variable objetivo (0=Dropout, 1=Graduate)
    
    # Dividir manteniendo proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y  # Mantener proporción de Dropout/Graduate en ambos sets
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
    Estandarizar features para regresión logística.
    
    La estandarización (StandardScaler) transforma cada variable para que tenga:
    - Media = 0
    - Desviación estándar = 1
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Ajustar y transformar
    X_test_scaled = scaler.transform(X_test)        # Solo transformar
    
    return X_train_scaled, X_test_scaled, scaler

def train_logistic_regression(X_train, y_train):
    """
    Entrenar modelo de regresión logística con regularización.
    """
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO DEL MODELO DE REGRESIÓN LOGÍSTICA")
    print("=" * 70)
    
    # Configurar y entrenar el modelo
    model = LogisticRegression(
        penalty='l2',           # Regularización L2 (Ridge) para evitar sobreajuste
        C=1.0,                  # Inverso de la fuerza de regularización
        solver='lbfgs',         # Algoritmo de optimización quasi-Newton
        max_iter=1000,          # Máximo de iteraciones para convergencia
        random_state=RANDOM_STATE,
        class_weight='balanced' # Compensar desbalance entre clases
    )
    
    model.fit(X_train, y_train)  # Ajustar modelo a datos de entrenamiento
    
    print("\n✅ Modelo entrenado exitosamente")
    print(f"   - Regularización: L2 (Ridge)")
    print(f"   - Solver: LBFGS")
    print(f"   - Class weight: Balanced")
    
    return model

def perform_cross_validation(model, X_train, y_train):
    """
    Realizar validación cruzada k-fold para evaluar estabilidad del modelo.
    
    Validación cruzada (5-fold):
    1. Divide los datos en 5 partes iguales
    2. Entrena en 4 partes, evalúa en la 5ta
    3. Repite 5 veces, cada parte siendo el conjunto de prueba una vez
    4. Promedia los resultados
    """
    print("\n" + "=" * 70)
    print("VALIDACIÓN CRUZADA (5-FOLD)")
    print("=" * 70)
    
    # Configurar validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Evaluar múltiples métricas usando validación cruzada
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
    Evaluación completa del modelo en el conjunto de prueba.
    
    Métricas calculadas:
    - Accuracy: proporción de predicciones correctas totales
    - Precision: de los predichos como Graduate, ¿cuántos realmente lo son?
    - Recall: de los Graduate reales, ¿cuántos identificamos?
    - F1-Score: media armónica de precision y recall
    - ROC-AUC: capacidad de distinguir entre clases (0.5=azar, 1=perfecto)
    - Average Precision: resumen de curva precision-recall
    
    Matriz de confusión:
    - Verdaderos Negativos (TN): Dropout predicho correctamente
    - Falsos Positivos (FP): Dropout predicho como Graduate (error Tipo I)
    - Falsos Negativos (FN): Graduate predicho como Dropout (error Tipo II)
    - Verdaderos Positivos (TP): Graduate predicho correctamente
    """
    print("\n" + "=" * 70)
    print("EVALUACIÓN DEL MODELO EN CONJUNTO DE PRUEBA")
    print("=" * 70)
    
    # Generar predicciones
    y_pred = model.predict(X_test)              # Clase predicha (0 o 1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de Graduate
    
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
    Analizar y clasificar la importancia de variables basada en coeficientes.
    
    En regresión logística, los coeficientes indican:
    - Coeficiente > 0: aumenta probabilidad de Graduate
    - Coeficiente < 0: aumenta probabilidad de Dropout
    
    Odds Ratio (OR) = exp(coeficiente):
    - OR > 1: factor favorece graduación
    - OR < 1: factor favorece abandono
    - OR = 1: sin efecto
    
    """
    print("\n" + "=" * 70)
    print("ANÁLISIS DE IMPORTANCIA DE VARIABLES")
    print("=" * 70)
    
    # Obtener coeficientes del modelo entrenado
    coefficients = model.coef_[0]
    
    # Crear DataFrame con métricas de importancia
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients),  # Valor absoluto para ordenar
        'Odds_Ratio': np.exp(coefficients)        # Transformar a odds ratio
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\n📊 Top 12 Variables más Importantes (por magnitud del coeficiente):")
    print("\n   " + "-" * 75)
    print(f"   {'#':<3} {'Variable':<45} {'Coef.':>10} {'Odds Ratio':>12}")
    print("   " + "-" * 75)
    
    for i, (_, row) in enumerate(feature_importance.head(12).iterrows()):
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
    Crear visualizaciones completas para los resultados de clasificación.
    
    Gráficos generados:
    1. Matriz de confusión: visualiza aciertos y errores del modelo
    2. Curva ROC: capacidad de discriminación a diferentes umbrales
    3. Curva Precision-Recall: rendimiento en diferentes puntos de corte
    4. Importancia de features: qué variables influyen más
    5. Distribución de probabilidades: separación entre clases
    6. Resumen de métricas: vista general del rendimiento
    
    También genera gráfico separado de Odds Ratios para interpretación.
    """
    print("\n" + "=" * 70)
    print("GENERACIÓN DE VISUALIZACIONES")
    print("=" * 70)
    
    # Solo lo esencial: matriz de confusión, curva ROC, importancia de variables y resumen de métricas
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 0])
    axs[0, 0].set_title('Matriz de Confusión')
    axs[0, 0].set_xlabel('Predicción')
    axs[0, 0].set_ylabel('Valor Real')

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axs[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.2f})')
    axs[0, 1].plot([0, 1], [0, 1], 'r--', label='Aleatorio')
    axs[0, 1].set_title('Curva ROC')
    axs[0, 1].set_xlabel('Falsos Positivos')
    axs[0, 1].set_ylabel('Verdaderos Positivos')
    axs[0, 1].legend()

    # Importancia de variables (Top 10)
    top_features = feature_importance.head(10)
    axs[1, 0].barh(top_features['Feature'], top_features['Coefficient'], color='skyblue')
    axs[1, 0].set_title('Top 10 Variables')
    axs[1, 0].set_xlabel('Coeficiente')
    axs[1, 0].invert_yaxis()

    # Resumen de métricas
    names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['roc_auc']]
    axs[1, 1].bar(names, values, color='lightgreen')
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_title('Métricas')
    for i, v in enumerate(values):
        axs[1, 1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'resultados_esenciales.png')

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()  # Mostrar la imagen en una ventana de matplotlib
    plt.close()

    print(f"\n✅ Visualización esencial guardada en: {fig_path}")
    return [fig_path]

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
    Generar conclusiones finales y responder a la pregunta de investigación.
    
    Criterios de evaluación del rendimiento (basados en ROC-AUC):
    - >= 0.85: Excelente - modelo muy confiable
    - >= 0.75: Bueno - modelo útil para aplicaciones prácticas
    - >= 0.65: Moderado - modelo tiene valor pero con limitaciones
    - < 0.65: Limitado - requiere mejoras significativas
    
    Esta función proporciona:
    - Respuesta directa a la pregunta de investigación
    - Resumen de métricas clave
    - Identificación de factores predictivos importantes
    - Implicaciones prácticas para intervención
    - Limitaciones y consideraciones éticas
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
    # 1. Cargar y preparar datos
    df = load_and_prepare_data(INPUT_PATH)

    # 2. Seleccionar features
    feature_columns = select_features(df)

    # 3. Verificar linealidad del logit y transformar variables si es necesario
    df = verificar_linealidad_logit(df, feature_columns, target_col='Target_binary')

    # 4. Verificar multicolinealidad
    verificar_multicolinealidad(df, feature_columns, threshold=10.0)

    # 5. Verificar tamaño de muestra EPV
    verificar_tamanio_muestra_epv(df, feature_columns, target_col='Target_binary', epv_min=10)

    # 6. Preparar conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = prepare_train_test_split(df, feature_columns)

    # 7. Estandarizar features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 8. Entrenar modelo de regresión logística
    model = train_logistic_regression(X_train_scaled, y_train)

    # 9. Validación cruzada
    cv_results = perform_cross_validation(model, X_train_scaled, y_train)

    # 10. Evaluar modelo en conjunto de prueba
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test_scaled, y_test, feature_columns)

    # 11. Analizar importancia de variables
    feature_importance = analyze_feature_importance(model, feature_columns)

    # 12. Crear visualizaciones
    output_dir = os.path.join('outputs', 'prediction_results')
    os.makedirs(output_dir, exist_ok=True)
    create_visualizations(model, X_test_scaled, y_test, y_pred, y_pred_proba, feature_importance, metrics, output_dir)

    # 13. Guardar resultados
    save_results(metrics, cv_results, feature_importance, output_dir)

    # 14. Generar conclusiones
    generate_conclusions(metrics, cv_results, feature_importance)

if __name__ == "__main__":
    main()