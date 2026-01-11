"""
Script de Preparación de Datos para el análisis de abandono estudiantil.

Este módulo implementa todas las transformaciones necesarias para que los datos
sean aptos para el análisis estadístico y machine learning. Incluye:
    - Manejo de valores faltantes
    - Codificación de variables categóricas
    - Estandarización/normalización de variables numéricas
    - Generación de reportes de transformaciones aplicadas

Uso:
    python scripts/data_preparation.py --input "dataset/dataset.csv" --output "outputs/prepared_data"

Autor: Statistics Project
"""

import argparse
import json
import re
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# CONFIGURACIÓN DE VARIABLES
# =============================================================================

# Variables que representan códigos categóricos (aunque sean numéricas en el CSV)
CATEGORICAL_CODED_VARS = [
    'Marital status',
    'Application mode',
    'Course',
    'Daytime/evening attendance',
    'Previous qualification',
    'Nacionality',
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'International'
]

# Variable objetivo
TARGET_VAR = 'Target'

# Variables numéricas continuas (para estandarización)
CONTINUOUS_NUMERIC_VARS = [
    'Application order',
    'Age at enrollment',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]


def safe_filename(name: str) -> str:
    """Sanitiza una cadena para que sea segura como nombre de archivo.
    
    Args:
        name: Cadena de entrada a sanitizar.
    
    Returns:
        Cadena válida para nombre de fichero.
    """
    return re.sub(r'[<>:"/\\|?*]', '_', str(name))


# =============================================================================
# ANÁLISIS Y MANEJO DE VALORES FALTANTES
# =============================================================================

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analiza los valores faltantes en el DataFrame.
    
    Genera un reporte detallado con el conteo y porcentaje de valores
    faltantes por cada columna.
    
    Args:
        df: DataFrame a analizar.
    
    Returns:
        DataFrame con el análisis de valores faltantes.
    """
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    missing_report = pd.DataFrame({
        'variable': df.columns,
        'valores_faltantes': missing_count.values,
        'porcentaje_faltantes': missing_pct.values,
        'tipo_dato': df.dtypes.values
    })
    
    missing_report = missing_report.sort_values('porcentaje_faltantes', ascending=False)
    missing_report = missing_report.reset_index(drop=True)
    
    return missing_report


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'auto',
    threshold_drop: float = 50.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Maneja los valores faltantes según la estrategia especificada.
    
    Justificación de estrategias:
    - Para variables numéricas: se utiliza la mediana por ser robusta a outliers.
    - Para variables categóricas: se utiliza la moda (valor más frecuente).
    - Columnas con más del threshold_drop% de faltantes se eliminan.
    
    Args:
        df: DataFrame con datos originales.
        strategy: Estrategia de imputación ('auto', 'median', 'mean', 'mode', 'drop').
        threshold_drop: Porcentaje umbral para eliminar columnas con muchos faltantes.
    
    Returns:
        Tuple con (DataFrame procesado, diccionario de transformaciones aplicadas).
    """
    df_clean = df.copy()
    transformations = {
        'missing_values': {
            'strategy': strategy,
            'threshold_drop': threshold_drop,
            'columns_dropped': [],
            'imputations': {}
        }
    }
    
    # Paso 1: Eliminar columnas con demasiados valores faltantes
    missing_pct = (df_clean.isnull().sum() / len(df_clean)) * 100
    cols_to_drop = missing_pct[missing_pct > threshold_drop].index.tolist()
    
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        transformations['missing_values']['columns_dropped'] = cols_to_drop
        print(f"⚠️  Columnas eliminadas por exceder {threshold_drop}% de faltantes: {cols_to_drop}")
    
    # Paso 2: Imputar valores faltantes restantes
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            n_missing = df_clean[col].isnull().sum()
            
            if strategy == 'auto':
                # Estrategia automática basada en el tipo de dato
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    # Mediana para numéricos (robusta a outliers)
                    impute_value = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(impute_value)
                    method = 'mediana'
                else:
                    # Moda para categóricos
                    impute_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(impute_value)
                    method = 'moda'
                    
            elif strategy == 'median':
                impute_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(impute_value)
                method = 'mediana'
                
            elif strategy == 'mean':
                impute_value = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(impute_value)
                method = 'media'
                
            elif strategy == 'mode':
                impute_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(impute_value)
                method = 'moda'
                
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                impute_value = None
                method = 'eliminación de filas'
            
            transformations['missing_values']['imputations'][col] = {
                'n_missing': int(n_missing),
                'method': method,
                'value': str(impute_value) if impute_value is not None else None
            }
            
            print(f"✓ {col}: {n_missing} valores faltantes imputados con {method}")
    
    return df_clean, transformations


# =============================================================================
# CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# =============================================================================

def encode_target_variable(
    df: pd.DataFrame,
    target_col: str = 'Target'
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Codifica la variable objetivo (Target) con Label Encoding.
    
    Justificación:
    La variable Target tiene 3 categorías ordenables conceptualmente:
    - Dropout (abandono): 0
    - Enrolled (matriculado): 1  
    - Graduate (graduado): 2
    
    Se utiliza Label Encoding porque existe un orden natural en el resultado
    académico del estudiante.
    
    Args:
        df: DataFrame con la variable objetivo.
        target_col: Nombre de la columna objetivo.
    
    Returns:
        Tuple con (DataFrame modificado, mapeo de codificación).
    """
    df_encoded = df.copy()
    
    # Mapeo ordenado conceptualmente
    target_mapping = {
        'Dropout': 0,
        'Enrolled': 1,
        'Graduate': 2
    }
    
    if target_col in df_encoded.columns:
        df_encoded[f'{target_col}_encoded'] = df_encoded[target_col].map(target_mapping)
        print(f"✓ Variable objetivo '{target_col}' codificada:")
        for k, v in target_mapping.items():
            print(f"    {k} → {v}")
    
    return df_encoded, target_mapping


def identify_variable_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identifica y clasifica las variables por tipo.
    
    Justificación:
    Aunque muchas variables categóricas aparecen como números en el dataset
    (códigos), es importante identificarlas correctamente para aplicar
    transformaciones apropiadas.
    
    Args:
        df: DataFrame a analizar.
    
    Returns:
        Diccionario con listas de variables por tipo.
    """
    var_types = {
        'categorical_coded': [],  # Categóricas representadas como códigos numéricos
        'categorical_string': [], # Categóricas como texto
        'numeric_continuous': [], # Numéricas continuas
        'binary': [],             # Binarias (0/1)
        'target': []              # Variable objetivo
    }
    
    for col in df.columns:
        if col == TARGET_VAR:
            var_types['target'].append(col)
        elif col in CATEGORICAL_CODED_VARS:
            # Verificar si es binaria (solo 0 y 1)
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                var_types['binary'].append(col)
            else:
                var_types['categorical_coded'].append(col)
        elif col in CONTINUOUS_NUMERIC_VARS:
            var_types['numeric_continuous'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Numérica no clasificada previamente
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 10:
                var_types['categorical_coded'].append(col)
            else:
                var_types['numeric_continuous'].append(col)
        else:
            var_types['categorical_string'].append(col)
    
    return var_types


def apply_label_encoding(
    df: pd.DataFrame,
    columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Aplica Label Encoding a las columnas especificadas.
    
    Justificación:
    Se utiliza Label Encoding para variables categóricas ordinales o cuando
    el número de categorías es pequeño y el algoritmo posterior puede
    manejar variables numéricas discretas.
    
    Args:
        df: DataFrame con los datos.
        columns: Lista de columnas a codificar.
    
    Returns:
        Tuple con (DataFrame modificado, diccionario de mapeos).
    """
    df_encoded = df.copy()
    mappings = {}
    
    for col in columns:
        if col in df_encoded.columns:
            unique_vals = sorted(df_encoded[col].dropna().unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            df_encoded[f'{col}_le'] = df_encoded[col].map(mapping)
            mappings[col] = mapping
            print(f"✓ Label Encoding aplicado a '{col}': {len(unique_vals)} categorías")
    
    return df_encoded, mappings


def apply_one_hot_encoding(
    df: pd.DataFrame,
    columns: List[str],
    drop_first: bool = True,
    max_categories: int = 10
) -> Tuple[pd.DataFrame, List[str]]:
    """Aplica One-Hot Encoding a las columnas especificadas.
    
    Justificación:
    Se utiliza One-Hot Encoding para variables categóricas nominales (sin orden)
    cuando se requiere evitar que el modelo asuma un orden entre categorías.
    Se usa drop_first=True para evitar multicolinealidad perfecta.
    
    Args:
        df: DataFrame con los datos.
        columns: Lista de columnas a codificar.
        drop_first: Si eliminar la primera categoría (referencia).
        max_categories: Máximo de categorías para aplicar OHE (evitar explosión dimensional).
    
    Returns:
        Tuple con (DataFrame modificado, lista de nuevas columnas creadas).
    """
    df_encoded = df.copy()
    new_columns = []
    
    for col in columns:
        if col in df_encoded.columns:
            n_unique = df_encoded[col].nunique()
            
            if n_unique > max_categories:
                print(f"⚠️  '{col}' tiene {n_unique} categorías (> {max_categories}). "
                      f"Se recomienda Label Encoding o agrupación.")
                continue
            
            # Crear dummies
            dummies = pd.get_dummies(
                df_encoded[col],
                prefix=col,
                drop_first=drop_first,
                dtype=int
            )
            
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            new_columns.extend(dummies.columns.tolist())
            print(f"✓ One-Hot Encoding aplicado a '{col}': {len(dummies.columns)} columnas creadas")
    
    return df_encoded, new_columns


# =============================================================================
# ESTANDARIZACIÓN Y NORMALIZACIÓN
# =============================================================================

def standardize_variables(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'zscore'
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Estandariza variables numéricas.
    
    Justificación:
    - Z-score (StandardScaler): Transforma los datos para tener media=0 y std=1.
      Útil cuando los datos siguen una distribución aproximadamente normal
      y se quiere comparar variables en diferentes escalas.
    - Min-Max: Escala los datos al rango [0, 1]. Útil cuando se necesita
      un rango acotado y los datos no tienen outliers severos.
    - Robust: Usa mediana e IQR, resistente a outliers.
    
    Args:
        df: DataFrame con los datos.
        columns: Lista de columnas a estandarizar.
        method: Método de estandarización ('zscore', 'minmax', 'robust').
    
    Returns:
        Tuple con (DataFrame modificado, parámetros de estandarización).
    """
    df_scaled = df.copy()
    scaling_params = {}
    
    for col in columns:
        if col not in df_scaled.columns:
            continue
            
        if not pd.api.types.is_numeric_dtype(df_scaled[col]):
            continue
        
        values = df_scaled[col].dropna()
        
        if method == 'zscore':
            mean = values.mean()
            std = values.std()
            if std == 0:
                std = 1  # Evitar división por cero
            df_scaled[f'{col}_zscore'] = (df_scaled[col] - mean) / std
            scaling_params[col] = {'method': 'zscore', 'mean': float(mean), 'std': float(std)}
            
        elif method == 'minmax':
            min_val = values.min()
            max_val = values.max()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1
            df_scaled[f'{col}_minmax'] = (df_scaled[col] - min_val) / range_val
            scaling_params[col] = {'method': 'minmax', 'min': float(min_val), 'max': float(max_val)}
            
        elif method == 'robust':
            median = values.median()
            q75 = values.quantile(0.75)
            q25 = values.quantile(0.25)
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1
            df_scaled[f'{col}_robust'] = (df_scaled[col] - median) / iqr
            scaling_params[col] = {'method': 'robust', 'median': float(median), 'iqr': float(iqr)}
        
        print(f"✓ {col}: estandarizado con método '{method}'")
    
    return df_scaled, scaling_params


def detect_and_handle_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'iqr',
    action: str = 'flag'
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Detecta y maneja outliers en variables numéricas.
    
    Justificación:
    Los outliers pueden distorsionar análisis estadísticos y modelos predictivos.
    Se ofrece la opción de marcarlos (flag), recortarlos (cap) o eliminarlos (remove).
    
    Args:
        df: DataFrame con los datos.
        columns: Lista de columnas a analizar.
        method: Método de detección ('iqr' o 'zscore').
        action: Acción a tomar ('flag', 'cap', 'remove').
    
    Returns:
        Tuple con (DataFrame modificado, conteo de outliers por columna).
    """
    df_processed = df.copy()
    outlier_counts = {}
    
    for col in columns:
        if col not in df_processed.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            continue
        
        values = df_processed[col].dropna()
        
        if method == 'iqr':
            q75 = values.quantile(0.75)
            q25 = values.quantile(0.25)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
        elif method == 'zscore':
            mean = values.mean()
            std = values.std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
        
        # Identificar outliers
        outlier_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        n_outliers = outlier_mask.sum()
        outlier_counts[col] = int(n_outliers)
        
        if n_outliers > 0:
            if action == 'flag':
                df_processed[f'{col}_outlier'] = outlier_mask.astype(int)
            elif action == 'cap':
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            elif action == 'remove':
                df_processed = df_processed[~outlier_mask]
            
            print(f"✓ {col}: {n_outliers} outliers detectados (método: {method}, acción: {action})")
    
    return df_processed, outlier_counts


# =============================================================================
# GENERACIÓN DE REPORTES
# =============================================================================

def generate_transformation_report(
    transformations: Dict[str, Any],
    output_path: Path
) -> None:
    """Genera un reporte de todas las transformaciones aplicadas.
    
    Args:
        transformations: Diccionario con todas las transformaciones.
        output_path: Ruta donde guardar el reporte.
    """
    report_path = output_path / 'transformation_report.json'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(transformations, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Reporte de transformaciones guardado en: {report_path}")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def prepare_data(
    input_path: str,
    output_dir: str,
    missing_strategy: str = 'auto',
    scaling_method: str = 'zscore',
    handle_outliers: bool = True,
    outlier_action: str = 'flag'
) -> pd.DataFrame:
    """Ejecuta el pipeline completo de preparación de datos.
    
    Args:
        input_path: Ruta al archivo CSV de entrada.
        output_dir: Directorio para guardar los resultados.
        missing_strategy: Estrategia para valores faltantes.
        scaling_method: Método de estandarización.
        handle_outliers: Si detectar y manejar outliers.
        outlier_action: Acción para outliers.
    
    Returns:
        DataFrame preparado para análisis.
    """
    print("=" * 70)
    print("🔧 PREPARACIÓN DE DATOS - Pipeline de Transformaciones")
    print("=" * 70)
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    print(f"\n📂 Cargando datos desde: {input_path}")
    df_original = pd.read_csv(input_path)
    print(f"   Dimensiones: {df_original.shape[0]:,} filas × {df_original.shape[1]} columnas")
    
    # Diccionario para almacenar todas las transformaciones
    all_transformations = {}
    
    # 1. ANÁLISIS Y MANEJO DE VALORES FALTANTES
    print("\n" + "=" * 70)
    print("📋 PASO 1: Análisis y Manejo de Valores Faltantes")
    print("=" * 70)
    
    missing_report = analyze_missing_values(df_original)
    
    total_missing = missing_report['valores_faltantes'].sum()
    if total_missing > 0:
        print(f"\n⚠️  Se encontraron {total_missing:,} valores faltantes en total")
        df_clean, missing_transforms = handle_missing_values(
            df_original, 
            strategy=missing_strategy
        )
        all_transformations.update(missing_transforms)
    else:
        print("\n✅ No se encontraron valores faltantes en el dataset")
        df_clean = df_original.copy()
        all_transformations['missing_values'] = {
            'strategy': missing_strategy,
            'columns_dropped': [],
            'imputations': {}
        }
    
    # 2. IDENTIFICACIÓN DE TIPOS DE VARIABLES
    print("\n" + "=" * 70)
    print("📋 PASO 2: Identificación de Tipos de Variables")
    print("=" * 70)
    
    var_types = identify_variable_types(df_clean)
    print(f"\n   Variables categóricas (codificadas): {len(var_types['categorical_coded'])}")
    print(f"   Variables binarias: {len(var_types['binary'])}")
    print(f"   Variables numéricas continuas: {len(var_types['numeric_continuous'])}")
    print(f"   Variable objetivo: {var_types['target']}")
    
    all_transformations['variable_types'] = var_types
    
    # 3. CODIFICACIÓN DE VARIABLE OBJETIVO
    print("\n" + "=" * 70)
    print("📋 PASO 3: Codificación de Variable Objetivo")
    print("=" * 70)
    
    df_encoded, target_mapping = encode_target_variable(df_clean)
    all_transformations['target_encoding'] = target_mapping
    
    # 4. ESTANDARIZACIÓN DE VARIABLES NUMÉRICAS
    print("\n" + "=" * 70)
    print("📋 PASO 4: Estandarización de Variables Numéricas")
    print("=" * 70)
    print(f"\n   Método seleccionado: {scaling_method}")
    print("   Justificación: Permite comparar variables en diferentes escalas\n")
    
    numeric_cols = var_types['numeric_continuous']
    df_scaled, scaling_params = standardize_variables(
        df_encoded, 
        numeric_cols, 
        method=scaling_method
    )
    all_transformations['scaling_params'] = scaling_params
    
    # 5. DETECCIÓN DE OUTLIERS (opcional)
    if handle_outliers:
        print("\n" + "=" * 70)
        print("📋 PASO 5: Detección y Manejo de Outliers")
        print("=" * 70)
        
        df_final, outlier_counts = detect_and_handle_outliers(
            df_scaled,
            numeric_cols,
            method='iqr',
            action=outlier_action
        )
        all_transformations['outliers'] = outlier_counts
    else:
        df_final = df_scaled
    
    # 6. GUARDAR RESULTADOS
    print("\n" + "=" * 70)
    print("📋 PASO 6: Guardando Resultados")
    print("=" * 70)
    
    # Guardar dataset preparado
    prepared_path = output_path / 'dataset_prepared.csv'
    df_final.to_csv(prepared_path, index=False)
    print(f"\n✅ Dataset preparado guardado en: {prepared_path}")
    print(f"   Dimensiones finales: {df_final.shape[0]:,} filas × {df_final.shape[1]} columnas")
    
    # Generar reportes
    generate_transformation_report(all_transformations, output_path)
    
    print("\n" + "=" * 70)
    print("✅ PREPARACIÓN DE DATOS COMPLETADA")
    print("=" * 70)
    
    return df_final


def main():
    """Punto de entrada principal del script."""
    parser = argparse.ArgumentParser(
        description='Script de preparación de datos para análisis estadístico',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Ejemplos de uso:
    python data_preparation.py --input dataset/dataset.csv --output outputs/prepared_data
    python data_preparation.py --input dataset/dataset.csv --output outputs/prepared_data --scaling minmax
    python data_preparation.py --input dataset/dataset.csv --output outputs/prepared_data --no-outliers
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='dataset/dataset.csv',
        help='Ruta al archivo CSV de entrada (default: dataset/dataset.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/prepared_data',
        help='Directorio de salida para resultados (default: outputs/prepared_data)'
    )
    
    parser.add_argument(
        '--missing-strategy',
        type=str,
        choices=['auto', 'median', 'mean', 'mode', 'drop'],
        default='auto',
        help='Estrategia para manejar valores faltantes (default: auto)'
    )
    
    parser.add_argument(
        '--scaling',
        type=str,
        choices=['zscore', 'minmax', 'robust'],
        default='zscore',
        help='Método de estandarización (default: zscore)'
    )
    
    parser.add_argument(
        '--no-outliers',
        action='store_true',
        help='Desactivar detección de outliers'
    )
    
    parser.add_argument(
        '--outlier-action',
        type=str,
        choices=['flag', 'cap', 'remove'],
        default='flag',
        help='Acción para outliers: flag=marcar, cap=recortar, remove=eliminar (default: flag)'
    )
    
    args = parser.parse_args()
    
    prepare_data(
        input_path=args.input,
        output_dir=args.output,
        missing_strategy=args.missing_strategy,
        scaling_method=args.scaling,
        handle_outliers=not args.no_outliers,
        outlier_action=args.outlier_action
    )


if __name__ == '__main__':
    main()
