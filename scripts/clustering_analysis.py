"""
Clustering Analysis - Estructura de Niveles de Desempeño
==========================================================
Este script responde a la Pregunta de Investigación:
"¿Cómo se estructuran los diversos niveles de desempeño académico... y en qué 
medida estos perfiles se encuentran vinculados a sus condiciones socioeconómicas 
y a la probabilidad de permanencia o abandono?"

Objetivos Específicos:
1. Caracterizar grupos (Clustering)
2. Comparar diferencias y condiciones socioeconómicas (ANOVA/Chi2)
3. Determinar impacto en deserción/graduación
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
import warnings

# Configuración General
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Rutas
DATA_PATH = "../dataset/dataset.csv"  # O 'outputs/prepared_data/dataset_prepared.csv' si usas el procesado
OUTPUT_DIR = "../outputs/clustering_analysis"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"

# Mapeos para visualización (Consistente con el proyecto)
CATEGORICAL_MAPPINGS = {
    'Marital status': {1: 'Soltero/a', 2: 'Casado/a', 3: 'Viudo/a', 4: 'Divorciado/a', 5: 'Unión de hecho', 6: 'Separado'},
    'Gender': {0: 'Masculino', 1: 'Femenino'},
    'Scholarship holder': {0: 'No Beca', 1: 'Becado'},
    'Debtor': {0: 'No Deudor', 1: 'Deudor'},
    'Tuition fees up to date': {0: 'Mora', 1: 'Al día'},
    'Target': {'Dropout': 'Abandono', 'Graduate': 'Graduado', 'Enrolled': 'Matriculado'}
}

def create_directories():
    """Crea la estructura de carpetas para los resultados."""
    try:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        print(f"✓ Directorios creados: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error creando directorios: {e}")

def load_and_prepare_data(filepath):
    """
    Estructura de Carga de Base de Datos.
    - Carga el CSV.
    - Filtra registros válidos.
    - Genera variables calculadas base (Tasa aprobación, Promedios).
    - Aplica mapeos iniciales.
    """
    print(f"\n⚡ Cargando datos desde: {filepath}")
    
    # 1. Carga
    try:
        # Intentar cargar con separador ';' (formato UCI original)
        df = pd.read_csv(filepath, delimiter=';')
        if df.shape[1] < 2: # Si colapsa todo en una columna, reintentar con ','
             print("⚠️  Detectado posible CSV separado por comas, reintentando...")
             df = pd.read_csv(filepath, delimiter=',')
    except FileNotFoundError:
        print("❌ Error: Archivo no encontrado.")
        return None

    # Normalizar nombres de columnas (eliminar espacios extra al final si existen)
    df.columns = df.columns.str.strip()

    # Verificar si 'Target' existe antes de filtrar
    if 'Target' not in df.columns:
        print(f"❌ Error Crítico: La columna 'Target' no se encuentra en el dataset.")
        print(f"   Columnas disponibles: {list(df.columns)}")
        # Intentar buscar variaciones comunes
        possible_targets = [col for col in df.columns if 'target' in col.lower()]
        if possible_targets:
             print(f"   ¿Quizás quisiste decir: {possible_targets}?")
        return None

    # 2. Filtrado inicial
    # Nota: Para clustering a veces es útil ver a todos, pero para 'Permanencia o Abandono' solemos usar Target definido.
    df = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()

    # 3. Ingeniería de Características Básica (Necesaria para el Clustering)
    # Crear tasas de aprobación si no existen
    if 'Curricular units 1st sem (enrolled)' in df.columns:
        df['Tasa_Aprobacion_Sem1'] = np.where(df['Curricular units 1st sem (enrolled)'] > 0,
                                            df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)'], 0)
    
    if 'Curricular units 2nd sem (enrolled)' in df.columns:
        df['Tasa_Aprobacion_Sem2'] = np.where(df['Curricular units 2nd sem (enrolled)'] > 0,
                                            df['Curricular units 2nd sem (approved)'] / df['Curricular units 2nd sem (enrolled)'], 0)

    # Tasa de asignaturas sin evaluar (Indicador de abandono implícito/ausentismo)
    # Fundamental para distinguir quien reprueba por nota vs quien abandona la materia
    if 'Curricular units 1st sem (without evaluations)' in df.columns:
         df['Tasa_Sin_Evaluacion_Sem1'] = np.where(df['Curricular units 1st sem (enrolled)'] > 0,
                                            df['Curricular units 1st sem (without evaluations)'] / df['Curricular units 1st sem (enrolled)'], 0)
    
    if 'Curricular units 2nd sem (without evaluations)' in df.columns:
         df['Tasa_Sin_Evaluacion_Sem2'] = np.where(df['Curricular units 2nd sem (enrolled)'] > 0,
                                            df['Curricular units 2nd sem (without evaluations)'] / df['Curricular units 2nd sem (enrolled)'], 0)

    # Promedio global de nota
    df['Promedio_Global'] = (df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']) / 2

    print(f"✓ Datos cargados y preparados: {len(df)} registros.")
    return df

# ==============================================================================
# ESQUELETO DE MÉTODOS (Objetivos Específicos)
# ==============================================================================

def perform_clustering_analysis(df):
    """
    OBJETIVO 1: "¿Cómo se estructuran los diversos niveles de desempeño?"
    
    Variables de Clustering (Comportamiento Académico):
    1. Eficacia: Tasa de Aprobación (Sem 1 y 2)
    2. Calidad: Notas Promedio (Sem 1 y 2)
    3. Compromiso: Tasa Sin Evaluación (Sem 1 y 2)
    """
    print(f"\n{'='*70}")
    print("🔍 OBJETIVO 1: CLUSTERING DE DESEMPEÑO ACADÉMICO")
    print(f"{'='*70}")

    # 1. Selección de Variables
    cluster_features = [
        'Tasa_Aprobacion_Sem1', 'Tasa_Aprobacion_Sem2',
        'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)',
        'Tasa_Sin_Evaluacion_Sem1', 'Tasa_Sin_Evaluacion_Sem2'
    ]
    
    # Validar que existan todas
    missing = [col for col in cluster_features if col not in df.columns]
    if missing:
        print(f"⚠️ Faltan variables calculadas: {missing}")
        return df

    print(f"Variables seleccionadas para el modelo: \n{cluster_features}")

    # 2. Preprocesamiento específico para Clustering
    X = df[cluster_features].copy()
    
    # Rellenar NaNs con 0 (Si no tiene nota, es 0 para el modelo de rendimiento)
    X = X.fillna(0)
    
    # Estandarización (Critical para K-Means porque mezclamos escalas 0-1 y 0-20)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. K-Means
    # Usamos k=4 hipótesis inicial: Excelente, Promedio, Riesgo (Bajas notas), Abandono (Sin evaluación)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['Cluster'] = clusters
    
    # 4. Caracterización de Clusters (Perfilamiento Data-Driven)
    print(f"\n📊 Perfilamiento de los {k} Clusters (Media de cada variable):")
    print("-" * 70)
    
    # Calcular promedios por cluster original (sin escalar)
    profile = df.groupby('Cluster')[cluster_features].mean()
    
    # Agregar conteo y porcentaje
    counts = df['Cluster'].value_counts().sort_index()
    profile['N_Estudiantes'] = counts
    profile['% Total'] = (counts / len(df)) * 100
    
    # Ordenar perfil para legibilidad (opcional, ej: por Tasa Aprobación)
    profile = profile.sort_values(by='Tasa_Aprobacion_Sem1', ascending=False)
    
    # Imprimir tabla limpia
    print(profile.round(2).to_string())

    print("\n⚠️ NOTA: Analiza la tabla anterior para asignar nombres a los Clusters.")
    print("   - Cluster con Tasa=1.0 y alta nota -> Probablemente 'Alto Rendimiento'")
    print("   - Cluster con Tasa=0.0 y Tasa_Sin_Eval alta -> Probablemente 'Deserción / Ausentismo'")
    
    # Crear etiquetas genéricas basadas en el orden para facilitar gráficos siguientes
    # (Ej: 'Grupo A', 'Grupo B'...) en lugar de intentar adivinar el nombre semántico
    rank_mapping = {original_idx: f"Grupo {i+1}" for i, original_idx in enumerate(profile.index)}
    df['Cluster_Label'] = df['Cluster'].map(rank_mapping)
    
    print(f"\n🏷️  Etiquetas asignadas por ranking de desempeño (Grupo 1 = Mayor Aprobación):")
    for original, label in rank_mapping.items():
        print(f"  Cluster Original {original} -> {label}")

    # Guardar gráfico de perfiles (Radar o Barras)
    plt.figure(figsize=(12, 6))
    # Normalizar perfil para visualización (0-1) para comparar variables dispares
    profile_norm = (profile[cluster_features] - profile[cluster_features].min()) / (profile[cluster_features].max() - profile[cluster_features].min())
    sns.heatmap(profile_norm, annot=True, cmap='RdYlGn', fmt='.2f')
    plt.title('Mapa de Calor de Perfiles Académicos (Clusters)')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/cluster_profiles_heatmap.png")
    plt.close()

    return df

def analyze_socioeconomic_links(df):
    """
    OBJETIVO 2: "¿En qué medida estos perfiles están vinculados a condiciones socioeconómicas?"
    
    Variables a analizar:
    - Entorno Personal: Gender, Age, Marital status, Displaced, Educational special needs, International
    - Entorno Económico: Scholarship holder, Debtor, Tuition fees up to date
    - Entorno Macro: Unemployment rate, Inflation rate, GDP
    - Entorno Familiar/Educativo: Mother's/Father's qualification/occupation (Omitidas en gráfico simple por alta cardinalidad, pero incluidas en análisis)
    """
    print(f"\n{'='*70}")
    print("📊 OBJETIVO 2: VÍNCULOS SOCIOECONÓMICOS (Chi-Cuadrado y ANOVA)")
    print(f"{'='*70}")
    
    # 1. Variables Categóricas (Nominales)
    # Seleccionamos las que tienen un impacto social/económico directo y manejable visualmente
    socio_vars_cat = [
        'Scholarship holder', 
        'Debtor', 
        'Tuition fees up to date', 
        'Gender', 
        'Marital status',
        'Displaced',                  # Importante: Gasto de vivienda
        'Daytime/evening attendance', # Importante: Estudiante trabajador
        'International'               # Importante: Contexto cultural/económico
    ]
    
    # Lista para guardar resultados estadísticos
    stats_results = []
    
    # Agregar mapeos faltantes para visualización limpia
    extra_mappings = {
        'Displaced': {1: 'Desplazado', 0: 'Residente Local'},
        'Daytime/evening attendance': {1: 'Diurno', 0: 'Nocturno'},
        'International': {1: 'Internacional', 0: 'Nacional'}
    }
    # Combinar con los globales
    local_mappings = {**CATEGORICAL_MAPPINGS, **extra_mappings}

    print("\n--- A. Variables Categóricas (Test Chi-Cuadrado) ---")
    
    # Ajustamos grid de gráficos para más variables (ahora son 8 aprox)
    n_vars = len([v for v in socio_vars_cat if v in df.columns])
    rows = (n_vars // 3) + (1 if n_vars % 3 > 0 else 0)
    
    fig, axes = plt.subplots(rows, 3, figsize=(18, rows*5))
    axes = axes.flatten()
    
    plot_idx = 0
    for col in socio_vars_cat:
        if col not in df.columns: continue
            
        contingency = pd.crosstab(df['Cluster_Label'], df[col])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        
        sig = "✅ Significativo" if p < 0.05 else "❌ No significativo"
        print(f"{col:30} | p={p:.2e} | {sig}")
        stats_results.append({'Variable': col, 'Test': 'Chi2', 'p-value': p})
        
        # Visualización
        contingency_pct = contingency.div(contingency.sum(1), axis=0) * 100
        
        # Aplicar mapeos
        if col in local_mappings:
            mapping = local_mappings[col]
            try:
                new_cols = [mapping.get(int(c), c) for c in contingency_pct.columns]
                contingency_pct.columns = new_cols
            except: pass

        if plot_idx < len(axes):
            ax = axes[plot_idx]
            contingency_pct.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', alpha=0.9)
            ax.set_title(f'{col}\n(p={p:.1e})', fontsize=10)
            ax.set_xlabel('')
            ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=8)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plot_idx += 1
            
    # Ocultar ejes sobrantes
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/socioeconomic_categorical_analysis.png")
    plt.close()
    
    # 2. Variables Numéricas (ANOVA)
    socio_vars_num = ['Age at enrollment', 'Unemployment rate', 'Inflation rate', 'GDP']
    
    print("\n--- B. Variables Numéricas (ANOVA One-Way y Levene) ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(socio_vars_num):
        if col not in df.columns: continue
            
        groups = [df[df['Cluster_Label'] == label][col].values for label in sorted(df['Cluster_Label'].unique())]
        
        # 1. Test de Levene (Homogeneidad de Varianzas)
        try:
            stat_levene, p_levene = stats.levene(*groups)
            levene_sig = "⚠️ Heterogéneas (Falla Supuesto)" if p_levene < 0.05 else "✅ Homogéneas (Cumple Supuesto)"
            print(f"   > Test Levene para '{col}': p={p_levene:.4f} -> {levene_sig}")
        except Exception as e:
            print(f"   > Error en Levene para '{col}': {e}")
            stat_levene, p_levene, levene_sig = 0, 1.0, "Error"

        # 2. Test ANOVA
        f_stat, p = stats.f_oneway(*groups)
        
        sig = "✅ Significativo" if p < 0.05 else "❌ No significativo"
        
        # print(f"{col:30} | ANOVA p={p:.2e} ({sig})") # Levene ya imprimió
        
        stats_results.append({
            'Variable': col, 
            'Test': 'ANOVA', 
            'p-value': p,
            'Levene_p': p_levene
        })
        
        ax = axes[i]
        sns.boxplot(x='Cluster_Label', y=col, data=df, ax=ax, palette='Set2')
        # Título enriquecido con ambos resultados
        ax.set_title(f'{col}\nANOVA p={p:.1e} | Levene p={p_levene:.1e}', fontsize=10)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/socioeconomic_numeric_analysis.png")
    plt.close()

    return pd.DataFrame(stats_results)

def analyze_dropout_probability(df):
    """
    OBJETIVO 3: "¿...y a la probabilidad de permanencia o abandono?"
    
    Calcula:
    1. Tasa de Deserción por Cluster.
    2. Riesgo Relativo (comparado con el mejor cluster).
    """
    print(f"\n{'='*70}")
    print("🎯 OBJETIVO 3: PROBABILIDAD DE ABANDONO Y ÉXITO")
    print(f"{'='*70}")
    
    # Tabla cruzada básica
    # Mapear Target a numérico para facilitar cálculos (Dropout=1, Graduate=0)
    df['Escenario_Fallo'] = np.where(df['Target'] == 'Dropout', 1, 0)
    
    pivot = df.pivot_table(
        index='Cluster_Label', 
        columns='Target', 
        aggfunc='size', 
        fill_value=0
    )
    
    # Calcular Tasas
    pivot['Total'] = pivot.sum(axis=1)
    pivot['Tasa_Desercion'] = pivot['Dropout'] / pivot['Total']
    pivot['Tasa_Graduacion'] = pivot['Graduate'] / pivot['Total']
    
    # Calcular Odds Ratio / Riesgo Relativo
    # Tomamos el "Grupo 1" (el mejor clasificadas) como base (Riesgo = 1.0)
    baseline_dropout_rate = pivot.loc['Grupo 1', 'Tasa_Desercion'] if 'Grupo 1' in pivot.index else 0.01
    # Evitar división por cero
    baseline_dropout_rate = max(baseline_dropout_rate, 0.01)
    
    pivot['Riesgo_Relativo'] = pivot['Tasa_Desercion'] / baseline_dropout_rate
    
    # Mostrar resultados numéricos
    print("\nImpacto del Perfil Académico en el Resultado Final:")
    print("-" * 75)
    print(f"{'Grupo':<15} | {'Dropout':<8} {'Graduate':<8} | {'% Deserción':<12} | {'Riesgo Relativo'}")
    print("-" * 75)
    
    for idx, row in pivot.iterrows():
        risk_str = f"{row['Riesgo_Relativo']:.1f}x"
        print(f"{idx:<15} | {row['Dropout']:<8} {row['Graduate']:<8} | {row['Tasa_Desercion']*100:5.1f}%      | {risk_str}")
        
    # Visualización Impactante
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar Graduados vs Abandonos
    colors = ['#e74c3c', '#2ecc71'] # Rojo Dropout, Verde Graduate
    pivot[['Tasa_Desercion', 'Tasa_Graduacion']] .plot(
        kind='bar', stacked=True, color=colors, ax=ax, width=0.7
    )
    
    # Añadir etiquetas de valor
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', label_type='center', color='white', fontweight='bold')
    
    ax.set_title('Desenlace Final por Perfil Académico Identificado', fontsize=14, pad=20)
    ax.set_ylabel('Proporción de Estudiantes')
    ax.set_xlabel('')
    ax.legend(['Abandono (Dropout)', 'Graduación (Graduate)'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/impacto_desercion_final.png")
    plt.close()
    
    print(f"\n✓ Gráfico generado: {FIGURES_DIR}/impacto_desercion_final.png")
    
    return pivot

def generate_report(df, socio_stats, dropout_stats):
    """
    Consolidación de resultados y guardado de conclusiones.
    """
    print(f"\n{'='*70}")
    print("📝 GENERANDO REPORTE FINAL")
    print(f"{'='*70}")
    
    report_path = f"{OUTPUT_DIR}/conclusiones_clustering.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE HALLAZGOS - CLUSTERING & ÉXITO ACADÉMICO\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. ESTRUCTURA DE DESEMPEÑO (Objetivo 1)\n")
        f.write("-" * 40 + "\n")
        counts = df['Cluster_Label'].value_counts()
        f.write(f"Se identificaron {len(counts)} grupos diferenciados de estudiantes.\n")
        for label in sorted(counts.index):
             n = counts[label]
             pct = (n / len(df)) * 100
             f.write(f"  - {label}: {n} estudiantes ({pct:.1f}%)\n")
        
        f.write("\n2. IMPACTO EN PERMANENCIA (Objetivo 3)\n")
        f.write("-" * 40 + "\n")
        highest_risk_group = dropout_stats['Tasa_Desercion'].idxmax()
        highest_risk_val = dropout_stats['Tasa_Desercion'].max()
        
        lowest_risk_group = dropout_stats['Tasa_Desercion'].idxmin()
        lowest_risk_val = dropout_stats['Tasa_Desercion'].min()
        
        f.write(f"  - El grupo de mayor riesgo es '{highest_risk_group}' con {highest_risk_val*100:.1f}% de deserción.\n")
        f.write(f"  - El perfil más seguro es '{lowest_risk_group}' con solo {lowest_risk_val*100:.1f}% de deserción.\n")
        f.write(f"  - Un estudiante en el grupo de riesgo tiene {dropout_stats.loc[highest_risk_group, 'Riesgo_Relativo']:.1f} veces más probabilidad de abandonar que uno del mejor grupo.\n")

        f.write("\n3. VÍNCULOS SOCIOECONÓMICOS SIGNIFICATIVOS (Objetivo 2)\n")
        f.write("-" * 40 + "\n")
        sig_vars = socio_stats[socio_stats['p-value'] < 0.05]['Variable'].tolist()
        f.write(f"Se encontraron diferencias estadísticamente significativas (p<0.05) en:\n")
        for var in sig_vars:
            f.write(f"  * {var}\n")

    print(f"✓ Reporte guardado en: {report_path}")

def main():
    """Pipeline Principal de Ejecución"""
    create_directories()
    
    # 1. Carga
    df = load_and_prepare_data(DATA_PATH)
    if df is None: return

    # 2. Objetivo 1: Clustering
    df_clustered = perform_clustering_analysis(df)
    
    # 3. Objetivo 2: Socioeconómico
    socio_stats = analyze_socioeconomic_links(df_clustered)
    
    # 4. Objetivo 3: Probabilidad/Impacto
    dropout_stats = analyze_dropout_probability(df_clustered)

    # 5. Generar Conclusiones
    generate_report(df_clustered, socio_stats, dropout_stats)    

if __name__ == "__main__":
    main()


