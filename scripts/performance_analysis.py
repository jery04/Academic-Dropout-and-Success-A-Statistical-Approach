"""
Performance Analysis Script - Análisis de Rendimiento Académico
=================================================================
Este script realiza el análisis comparativo de rendimiento académico 
entre estudiantes que abandonaron (Dropout) vs los que se graduaron (Graduate).

Pregunta de investigación:
¿Existe diferencia significativa en el rendimiento académico entre 
estudiantes que abandonaron y los que completaron sus estudios?

Proceso:
1. Identificación de grupos (abandono / no abandono)
2. Creación de variable de rendimiento (tasa de aprobación)
3. Análisis exploratorio (visualizaciones)
4. Pruebas de normalidad
5. Prueba estadística comparativa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, mannwhitneyu, ttest_ind, levene
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Directorios de salida
OUTPUT_DIR = "outputs/performance_analysis"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"

def create_output_directories():
    """Crear directorios de salida si no existen."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"✓ Directorios de salida creados: {OUTPUT_DIR}")

def load_data(filepath):
    """Cargar el dataset preparado."""
    print(f"\n{'='*60}")
    print("1️⃣  CARGA DE DATOS")
    print('='*60)
    
    df = pd.read_csv(filepath)
    print(f"✓ Dataset cargado: {filepath}")
    print(f"  - Filas: {len(df):,}")
    print(f"  - Columnas: {len(df.columns)}")
    
    return df

def identify_groups(df):
    """
    Identificar y separar los grupos de abandono y no abandono.
    
    Grupos:
    - grupo_abandono: Target = 'Dropout'
    - grupo_no_abandono: Target = 'Graduate'
    """
    print(f"\n{'='*60}")
    print("2️⃣  IDENTIFICACIÓN DE GRUPOS")
    print('='*60)
    
    # Verificar valores únicos en Target
    print(f"\nValores únicos en 'Target': {df['Target'].unique()}")
    print(f"\nDistribución de Target:")
    print(df['Target'].value_counts())
    
    # Separar grupos
    grupo_abandono = df[df['Target'] == 'Dropout'].copy()
    grupo_no_abandono = df[df['Target'] == 'Graduate'].copy()
    
    # También identificamos "Enrolled" si existe (estudiantes aún activos)
    if 'Enrolled' in df['Target'].values:
        grupo_enrolled = df[df['Target'] == 'Enrolled'].copy()
        print(f"\n⚠️  Nota: Se encontraron {len(grupo_enrolled)} estudiantes 'Enrolled' (aún activos)")
        print("    Estos NO se incluyen en el análisis comparativo Dropout vs Graduate")
    
    print(f"\n📊 Resumen de grupos para análisis:")
    print(f"  • Grupo ABANDONO (Dropout):      {len(grupo_abandono):,} estudiantes ({len(grupo_abandono)/len(df)*100:.1f}%)")
    print(f"  • Grupo NO ABANDONO (Graduate):  {len(grupo_no_abandono):,} estudiantes ({len(grupo_no_abandono)/len(df)*100:.1f}%)")
    
    return grupo_abandono, grupo_no_abandono

def create_performance_variable(df, grupo_abandono, grupo_no_abandono):
    """
    Crear la variable de rendimiento: Tasa de Aprobación del 1er semestre.
    
    Fórmula:
    Tasa_aprobacion = Curricular units 1st sem (approved) / Curricular units 1st sem (enrolled)
    
    Manejo de casos especiales:
    - Si enrolled = 0, la tasa se asigna como 0
    """
    print(f"\n{'='*60}")
    print("3️⃣  CREACIÓN DE VARIABLE DE RENDIMIENTO")
    print('='*60)
    
    # Columnas de interés
    col_approved = 'Curricular units 1st sem (approved)'
    col_enrolled = 'Curricular units 1st sem (enrolled)'
    col_evaluations = 'Curricular units 1st sem (evaluations)'
    col_grade = 'Curricular units 1st sem (grade)'
    
    # Verificar que las columnas existen
    print(f"\n📋 Columnas utilizadas:")
    print(f"  - Aprobadas: '{col_approved}'")
    print(f"  - Matriculadas: '{col_enrolled}'")
    
    # Crear tasa de aprobación para todo el dataframe
    df['Tasa_aprobacion_1sem'] = np.where(
        df[col_enrolled] > 0,
        df[col_approved] / df[col_enrolled],
        0  # Si no hay unidades matriculadas, tasa = 0
    )
    
    # Crear tasa para 2do semestre también
    col_approved_2 = 'Curricular units 2nd sem (approved)'
    col_enrolled_2 = 'Curricular units 2nd sem (enrolled)'
    
    df['Tasa_aprobacion_2sem'] = np.where(
        df[col_enrolled_2] > 0,
        df[col_approved_2] / df[col_enrolled_2],
        0
    )
    
    # Tasa promedio de ambos semestres
    df['Tasa_aprobacion_promedio'] = (df['Tasa_aprobacion_1sem'] + df['Tasa_aprobacion_2sem']) / 2
    
    # Actualizar los grupos con las nuevas columnas
    grupo_abandono = df[df['Target'] == 'Dropout'].copy()
    grupo_no_abandono = df[df['Target'] == 'Graduate'].copy()
    
    print(f"\n📊 Estadísticas de Tasa de Aprobación (1er semestre):")
    print("\n  GRUPO ABANDONO (Dropout):")
    print(f"    - Media:     {grupo_abandono['Tasa_aprobacion_1sem'].mean():.4f}")
    print(f"    - Mediana:   {grupo_abandono['Tasa_aprobacion_1sem'].median():.4f}")
    print(f"    - Desv.Std:  {grupo_abandono['Tasa_aprobacion_1sem'].std():.4f}")
    print(f"    - Mín:       {grupo_abandono['Tasa_aprobacion_1sem'].min():.4f}")
    print(f"    - Máx:       {grupo_abandono['Tasa_aprobacion_1sem'].max():.4f}")
    
    print("\n  GRUPO NO ABANDONO (Graduate):")
    print(f"    - Media:     {grupo_no_abandono['Tasa_aprobacion_1sem'].mean():.4f}")
    print(f"    - Mediana:   {grupo_no_abandono['Tasa_aprobacion_1sem'].median():.4f}")
    print(f"    - Desv.Std:  {grupo_no_abandono['Tasa_aprobacion_1sem'].std():.4f}")
    print(f"    - Mín:       {grupo_no_abandono['Tasa_aprobacion_1sem'].min():.4f}")
    print(f"    - Máx:       {grupo_no_abandono['Tasa_aprobacion_1sem'].max():.4f}")
    
    # Casos con tasa = 0 (sin unidades matriculadas o sin aprobar ninguna)
    casos_cero_abandono = (grupo_abandono['Tasa_aprobacion_1sem'] == 0).sum()
    casos_cero_no_abandono = (grupo_no_abandono['Tasa_aprobacion_1sem'] == 0).sum()
    
    print(f"\n⚠️  Casos con Tasa = 0:")
    print(f"    - Dropout:  {casos_cero_abandono} ({casos_cero_abandono/len(grupo_abandono)*100:.1f}%)")
    print(f"    - Graduate: {casos_cero_no_abandono} ({casos_cero_no_abandono/len(grupo_no_abandono)*100:.1f}%)")
    
    return df, grupo_abandono, grupo_no_abandono

def exploratory_analysis(grupo_abandono, grupo_no_abandono):
    """
    Análisis exploratorio con visualizaciones.
    """
    print(f"\n{'='*60}")
    print("4️⃣  ANÁLISIS EXPLORATORIO - VISUALIZACIONES")
    print('='*60)
    
    variable = 'Tasa_aprobacion_1sem'
    
    # Figura: Boxplot e Histograma superpuesto
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Boxplot
    ax1 = axes[0]
    data_boxplot = [grupo_abandono[variable], grupo_no_abandono[variable]]
    bp = ax1.boxplot(data_boxplot, labels=['Dropout\n(Abandono)', 'Graduate\n(No Abandono)'],
                     patch_artist=True)
    colors = ['#e74c3c', '#27ae60']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('Tasa de Aprobación', fontsize=12)
    ax1.set_title('Comparación de Tasa de Aprobación\n(1er Semestre)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Añadir medias al boxplot
    means = [grupo_abandono[variable].mean(), grupo_no_abandono[variable].mean()]
    ax1.scatter([1, 2], means, color='white', s=100, zorder=5, edgecolor='black', marker='D')
    for i, mean in enumerate(means):
        ax1.annotate(f'μ={mean:.3f}', (i+1, mean), textcoords="offset points", 
                     xytext=(30, 0), fontsize=10, ha='left')
    
    # Subplot 2: Histogramas superpuestos
    ax2 = axes[1]
    ax2.hist(grupo_abandono[variable], bins=20, alpha=0.6, label='Dropout', 
             color='#e74c3c', edgecolor='white', density=True)
    ax2.hist(grupo_no_abandono[variable], bins=20, alpha=0.6, label='Graduate', 
             color='#27ae60', edgecolor='white', density=True)
    ax2.set_xlabel('Tasa de Aprobación', fontsize=12)
    ax2.set_ylabel('Densidad', fontsize=12)
    ax2.set_title('Distribución de Tasa de Aprobación\nHistogramas Superpuestos', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/boxplot_histograma.png', dpi=150, bbox_inches='tight')
    print(f"✓ Figura guardada: {FIGURES_DIR}/boxplot_histograma.png")
    plt.show()
    
    return

def normality_tests(grupo_abandono, grupo_no_abandono):
    """
    Realizar pruebas de normalidad para decidir qué prueba estadística usar.
    
    Prueba de Shapiro-Wilk:
    - H0: Los datos siguen una distribución normal
    - H1: Los datos NO siguen una distribución normal
    - Si p-value < 0.05: Rechazamos H0 → NO es normal
    """
    print(f"\n{'='*60}")
    print("5️⃣  PRUEBAS DE NORMALIDAD (Shapiro-Wilk)")
    print('='*60)
    
    variable = 'Tasa_aprobacion_1sem'
    alpha = 0.05
    
    results = {}
    
    # Limitar muestra para Shapiro-Wilk (máximo 5000)
    n_max = 5000
    
    print(f"\n📊 Hipótesis:")
    print(f"   H0: Los datos siguen una distribución normal")
    print(f"   H1: Los datos NO siguen una distribución normal")
    print(f"   α = {alpha}")
    
    # Test para grupo abandono
    sample_abandono = grupo_abandono[variable]
    if len(sample_abandono) > n_max:
        sample_abandono = sample_abandono.sample(n_max, random_state=42)
        print(f"\n⚠️  Muestra Dropout reducida a {n_max} para Shapiro-Wilk")
    
    stat_abandono, p_abandono = shapiro(sample_abandono)
    normal_abandono = p_abandono > alpha
    
    # Test para grupo no abandono
    sample_no_abandono = grupo_no_abandono[variable]
    if len(sample_no_abandono) > n_max:
        sample_no_abandono = sample_no_abandono.sample(n_max, random_state=42)
        print(f"⚠️  Muestra Graduate reducida a {n_max} para Shapiro-Wilk")
    
    stat_no_abandono, p_no_abandono = shapiro(sample_no_abandono)
    normal_no_abandono = p_no_abandono > alpha
    
    print(f"\n📈 Resultados Shapiro-Wilk:")
    print(f"\n   GRUPO DROPOUT (Abandono):")
    print(f"   - Estadístico W: {stat_abandono:.6f}")
    print(f"   - p-value:       {p_abandono:.2e}")
    print(f"   - Conclusión:    {'✓ Distribución NORMAL' if normal_abandono else '✗ Distribución NO normal'}")
    
    print(f"\n   GRUPO GRADUATE (No Abandono):")
    print(f"   - Estadístico W: {stat_no_abandono:.6f}")
    print(f"   - p-value:       {p_no_abandono:.2e}")
    print(f"   - Conclusión:    {'✓ Distribución NORMAL' if normal_no_abandono else '✗ Distribución NO normal'}")
    
    # Decisión sobre prueba a usar
    both_normal = normal_abandono and normal_no_abandono
    
    print(f"\n{'─'*60}")
    print(f"📋 DECISIÓN PARA PRUEBA ESTADÍSTICA:")
    if both_normal:
        print(f"   ✓ Ambos grupos tienen distribución NORMAL")
        print(f"   → Usar prueba paramétrica: t-test de Student")
    else:
        print(f"   ✗ Al menos un grupo NO tiene distribución normal")
        print(f"   → Usar prueba NO paramétrica: Mann-Whitney U")
    print(f"{'─'*60}")
    
    results = {
        'shapiro_dropout': {
            'statistic': float(stat_abandono),
            'p_value': float(p_abandono),
            'is_normal': True if normal_abandono else False
        },
        'shapiro_graduate': {
            'statistic': float(stat_no_abandono),
            'p_value': float(p_no_abandono),
            'is_normal': True if normal_no_abandono else False
        },
        'both_normal': True if both_normal else False
    }
    
    return results, both_normal

def statistical_comparison(grupo_abandono, grupo_no_abandono, both_normal):
    """
    Realizar la prueba estadística para comparar los grupos:
    H0: No hay diferencia significativa en el rendimiento entre grupos
    H1: Existe diferencia significativa en el rendimiento entre grupos
    
    Si ambos son normales: t-test de Student (paramétrico)
    Si no son normales: Mann-Whitney U (no paramétrico)
    """
    print(f"\n{'='*60}")
    print("6️⃣  PRUEBA ESTADÍSTICA COMPARATIVA")
    print('='*60)
    
    variable = 'Tasa_aprobacion_1sem'
    alpha = 0.05
    
    results = {}
    
    data_abandono = grupo_abandono[variable]
    data_no_abandono = grupo_no_abandono[variable]
    
    print(f"\n📊 Hipótesis:")
    print(f"   H0: No hay diferencia significativa en el rendimiento entre grupos")
    print(f"   H1: Existe diferencia significativa en el rendimiento entre grupos")
    print(f"   α = {alpha}")
    
    if both_normal:
        # Prueba t-test
        print(f"\n🔬 Prueba seleccionada: t-test de Student (paramétrica)")
        
        # Primero verificar homogeneidad de varianzas con Levene
        stat_levene, p_levene = levene(data_abandono, data_no_abandono)
        equal_var = p_levene > alpha
        
        print(f"\n   Prueba de Levene (homogeneidad de varianzas):")
        print(f"   - Estadístico: {stat_levene:.4f}")
        print(f"   - p-value:     {p_levene:.2e}")
        print(f"   - Varianzas:   {'Homogéneas' if equal_var else 'No homogéneas'}")
        
        # t-test
        stat_test, p_value = ttest_ind(data_abandono, data_no_abandono, equal_var=equal_var)
        test_name = "t-test de Student" if equal_var else "t-test de Welch"
        
        results['levene'] = {
            'statistic': float(stat_levene),
            'p_value': float(p_levene),
            'equal_variance': True if equal_var else False
        }
        
    else:
        # Prueba Mann-Whitney U
        print(f"\n🔬 Prueba seleccionada: Mann-Whitney U (no paramétrica)")
        stat_test, p_value = mannwhitneyu(data_abandono, data_no_abandono, alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    # Calcular tamaño del efecto (Cohen's d)
    cohens_d = (data_no_abandono.mean() - data_abandono.mean()) / np.sqrt(
        ((len(data_abandono) - 1) * data_abandono.std()**2 + 
         (len(data_no_abandono) - 1) * data_no_abandono.std()**2) / 
        (len(data_abandono) + len(data_no_abandono) - 2)
    )
    
    # Interpretar tamaño del efecto
    if abs(cohens_d) < 0.2:
        effect_interpretation = "pequeño"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "pequeño a mediano"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "mediano"
    else:
        effect_interpretation = "grande"
    
    significant = p_value < alpha
    
    print(f"\n{'─'*60}")
    print(f"📈 RESULTADOS DE {test_name.upper()}")
    print(f"{'─'*60}")
    print(f"\n   Estadístico:     {stat_test:.4f}")
    print(f"   p-value:         {p_value:.2e}")
    print(f"   Cohen's d:       {cohens_d:.4f} ({effect_interpretation})")
    print(f"\n   Media Dropout:   {data_abandono.mean():.4f}")
    print(f"   Media Graduate:  {data_no_abandono.mean():.4f}")
    print(f"   Diferencia:      {data_no_abandono.mean() - data_abandono.mean():.4f}")
    
    print(f"\n{'─'*60}")
    print(f"📋 CONCLUSIÓN:")
    print(f"{'─'*60}")
    
    if significant:
        print(f"\n   ✓ RESULTADO SIGNIFICATIVO (p-value < {alpha})")
        print(f"\n   Se RECHAZA la hipótesis nula (H0).")
        print(f"\n   INTERPRETACIÓN:")
        print(f"   Existe evidencia estadística suficiente para afirmar que HAY")
        print(f"   una diferencia significativa en la tasa de aprobación entre")
        print(f"   estudiantes que abandonaron y los que se graduaron.")
        print(f"\n   Los estudiantes graduados tienen una tasa de aprobación")
        print(f"   significativamente MAYOR ({data_no_abandono.mean():.2%}) que los")
        print(f"   estudiantes que abandonaron ({data_abandono.mean():.2%}).")
        print(f"\n   El tamaño del efecto es {effect_interpretation} (d = {cohens_d:.3f}).")
    else:
        print(f"\n   ✗ RESULTADO NO SIGNIFICATIVO (p-value >= {alpha})")
        print(f"\n   NO se rechaza la hipótesis nula (H0).")
        print(f"\n   No hay evidencia suficiente para afirmar que existe")
        print(f"   diferencia en el rendimiento académico entre los grupos.")
    
    results['test'] = {
        'name': test_name,
        'statistic': float(stat_test),
        'p_value': float(p_value),
        'significant': True if significant else False,
        'cohens_d': float(cohens_d),
        'effect_size': effect_interpretation,
        'alpha': alpha,
        'mean_dropout': float(data_abandono.mean()),
        'mean_graduate': float(data_no_abandono.mean()),
        'difference': float(data_no_abandono.mean() - data_abandono.mean())
    }
    
    return results

def generate_report(df, grupo_abandono, grupo_no_abandono, normality_results, comparison_results):
    """
    Generar reporte completo del análisis.
    """
    print(f"\n{'='*60}")
    print("7️⃣  GENERACIÓN DE REPORTE")
    print('='*60)
    
    variable = 'Tasa_aprobacion_1sem'
    
    report = {
        'metadata': {
            'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': 'outputs/prepared_data/dataset_prepared.csv',
            'variable_analizada': 'Tasa de Aprobación (1er Semestre)',
            'n_total': len(df)
        },
        'grupos': {
            'dropout': {
                'n': len(grupo_abandono),
                'porcentaje': round(len(grupo_abandono) / len(df) * 100, 2),
                'estadisticas': {
                    'media': round(grupo_abandono[variable].mean(), 4),
                    'mediana': round(grupo_abandono[variable].median(), 4),
                    'desv_std': round(grupo_abandono[variable].std(), 4),
                    'min': round(grupo_abandono[variable].min(), 4),
                    'max': round(grupo_abandono[variable].max(), 4),
                    'q1': round(grupo_abandono[variable].quantile(0.25), 4),
                    'q3': round(grupo_abandono[variable].quantile(0.75), 4)
                }
            },
            'graduate': {
                'n': len(grupo_no_abandono),
                'porcentaje': round(len(grupo_no_abandono) / len(df) * 100, 2),
                'estadisticas': {
                    'media': round(grupo_no_abandono[variable].mean(), 4),
                    'mediana': round(grupo_no_abandono[variable].median(), 4),
                    'desv_std': round(grupo_no_abandono[variable].std(), 4),
                    'min': round(grupo_no_abandono[variable].min(), 4),
                    'max': round(grupo_no_abandono[variable].max(), 4),
                    'q1': round(grupo_no_abandono[variable].quantile(0.25), 4),
                    'q3': round(grupo_no_abandono[variable].quantile(0.75), 4)
                }
            }
        },
        'pruebas_normalidad': normality_results,
        'prueba_comparativa': comparison_results
    }
    
    # Guardar reporte JSON
    report_path = f'{OUTPUT_DIR}/analysis_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ Reporte JSON guardado: {report_path}")
    
    return report

def main():
    """Función principal del análisis."""
    print("\n" + "="*70)
    print("   ANÁLISIS DE RENDIMIENTO ACADÉMICO: ABANDONO vs GRADUACIÓN")
    print("   Pregunta: ¿Existe diferencia significativa en el rendimiento?")
    print("="*70)
    
    # Crear directorios
    create_output_directories()
    
    # 1. Cargar datos
    df = load_data('outputs/prepared_data/dataset_prepared.csv')
    
    # 2. Identificar grupos
    grupo_abandono, grupo_no_abandono = identify_groups(df)
    
    # 3. Crear variable de rendimiento
    df, grupo_abandono, grupo_no_abandono = create_performance_variable(
        df, grupo_abandono, grupo_no_abandono
    )
    
    # 4. Análisis exploratorio
    exploratory_analysis(grupo_abandono, grupo_no_abandono)
    
    # 5. Pruebas de normalidad
    normality_results, both_normal = normality_tests(grupo_abandono, grupo_no_abandono)
    
    # 6. Prueba estadística comparativa
    comparison_results = statistical_comparison(grupo_abandono, grupo_no_abandono, both_normal)
    
    # 7. Generar reporte
    report = generate_report(df, grupo_abandono, grupo_no_abandono, normality_results, comparison_results)
    
    # Resumen final
    print(f"\n{'='*70}")
    print("   ✅ ANÁLISIS COMPLETADO")
    print('='*70)
    print(f"\n📁 Archivos generados en: {OUTPUT_DIR}/")
    print(f"   • analysis_report.json")
    print(f"   • figures/boxplot_histograma.png")
    print(f"\n{'='*70}\n")
    
    return df, report

if __name__ == "__main__":
    df_result, report_result = main()
