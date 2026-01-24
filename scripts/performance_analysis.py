"""
Este script realiza el análisis comparativo de rendimiento académico 
entre estudiantes que abandonaron (Dropout) vs los que se graduaron (Graduate).

Pregunta de investigación:
¿Tienden los estudiantes que se graduaron a mostrar valores de 
rendimiento mayores que los estudiantes que abandonaron la carrera?

Proceso:
1. Identificación de grupos (abandono / no abandono)
2. Análisis exploratorio para ambos grupos (visualizaciones)
3. Pruebas de normalidad
4. Prueba estadística comparativa
"""
import pandas as pd              # Manipulación y análisis de datos tabulares
import numpy as np               # Operaciones numéricas y cálculos matemáticos
import matplotlib.pyplot as plt  # Creación de gráficos y visualizaciones
import seaborn as sns            # Visualizaciones estadísticas mejoradas
from scipy import stats          # Funciones estadísticas generales
from scipy.stats import shapiro, mannwhitneyu, ttest_ind, levene # Importaciones específicas para pruebas estadísticas
import warnings                  # Manejo de advertencias
import os                        # Operaciones del sistema de archivos
import json                      # Lectura/escritura de archivos JSON
from datetime import datetime    # Manejo de fechas

warnings.filterwarnings('ignore')          # Suprimir advertencias para una salida más limpia
plt.style.use('seaborn-v0_8-whitegrid')    # Estilo de gráficos: fondo blanco con cuadrícula suave
sns.set_palette("husl")                    # Paleta de colores diversa para mejor distinción visual
OUTPUT_DIR = "outputs/performance_analysis"    # Directorio de salida para resultados
FIGURES_DIR = f"{OUTPUT_DIR}/figures"          # Directorio para guardar figuras generadas

def create_output_directories():
    """Crear directorios de salida si no existen."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"✓ Directorios de salida creados: {OUTPUT_DIR}")

def remove_outliers_from_groups(grupo_abandono, grupo_no_abandono, variable='Tasa_aprobacion_1sem', k=3):
    """
    Remueve outliers usando IQR clásico.

    - Usa factor k por defecto 3 con sensibilidad baja.
    - Añade columna 'Outlier_Tasa_1sem' (1 = outlier, 0 = no outlier) y
      devuelve los dataframes filtrados (sin outliers) y el conteo removido.

    Retorna: (grupo_abandono_filtrado, grupo_no_abandono_filtrado, removed_counts)
    """
    # Verificar existencia de la variable
    if variable not in grupo_abandono.columns or variable not in grupo_no_abandono.columns:
        return grupo_abandono, grupo_no_abandono, {'abandono': 0, 'graduate': 0}

    def mark_and_filter(df):
        s = df[variable].dropna()
        if s.empty:
            df['Outlier_Tasa_1sem'] = 0
            return df

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        # Si IQR == 0 (datos constantes) -> no marcar outliers
        if iqr == 0 or np.isclose(iqr, 0):
            df['Outlier_Tasa_1sem'] = 0
            return df

        lower_fence = q1 - k * iqr
        upper_fence = q3 + k * iqr

        df['Outlier_Tasa_1sem'] = df[variable].apply(
            lambda x: 1 if (pd.notna(x) and (x < lower_fence or x > upper_fence)) else 0
        )
        return df

    ga_marked = mark_and_filter(grupo_abandono.copy())
    gg_marked = mark_and_filter(grupo_no_abandono.copy())

    ga = ga_marked[ga_marked['Outlier_Tasa_1sem'] == 0].copy()
    gg = gg_marked[gg_marked['Outlier_Tasa_1sem'] == 0].copy()

    removed = {
        'abandono': int(ga_marked['Outlier_Tasa_1sem'].sum()),
        'graduate': int(gg_marked['Outlier_Tasa_1sem'].sum())
    }

    return ga, gg, removed

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
    # Mostrar total considerado (solo Dropout + Graduate) y % dentro de ese total
    total_two_groups = len(grupo_abandono) + len(grupo_no_abandono)
    print(f"\n🔢 Total considerado (Dropout + Graduate): {total_two_groups:,}")
    if total_two_groups > 0:
        pct_ab = len(grupo_abandono) / total_two_groups * 100
        pct_gr = len(grupo_no_abandono) / total_two_groups * 100
        print(f"  • % sobre Total (Dropout):   {pct_ab:.1f}%")
        print(f"  • % sobre Total (Graduate):  {pct_gr:.1f}%")
    
    return grupo_abandono, grupo_no_abandono

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
    # Mostrar la figura en pantalla al ejecutar el script
    try:
        plt.show()
    except Exception:
        # En entornos no interactivos, plt.show() puede fallar; seguimos guardando la figura
        pass
    plt.close()
    
    return

def normality_tests(grupo_abandono, grupo_no_abandono):
    """
    Realizar pruebas de normalidad para decidir qué prueba estadística usar.
    
    Importancia:
    Esta prueba determina si podemos usar pruebas paramétricas (como t-test)
    que asumen normalidad, o debemos usar pruebas no paramétricas (Mann-Whitney).
    
    Prueba de Shapiro-Wilk:
    - Es una de las pruebas de normalidad más potentes para muestras pequeñas
    - H0: Los datos siguen una distribución normal
    - H1: Los datos NO siguen una distribución normal
    - Si p-value < 0.05: Rechazamos H0 → distribución NO es normal
    - Si p-value >= 0.05: No rechazamos H0 → podemos asumir normalidad
    
    Limitación:
    - Máximo 5000 observaciones, por lo que se usa muestreo si es necesario
    """
    print(f"\n{'='*60}")
    print("5️⃣  PRUEBAS DE NORMALIDAD (Shapiro-Wilk)")
    print('='*60)
    
    variable = 'Tasa_aprobacion_1sem'
    alpha = 0.05  # Nivel de significancia estándar

    results = {}

    # Limitar muestra para Shapiro-Wilk (máximo 5000)
    n_max = 5000

    print(f"\n📊 Hipótesis:")
    print(f"   H0: Los datos siguen una distribución normal")
    print(f"   H1: Los datos NO siguen una distribución normal")
    print(f"   α = {alpha}")

    # Quitar outliers marcados antes de las pruebas (Shapiro es sensible a outliers)
    # Usamos la versión sencilla IQR (k=1.5)
    grupo_filtrado_a, grupo_filtrado_g, removed = remove_outliers_from_groups(
        grupo_abandono, grupo_no_abandono
    )
    if removed['abandono'] or removed['graduate']:
        print(f"\n⚠️ Outliers removidos antes de Shapiro-Wilk:")
        print(f"    - Dropout:  {removed['abandono']}")
        print(f"    - Graduate: {removed['graduate']}")

    # Test para grupo abandono (sin outliers)
    sample_abandono = grupo_filtrado_a[variable]
    if len(sample_abandono) > n_max:
        sample_abandono = sample_abandono.sample(n_max, random_state=42)
        print(f"\n⚠️  Muestra Dropout reducida a {n_max} para Shapiro-Wilk")

    stat_abandono, p_abandono = shapiro(sample_abandono)
    normal_abandono = p_abandono > alpha

    # Test para grupo no abandono (sin outliers)
    sample_no_abandono = grupo_filtrado_g[variable]
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
    # Comparación explícita: mostrar valores numéricos de p-value y alpha
    print(f"   - Comparación: {p_abandono:.2e} < {alpha} -> {'p-value < α: Rechazamos H0 (NO normal)' if p_abandono < alpha else 'p-value >= α: No rechazamos H0 (posible normalidad)'}")
    
    print(f"\n   GRUPO GRADUATE (No Abandono):")
    print(f"   - Estadístico W: {stat_no_abandono:.6f}")
    print(f"   - p-value:       {p_no_abandono:.2e}")
    print(f"   - Conclusión:    {'✓ Distribución NORMAL' if normal_no_abandono else '✗ Distribución NO normal'}")
    # Comparación explícita: mostrar valores numéricos de p-value y alpha
    print(f"   - Comparación: {p_no_abandono:.2e} < {alpha} -> {'p-value < α: Rechazamos H0 (NO normal)' if p_no_abandono < alpha else 'p-value >= α: No rechazamos H0 (posible normalidad)'}")
    
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
    # --- Visualización: Q-Q plots (Shapiro diagnostic) usando datos sin outliers ---
    try:
        # Preparar datos sin NaNs a partir de las muestras usadas en Shapiro
        sa = sample_abandono.dropna()
        sg = sample_no_abandono.dropna()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Dropout Q-Q
        (osm_a, osr_a), (slope_a, intercept_a, r_a) = stats.probplot(sa, dist='norm')
        axes[0].scatter(osm_a, osr_a, facecolors='none', edgecolors='#e74c3c')
        axes[0].plot(osm_a, slope_a * osm_a + intercept_a, color='#2c3e50', linestyle='--')
        axes[0].set_title('Normal Q-Q Plot\nDropout (Abandono)')
        axes[0].set_xlabel('Theoretical Quantiles')
        axes[0].set_ylabel('Ordered Values')

        # Graduate Q-Q
        (osm_g, osr_g), (slope_g, intercept_g, r_g) = stats.probplot(sg, dist='norm')
        axes[1].scatter(osm_g, osr_g, facecolors='none', edgecolors='#27ae60')
        axes[1].plot(osm_g, slope_g * osm_g + intercept_g, color='#2c3e50', linestyle='--')
        axes[1].set_title('Normal Q-Q Plot\nGraduate (No Abandono)')
        axes[1].set_xlabel('Theoretical Quantiles')
        axes[1].set_ylabel('Ordered Values')

        plt.tight_layout()
        qq_path = f'{FIGURES_DIR}/shapiro_qq_plots.png'
        plt.savefig(qq_path, dpi=150, bbox_inches='tight')
        print(f"✓ Q-Q plots guardados: {qq_path}")
        try:
            plt.show()
        except Exception:
            pass
        plt.close()
    except Exception as e:
        print(f"⚠️  No se pudo generar Q-Q plots: {e}")

    return results, both_normal

def statistical_comparison(grupo_abandono, grupo_no_abandono, both_normal):
    """
    Realizar la prueba estadística para comparar los grupos.
    
    Esta es la prueba central del análisis que responde a la pregunta:
    "¿Tiende un grupo a mostrar valores de rendimiento mayores que el otro?"
    
    Hipótesis:
    - H0 (nula): No existe evidencia de que un grupo muestre valores de rendimiento mayores que el otro
    - H1 (alternativa): Un grupo muestra valores de rendimiento mayor que el otro
    
    Selección de prueba:
    - Si ambos son normales: t-test de Student (paramétrico)
      * Más potente cuando se cumplen supuestos
      * Incluye verificación de homogeneidad de varianzas (Levene)
    - Si no son normales: Mann-Whitney U (no paramétrico)
      * Más robusto ante violaciones de normalidad
      * Compara tendencia estocástica
      * Independencia entre observaciones
      * Sensibilidad a outliers baja
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
    print(f"   H0 (nula): No existe evidencia de que un grupo muestre valores de rendimiento mayores que el otro")
    print(f"   H1 (alternativa): Un grupo muestra valores de rendimiento mayor que el otro")
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
        # Explicación breve sobre la elección de Mann-Whitney y sus supuestos
        print(f"\n¿Por qué elegir Mann-Whitney U?")
        print(f" - Se elige cuando al menos uno de los grupos NO cumple la suposición de normalidad.")
        print(f" - Mann-Whitney es una prueba no paramétrica que compara el orden (ranks) de las observaciones entre grupos, es más robusta ante distribuciones no normales y outliers.")
        print(f"\nSupuestos principales de Mann-Whitney:")
        print(f"  1) Observaciones independientes entre y dentro de los grupos.")
        print(f"  2) La variable de respuesta es al menos de escala ordinal (o continua).")
        print(f"  3) Las distribuciones de ambas poblaciones deben tener forma similar si se quiere interpretar la prueba como comparación de medianas. Si las formas difieren, que es el caso, la prueba compara más bien la tendencia estocástica, que es el objetivo.")
        stat_test, p_value = mannwhitneyu(data_abandono, data_no_abandono, alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    # Determinar si el resultado es estadísticamente significativo
    significant = p_value < alpha
    
    print(f"\n{'─'*60}")
    print(f"📈 RESULTADOS DE {test_name.upper()}")
    print(f"{'─'*60}")
    print(f"\n   Estadístico:     {stat_test:.4f}")
    print(f"   p-value:         {p_value:.2e}")
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
        print(f"   Existe evidencia estadística suficiente para afirmar que")
        print(f"   los estudiantes que se graduaron tienden a mostrar valores")
        print(f"   de rendimiento mayores que los estudiantes que abandonaron.")
        print(f"\n   Los estudiantes graduados tienen una tasa de aprobación")
        print(f"   significativamente MAYOR ({data_no_abandono.mean():.2%}) que los")
        print(f"   estudiantes que abandonaron ({data_abandono.mean():.2%}).")
      
        pass
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
            'variable_analizada': 'Tasa de Aprobación (1er Semestre) - zscore',
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

def main(input_path: str = None, output_dir: str = None):
    """
    Función principal del análisis de rendimiento académico.
    
    Objetivo:
    Responder a la pregunta: "¿Existe diferencia significativa en el 
    rendimiento académico entre estudiantes que abandonaron y los que 
    se graduaron?"
    
    Flujo de ejecución:
    1. Cargar datos del dataset preparado
    2. Identificar y separar grupos (Dropout vs Graduate)
    3. Análisis exploratorio para ambos grupos (visualizaciones)
    4. Pruebas de normalidad (determinar tipo de prueba)
    5. Prueba estadística comparativa
    6. Generar reporte con conclusiones
    """
    global OUTPUT_DIR, FIGURES_DIR
    
    # Configurar directorios de salida
    if output_dir is not None:
        OUTPUT_DIR = output_dir
        FIGURES_DIR = f"{OUTPUT_DIR}/figures"
    
    # Configurar ruta de entrada
    if input_path is None:
        input_path = 'outputs/prepared_data/dataset_prepared.csv'
    
    print("\n" + "="*70)
    print("   ANÁLISIS DE RENDIMIENTO ACADÉMICO: ABANDONO vs GRADUACIÓN")
    print("   Pregunta: ¿Existe diferencia significativa en el rendimiento?")
    print("="*70)
    
    # Crear directorios
    create_output_directories()
    
    # 1. Cargar datos
    df = load_data(input_path)
    
    # 2. Identificar grupos
    grupo_abandono, grupo_no_abandono = identify_groups(df)
    
    # 3. Análisis exploratorio para ambos grupos (visualizaciones)
    exploratory_analysis(grupo_abandono, grupo_no_abandono)
    
    # 4. Pruebas de normalidad
    normality_results, both_normal = normality_tests(grupo_abandono, grupo_no_abandono)
    
    # 5. Prueba estadística comparativa
    comparison_results = statistical_comparison(grupo_abandono, grupo_no_abandono, both_normal)
    
    # 6. Generar reporte
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
