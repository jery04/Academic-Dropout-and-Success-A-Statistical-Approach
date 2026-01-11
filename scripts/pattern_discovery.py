"""
Pattern Discovery - Identificación de Patrones Académicos
==========================================================
Este script identifica patrones específicos para estudiantes que abandonan
vs. estudiantes que completan sus estudios.

Objetivo: Encontrar patrones distintivos que caracterizan a cada grupo

Análisis incluidos:
1. Perfiles demográficos y socioeconómicos
2. Patrones de rendimiento académico
3. Patrones de comportamiento de estudio
4. Correlaciones específicas por grupo
5. Clustering para identificar subgrupos
6. Análisis de características discriminantes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR = "outputs/pattern_analysis"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"

# Mapas de etiquetas para variables categóricas (mostrar cualitativo en vez de números)
CATEGORICAL_MAPPINGS = {
    'Marital status': {
        1: 'Soltero/a',
        2: 'Casado/a',
        3: 'Viudo/a',
        4: 'Divorciado/a',
        5: 'Unión de hecho',
        6: 'Separado legalmente'
    },
    'Gender': {
        0: 'Masculino',
        1: 'Femenino'
    },
    'Scholarship holder': {
        0: 'No',
        1: 'Sí'
    },
    'Debtor': {
        0: 'No',
        1: 'Sí'
    },
    'Tuition fees up to date': {
        1: 'Al día',
        0: 'Atrasado'
    },
    'Displaced': {
        0: 'No',
        1: 'Sí'
    },
    'Daytime/evening attendance': {
        1: 'Diurno',
        0: 'Nocturno'
    }
}

def create_output_directories():
    """Crear directorios de salida."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"✓ Directorios creados: {OUTPUT_DIR}")

def load_and_prepare_data(filepath):
    """Cargar y preparar datos."""
    print(f"\n{'='*70}")
    print("📊 CARGA Y PREPARACIÓN DE DATOS")
    print('='*70)
    
    df = pd.read_csv(filepath)
    
    # Filtrar solo Dropout y Graduate (excluir Enrolled)
    df_filtered = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
    
    # Aplicar mapeos cualitativos a variables categóricas para que las salidas
    # muestren etiquetas legibles en lugar de números.
    for col, mapping in CATEGORICAL_MAPPINGS.items():
        if col in df_filtered.columns:
            def _map_val(x):
                try:
                    # convertir a int cuando sea posible (1.0 -> 1)
                    if pd.isna(x):
                        return 'Desconocido'
                    key = int(x)
                    return mapping.get(key, f'Desconocido ({x})')
                except Exception:
                    return f'Desconocido ({x})'

            df_filtered[col] = df_filtered[col].apply(_map_val)
    print(f"\n✓ Dataset cargado: {len(df):,} registros totales")
    print(f"✓ Filtrado a Dropout/Graduate: {len(df_filtered):,} registros")
    print(f"\n  - Dropout:  {len(df_filtered[df_filtered['Target']=='Dropout']):,}")
    print(f"  - Graduate: {len(df_filtered[df_filtered['Target']=='Graduate']):,}")
    
    # Crear variables adicionales de rendimiento
    df_filtered['Tasa_aprobacion_1sem'] = np.where(
        df_filtered['Curricular units 1st sem (enrolled)'] > 0,
        df_filtered['Curricular units 1st sem (approved)'] / df_filtered['Curricular units 1st sem (enrolled)'],
        0
    )
    
    df_filtered['Tasa_aprobacion_2sem'] = np.where(
        df_filtered['Curricular units 2nd sem (enrolled)'] > 0,
        df_filtered['Curricular units 2nd sem (approved)'] / df_filtered['Curricular units 2nd sem (enrolled)'],
        0
    )
    
    df_filtered['Tasa_evaluacion_1sem'] = np.where(
        df_filtered['Curricular units 1st sem (enrolled)'] > 0,
        df_filtered['Curricular units 1st sem (evaluations)'] / df_filtered['Curricular units 1st sem (enrolled)'],
        0
    )
    
    df_filtered['Rendimiento_promedio'] = (
        df_filtered['Tasa_aprobacion_1sem'] + df_filtered['Tasa_aprobacion_2sem']
    ) / 2
    
    return df_filtered

def analyze_demographic_patterns(df):
    """
    Analizar patrones demográficos y socioeconómicos por grupo.
    """
    print(f"\n{'='*70}")
    print("👥 ANÁLISIS 1: PATRONES DEMOGRÁFICOS Y SOCIOECONÓMICOS")
    print('='*70)
    
    patterns = {'dropout': {}, 'graduate': {}}
    
    # Separar grupos
    dropout = df[df['Target'] == 'Dropout']
    graduate = df[df['Target'] == 'Graduate']
    
    # Variables categóricas de interés
    categorical_vars = [
        'Marital status', 'Gender', 'Scholarship holder', 
        'Debtor', 'Tuition fees up to date', 'Displaced',
        'Daytime/evening attendance'
    ]
    
    print("\n" + "─"*70)
    print("PATRONES DEMOGRÁFICOS - DROPOUT")
    print("─"*70)
    
    for var in categorical_vars:
        if var in df.columns:
            # Patrón para Dropout
            dropout_pattern = dropout[var].value_counts(normalize=True).head(3)
            patterns['dropout'][var] = dropout_pattern.to_dict()
            
            print(f"\n{var}:")
            for value, pct in dropout_pattern.items():
                print(f"  • {value}: {pct*100:.1f}%")
    
    # Edad promedio
    patterns['dropout']['edad_promedio'] = float(dropout['Age at enrollment'].mean())
    print(f"\nEdad promedio al matricularse: {dropout['Age at enrollment'].mean():.1f} años")
    
    print("\n" + "─"*70)
    print("PATRONES DEMOGRÁFICOS - GRADUATE")
    print("─"*70)
    
    for var in categorical_vars:
        if var in df.columns:
            # Patrón para Graduate
            graduate_pattern = graduate[var].value_counts(normalize=True).head(3)
            patterns['graduate'][var] = graduate_pattern.to_dict()
            
            print(f"\n{var}:")
            for value, pct in graduate_pattern.items():
                print(f"  • {value}: {pct*100:.1f}%")
    
    # Edad promedio
    patterns['graduate']['edad_promedio'] = float(graduate['Age at enrollment'].mean())
    print(f"\nEdad promedio al matricularse: {graduate['Age at enrollment'].mean():.1f} años")
    
    # Visualización comparativa
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    categorical_to_plot = ['Gender', 'Scholarship holder', 'Debtor', 
                           'Tuition fees up to date', 'Displaced', 
                           'Daytime/evening attendance']
    
    for idx, var in enumerate(categorical_to_plot):
        if var in df.columns:
            ax = axes[idx]
            
            # Preparar datos
            dropout_counts = dropout[var].value_counts(normalize=True) * 100
            graduate_counts = graduate[var].value_counts(normalize=True) * 100
            
            # Combinar
            combined = pd.DataFrame({
                'Dropout': dropout_counts,
                'Graduate': graduate_counts
            }).fillna(0)
            
            combined.plot(kind='bar', ax=ax, color=['#e74c3c', '#27ae60'])
            ax.set_title(var, fontweight='bold', fontsize=11)
            ax.set_ylabel('Porcentaje (%)')
            ax.set_xlabel('')
            ax.legend(['Dropout', 'Graduate'], fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/patrones_demograficos.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Figura guardada: {FIGURES_DIR}/patrones_demograficos.png")
    plt.close()
    
    return patterns

def analyze_academic_performance_patterns(df):
    """
    Analizar patrones de rendimiento académico específicos por grupo.
    """
    print(f"\n{'='*70}")
    print("📚 ANÁLISIS 2: PATRONES DE RENDIMIENTO ACADÉMICO")
    print('='*70)
    
    patterns = {'dropout': {}, 'graduate': {}}
    
    dropout = df[df['Target'] == 'Dropout']
    graduate = df[df['Target'] == 'Graduate']
    
    # Variables de rendimiento
    performance_vars = [
        'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Tasa_aprobacion_1sem',
        'Tasa_aprobacion_2sem',
        'Rendimiento_promedio'
    ]
    
    print("\n" + "─"*70)
    print("ESTADÍSTICAS DE RENDIMIENTO - DROPOUT")
    print("─"*70 + "\n")
    
    for var in performance_vars:
        if var in dropout.columns:
            stats_dropout = {
                'media': float(dropout[var].mean()),
                'mediana': float(dropout[var].median()),
                'desv_std': float(dropout[var].std()),
                'min': float(dropout[var].min()),
                'max': float(dropout[var].max()),
                'q25': float(dropout[var].quantile(0.25)),
                'q75': float(dropout[var].quantile(0.75))
            }
            patterns['dropout'][var] = stats_dropout
            
            print(f"{var}:")
            print(f"  Media: {stats_dropout['media']:.3f} | Mediana: {stats_dropout['mediana']:.3f}")
            print(f"  Rango: [{stats_dropout['min']:.3f}, {stats_dropout['max']:.3f}]")
            print()
    
    # Identificar patrones críticos para Dropout
    print("🔍 PATRONES CRÍTICOS PARA DROPOUT:")
    
    # Patrón 1: Estudiantes con tasa de aprobación = 0
    zero_approval_pct = (dropout['Tasa_aprobacion_1sem'] == 0).sum() / len(dropout) * 100
    patterns['dropout']['tasa_cero_aprobacion'] = float(zero_approval_pct)
    print(f"  • {zero_approval_pct:.1f}% tienen tasa de aprobación = 0 en 1er semestre")
    
    # Patrón 2: Estudiantes con bajo rendimiento (< 50% aprobación)
    low_performance_pct = (dropout['Tasa_aprobacion_1sem'] < 0.5).sum() / len(dropout) * 100
    patterns['dropout']['bajo_rendimiento_pct'] = float(low_performance_pct)
    print(f"  • {low_performance_pct:.1f}% tienen tasa < 50% de aprobación en 1er semestre")
    
    # Patrón 3: Calificación promedio baja
    low_grade_pct = (dropout['Curricular units 1st sem (grade)'] < 10).sum() / len(dropout) * 100
    patterns['dropout']['calificacion_baja_pct'] = float(low_grade_pct)
    print(f"  • {low_grade_pct:.1f}% tienen calificación promedio < 10 en 1er semestre")
    
    print("\n" + "─"*70)
    print("ESTADÍSTICAS DE RENDIMIENTO - GRADUATE")
    print("─"*70 + "\n")
    
    for var in performance_vars:
        if var in graduate.columns:
            stats_graduate = {
                'media': float(graduate[var].mean()),
                'mediana': float(graduate[var].median()),
                'desv_std': float(graduate[var].std()),
                'min': float(graduate[var].min()),
                'max': float(graduate[var].max()),
                'q25': float(graduate[var].quantile(0.25)),
                'q75': float(graduate[var].quantile(0.75))
            }
            patterns['graduate'][var] = stats_graduate
            
            print(f"{var}:")
            print(f"  Media: {stats_graduate['media']:.3f} | Mediana: {stats_graduate['mediana']:.3f}")
            print(f"  Rango: [{stats_graduate['min']:.3f}, {stats_graduate['max']:.3f}]")
            print()
    
    # Identificar patrones de éxito para Graduate
    print("🔍 PATRONES DE ÉXITO PARA GRADUATE:")
    
    # Patrón 1: Alto rendimiento
    high_performance_pct = (graduate['Tasa_aprobacion_1sem'] >= 0.8).sum() / len(graduate) * 100
    patterns['graduate']['alto_rendimiento_pct'] = float(high_performance_pct)
    print(f"  • {high_performance_pct:.1f}% tienen tasa ≥ 80% de aprobación en 1er semestre")
    
    # Patrón 2: Rendimiento perfecto
    perfect_performance_pct = (graduate['Tasa_aprobacion_1sem'] == 1.0).sum() / len(graduate) * 100
    patterns['graduate']['rendimiento_perfecto_pct'] = float(perfect_performance_pct)
    print(f"  • {perfect_performance_pct:.1f}% tienen tasa = 100% de aprobación en 1er semestre")
    
    # Patrón 3: Calificación alta
    high_grade_pct = (graduate['Curricular units 1st sem (grade)'] >= 14).sum() / len(graduate) * 100
    patterns['graduate']['calificacion_alta_pct'] = float(high_grade_pct)
    print(f"  • {high_grade_pct:.1f}% tienen calificación promedio ≥ 14 en 1er semestre")
    
    # Visualización de distribuciones
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Tasa aprobación 1er semestre
    ax1 = axes[0, 0]
    ax1.hist(dropout['Tasa_aprobacion_1sem'], bins=20, alpha=0.6, 
             label='Dropout', color='#e74c3c', density=True)
    ax1.hist(graduate['Tasa_aprobacion_1sem'], bins=20, alpha=0.6, 
             label='Graduate', color='#27ae60', density=True)
    ax1.axvline(dropout['Tasa_aprobacion_1sem'].mean(), color='#e74c3c', 
                linestyle='--', linewidth=2, label=f'Media Dropout: {dropout["Tasa_aprobacion_1sem"].mean():.2f}')
    ax1.axvline(graduate['Tasa_aprobacion_1sem'].mean(), color='#27ae60', 
                linestyle='--', linewidth=2, label=f'Media Graduate: {graduate["Tasa_aprobacion_1sem"].mean():.2f}')
    ax1.set_xlabel('Tasa Aprobación 1er Semestre', fontsize=11)
    ax1.set_ylabel('Densidad', fontsize=11)
    ax1.set_title('Distribución: Tasa de Aprobación (1er Sem)', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Calificaciones 1er semestre
    ax2 = axes[0, 1]
    ax2.hist(dropout['Curricular units 1st sem (grade)'], bins=20, alpha=0.6, 
             label='Dropout', color='#e74c3c', density=True)
    ax2.hist(graduate['Curricular units 1st sem (grade)'], bins=20, alpha=0.6, 
             label='Graduate', color='#27ae60', density=True)
    ax2.axvline(dropout['Curricular units 1st sem (grade)'].mean(), color='#e74c3c', 
                linestyle='--', linewidth=2)
    ax2.axvline(graduate['Curricular units 1st sem (grade)'].mean(), color='#27ae60', 
                linestyle='--', linewidth=2)
    ax2.set_xlabel('Calificación Promedio 1er Semestre', fontsize=11)
    ax2.set_ylabel('Densidad', fontsize=11)
    ax2.set_title('Distribución: Calificación Promedio (1er Sem)', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Unidades aprobadas 1er semestre
    ax3 = axes[1, 0]
    ax3.hist(dropout['Curricular units 1st sem (approved)'], bins=15, alpha=0.6, 
             label='Dropout', color='#e74c3c', density=True)
    ax3.hist(graduate['Curricular units 1st sem (approved)'], bins=15, alpha=0.6, 
             label='Graduate', color='#27ae60', density=True)
    ax3.set_xlabel('Unidades Aprobadas 1er Semestre', fontsize=11)
    ax3.set_ylabel('Densidad', fontsize=11)
    ax3.set_title('Distribución: Unidades Aprobadas (1er Sem)', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rendimiento promedio (ambos semestres)
    ax4 = axes[1, 1]
    ax4.hist(dropout['Rendimiento_promedio'], bins=20, alpha=0.6, 
             label='Dropout', color='#e74c3c', density=True)
    ax4.hist(graduate['Rendimiento_promedio'], bins=20, alpha=0.6, 
             label='Graduate', color='#27ae60', density=True)
    ax4.axvline(dropout['Rendimiento_promedio'].mean(), color='#e74c3c', 
                linestyle='--', linewidth=2)
    ax4.axvline(graduate['Rendimiento_promedio'].mean(), color='#27ae60', 
                linestyle='--', linewidth=2)
    ax4.set_xlabel('Rendimiento Promedio (Ambos Semestres)', fontsize=11)
    ax4.set_ylabel('Densidad', fontsize=11)
    ax4.set_title('Distribución: Rendimiento Promedio', fontweight='bold', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/patrones_rendimiento.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Figura guardada: {FIGURES_DIR}/patrones_rendimiento.png")
    plt.close()
    
    return patterns

def analyze_correlation_patterns(df):
    """
    Analizar correlaciones específicas para cada grupo.
    """
    print(f"\n{'='*70}")
    print("🔗 ANÁLISIS 3: PATRONES DE CORRELACIÓN POR GRUPO")
    print('='*70)
    
    patterns = {}
    
    # Variables numéricas para correlación
    numeric_vars = [
        'Age at enrollment',
        'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Unemployment rate',
        'Inflation rate',
        'GDP'
    ]
    
    dropout = df[df['Target'] == 'Dropout'][numeric_vars]
    graduate = df[df['Target'] == 'Graduate'][numeric_vars]
    
    # Calcular correlaciones
    corr_dropout = dropout.corr()
    corr_graduate = graduate.corr()
    
    # Guardar correlaciones
    patterns['dropout_correlations'] = corr_dropout.to_dict()
    patterns['graduate_correlations'] = corr_graduate.to_dict()
    
    # Visualización de matrices de correlación
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Dropout correlations
    ax1 = axes[0]
    sns.heatmap(corr_dropout, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                ax=ax1, cbar_kws={'label': 'Correlación'}, 
                vmin=-1, vmax=1, square=True)
    ax1.set_title('Matriz de Correlación - DROPOUT', fontweight='bold', fontsize=14)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)
    
    # Graduate correlations
    ax2 = axes[1]
    sns.heatmap(corr_graduate, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                ax=ax2, cbar_kws={'label': 'Correlación'}, 
                vmin=-1, vmax=1, square=True)
    ax2.set_title('Matriz de Correlación - GRADUATE', fontweight='bold', fontsize=14)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/patrones_correlacion.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Figura guardada: {FIGURES_DIR}/patrones_correlacion.png")
    plt.close()
    
    # Identificar correlaciones más fuertes
    print("\n" + "─"*70)
    print("CORRELACIONES MÁS FUERTES - DROPOUT")
    print("─"*70)
    
    # Extraer correlaciones (excluyendo diagonal)
    dropout_corr_flat = []
    for i in range(len(corr_dropout.columns)):
        for j in range(i+1, len(corr_dropout.columns)):
            dropout_corr_flat.append({
                'var1': corr_dropout.columns[i],
                'var2': corr_dropout.columns[j],
                'corr': corr_dropout.iloc[i, j]
            })
    
    dropout_corr_df = pd.DataFrame(dropout_corr_flat).sort_values('corr', 
                                                                   key=abs, 
                                                                   ascending=False)
    
    print("\nTop 10 correlaciones más fuertes:")
    for idx, row in dropout_corr_df.head(10).iterrows():
        print(f"  • {row['var1'][:40]:40} <-> {row['var2'][:40]:40} : {row['corr']:+.3f}")
    
    print("\n" + "─"*70)
    print("CORRELACIONES MÁS FUERTES - GRADUATE")
    print("─"*70)
    
    graduate_corr_flat = []
    for i in range(len(corr_graduate.columns)):
        for j in range(i+1, len(corr_graduate.columns)):
            graduate_corr_flat.append({
                'var1': corr_graduate.columns[i],
                'var2': corr_graduate.columns[j],
                'corr': corr_graduate.iloc[i, j]
            })
    
    graduate_corr_df = pd.DataFrame(graduate_corr_flat).sort_values('corr', 
                                                                     key=abs, 
                                                                     ascending=False)
    
    print("\nTop 10 correlaciones más fuertes:")
    for idx, row in graduate_corr_df.head(10).iterrows():
        print(f"  • {row['var1'][:40]:40} <-> {row['var2'][:40]:40} : {row['corr']:+.3f}")
    
    return patterns

def perform_clustering_analysis(df):
    """
    Realizar análisis de segmentación para identificar subgrupos dentro de cada categoría.
    Versión alternativa sin sklearn - usa percentiles para segmentar.
    """
    print(f"\n{'='*70}")
    print("🎯 ANÁLISIS 4: SEGMENTACIÓN - IDENTIFICACIÓN DE SUBGRUPOS")
    print('='*70)
    
    patterns = {}
    
    # Variables para segmentación
    cluster_vars = [
        'Age at enrollment',
        'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Tasa_aprobacion_1sem',
        'Tasa_aprobacion_2sem',
        'Unemployment rate',
        'GDP'
    ]
    
    # Función auxiliar para segmentar por rendimiento
    def segment_group(data, group_name):
        print(f"\n{'─'*70}")
        print(f"Segmentación para grupo: {group_name.upper()}")
        print('─'*70)
        
        # Segmentar por tasa de aprobación del 1er semestre
        tasa_var = 'Tasa_aprobacion_1sem'
        
        # Definir tres segmentos: Bajo, Medio, Alto rendimiento
        q33 = data[tasa_var].quantile(0.33)
        q67 = data[tasa_var].quantile(0.67)
        
        data_segmented = data.copy()
        data_segmented['Segmento'] = pd.cut(
            data_segmented[tasa_var],
            bins=[-np.inf, q33, q67, np.inf],
            labels=['Bajo Rendimiento', 'Rendimiento Medio', 'Alto Rendimiento']
        )
        
        print(f"\nSe identificaron 3 segmentos basados en tasa de aprobación:")
        print(f"  • Bajo Rendimiento: Tasa ≤ {q33:.2f}")
        print(f"  • Rendimiento Medio: Tasa entre {q33:.2f} y {q67:.2f}")
        print(f"  • Alto Rendimiento: Tasa > {q67:.2f}")
        
        segment_profiles = {}
        
        for seg in ['Bajo Rendimiento', 'Rendimiento Medio', 'Alto Rendimiento']:
            segment_data = data_segmented[data_segmented['Segmento'] == seg]
            n_students = len(segment_data)
            pct = n_students / len(data_segmented) * 100
            
            print(f"\n  📌 {seg} ({n_students} estudiantes, {pct:.1f}%):")
            
            profile = {
                'n_estudiantes': int(n_students),
                'porcentaje': float(pct),
                'caracteristicas': {}
            }
            
            # Características principales
            for var in cluster_vars:
                if var in segment_data.columns:
                    mean_val = segment_data[var].mean()
                    overall_mean = data_segmented[var].mean()
                    diff_pct = ((mean_val - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0
                    
                    profile['caracteristicas'][var] = {
                        'media': float(mean_val),
                        'diferencia_vs_promedio': f"{diff_pct:+.1f}%"
                    }
                    
                    if abs(diff_pct) > 15:  # Solo mostrar diferencias significativas
                        print(f"     • {var}: {mean_val:.2f} ({diff_pct:+.1f}% vs promedio)")
            
            segment_profiles[seg.lower().replace(' ', '_')] = profile
        
        # Visualización de segmentos
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Distribución por segmento
        ax1 = axes[0, 0]
        segment_counts = data_segmented['Segmento'].value_counts()
        colors_seg = ['#e74c3c', '#f39c12', '#27ae60']
        ax1.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%',
               colors=colors_seg, startangle=90)
        ax1.set_title(f'Distribución de Segmentos\n{group_name.upper()}', 
                     fontweight='bold', fontsize=12)
        
        # Plot 2: Boxplot de tasa por segmento
        ax2 = axes[0, 1]
        data_segmented.boxplot(column='Tasa_aprobacion_1sem', by='Segmento', ax=ax2)
        ax2.set_title('Tasa Aprobación por Segmento', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Segmento')
        ax2.set_ylabel('Tasa Aprobación 1er Sem')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # Plot 3: Calificación promedio por segmento
        ax3 = axes[1, 0]
        segment_means = data_segmented.groupby('Segmento')['Curricular units 1st sem (grade)'].mean()
        ax3.bar(range(len(segment_means)), segment_means.values, color=colors_seg, alpha=0.7)
        ax3.set_xticks(range(len(segment_means)))
        ax3.set_xticklabels(segment_means.index, rotation=15, ha='right')
        ax3.set_ylabel('Calificación Promedio')
        ax3.set_title('Calificación Promedio por Segmento', fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Edad por segmento
        ax4 = axes[1, 1]
        data_segmented.boxplot(column='Age at enrollment', by='Segmento', ax=ax4)
        ax4.set_title('Edad por Segmento', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Segmento')
        ax4.set_ylabel('Edad al Matricularse')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        plt.suptitle('')  # Remover título automático
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/segmentacion_{group_name.lower()}.png', 
                   dpi=150, bbox_inches='tight')
        print(f"\n✓ Figura guardada: {FIGURES_DIR}/segmentacion_{group_name.lower()}.png")
        plt.close()
        
        return segment_profiles
    
    # Segmentación para Dropout
    dropout_data = df[df['Target'] == 'Dropout']
    patterns['dropout_segments'] = segment_group(dropout_data, 'Dropout')
    
    # Segmentación para Graduate
    graduate_data = df[df['Target'] == 'Graduate']
    patterns['graduate_segments'] = segment_group(graduate_data, 'Graduate')
    
    return patterns

def identify_key_differentiators(df):
    """
    Identificar las variables que más diferencian a los dos grupos.
    """
    print(f"\n{'='*70}")
    print("🔑 ANÁLISIS 5: VARIABLES DISCRIMINANTES CLAVE")
    print('='*70)
    
    patterns = {}
    
    dropout = df[df['Target'] == 'Dropout']
    graduate = df[df['Target'] == 'Graduate']
    
    # Variables a analizar
    test_vars = [
        'Age at enrollment',
        'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Tasa_aprobacion_1sem',
        'Tasa_aprobacion_2sem',
        'Rendimiento_promedio',
        'Scholarship holder',
        'Debtor',
        'Tuition fees up to date',
        'Unemployment rate',
        'GDP'
    ]
    
    differentiators = []
    
    print("\nPruebas de significancia para identificar variables discriminantes:\n")
    
    for var in test_vars:
        if var in df.columns:
            dropout_vals = dropout[var].dropna()
            graduate_vals = graduate[var].dropna()
            # Determinar si la variable es numérica o categórica
            if pd.api.types.is_numeric_dtype(dropout_vals) and pd.api.types.is_numeric_dtype(graduate_vals):
                # Mann-Whitney U test (no paramétrico)
                stat, p_value = stats.mannwhitneyu(dropout_vals, graduate_vals, 
                                                   alternative='two-sided')

                # Calcular tamaño del efecto (r = Z / sqrt(N))
                n = len(dropout_vals) + len(graduate_vals)
                # Evitar problemas con p_value extremos
                try:
                    z_score = stats.norm.ppf(1 - p_value/2)
                    effect_size = abs(z_score) / np.sqrt(n)
                except Exception:
                    effect_size = 0.0

                # Diferencia de medias
                mean_dropout = float(dropout_vals.mean())
                mean_graduate = float(graduate_vals.mean())
                mean_diff = mean_graduate - mean_dropout
                mean_diff_pct = (mean_diff / mean_dropout * 100) if mean_dropout != 0 else 0

                differentiators.append({
                    'variable': var,
                    'p_value': float(p_value),
                    'effect_size': float(effect_size),
                    'mean_dropout': mean_dropout,
                    'mean_graduate': mean_graduate,
                    'mean_diff': float(mean_diff),
                    'mean_diff_pct': float(mean_diff_pct),
                    'significant': p_value < 0.05,
                    'type': 'numeric'
                })
            else:
                # Variable categórica: usar chi-cuadrado sobre tabla de contingencia
                try:
                    contingency = pd.crosstab(df[var], df['Target'])
                    # Asegurar columnas Dropout/Graduate
                    cols = [c for c in ['Dropout', 'Graduate'] if c in contingency.columns]
                    if len(cols) == 2:
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency[cols])
                        n = contingency[cols].values.sum()
                        k = min(contingency.shape)
                        # Cramér's V
                        if n > 0 and (k - 1) > 0:
                            cramers_v = (chi2 / (n * (k - 1)))**0.5
                        else:
                            cramers_v = 0.0
                    else:
                        p_value = 1.0
                        cramers_v = 0.0
                except Exception:
                    p_value = 1.0
                    cramers_v = 0.0

                differentiators.append({
                    'variable': var,
                    'p_value': float(p_value),
                    'effect_size': float(cramers_v),
                    'mean_dropout': None,
                    'mean_graduate': None,
                    'mean_diff': None,
                    'mean_diff_pct': None,
                    'significant': p_value < 0.05,
                    'type': 'categorical'
                })
    
    # Ordenar por p-value (más significativo primero)
    differentiators_df = pd.DataFrame(differentiators).sort_values('p_value')
    
    patterns['key_differentiators'] = differentiators_df.to_dict('records')
    
    print("Variables ordenadas por poder discriminante (p-value):\n")
    print(f"{'Variable':<50} {'p-value':<12} {'Efecto':<10} {'Diferencia':<15}")
    print("─"*90)
    
    for _, row in differentiators_df.head(15).iterrows():
        sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['variable']:<50} {row['p_value']:<12.2e} {row['effect_size']:<10.3f} "
              f"{row['mean_diff_pct']:+.1f}% {sig_marker}")
    
    # Visualización de las top variables discriminantes
    top_vars = differentiators_df.head(10)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(top_vars))
    
    # Crear barras para Graduate y Dropout
    width = 0.35
    
    graduate_means = [row['mean_graduate'] for _, row in top_vars.iterrows()]
    dropout_means = [row['mean_dropout'] for _, row in top_vars.iterrows()]
    var_names = [row['variable'] for _, row in top_vars.iterrows()]
    
    ax.barh(x_pos - width/2, graduate_means, width, label='Graduate', 
            color='#27ae60', alpha=0.8)
    ax.barh(x_pos + width/2, dropout_means, width, label='Dropout', 
            color='#e74c3c', alpha=0.8)
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels(var_names, fontsize=10)
    ax.set_xlabel('Valor Promedio', fontsize=11)
    ax.set_title('Top 10 Variables Discriminantes\n(Comparación Graduate vs Dropout)', 
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/variables_discriminantes.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Figura guardada: {FIGURES_DIR}/variables_discriminantes.png")
    plt.close()
    
    return patterns

def generate_summary_report(demographic_patterns, performance_patterns, 
                           correlation_patterns, segmentation_patterns, 
                           differentiator_patterns, df):
    """
    Generar reporte resumen de todos los patrones identificados.
    """
    print(f"\n{'='*70}")
    print("📄 GENERACIÓN DE REPORTE FINAL")
    print('='*70)
    
    report = {
        'metadata': {
            'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': 'outputs/prepared_data/dataset_prepared.csv',
            'n_dropout': int((df['Target'] == 'Dropout').sum()),
            'n_graduate': int((df['Target'] == 'Graduate').sum()),
            'n_total_analizado': len(df)
        },
        'patrones_demograficos': demographic_patterns,
        'patrones_rendimiento': performance_patterns,
        'patrones_correlacion': 'Ver archivo separado (muy extenso)',
        'segmentacion': segmentation_patterns,
        'variables_discriminantes': differentiator_patterns,
        'resumen_ejecutivo': {
            'dropout': {
                'descripcion': 'Estudiantes que abandonaron los estudios',
                'n': int((df['Target'] == 'Dropout').sum()),
                'caracteristicas_clave': [
                    f"Tasa promedio de aprobación 1er sem: {df[df['Target']=='Dropout']['Tasa_aprobacion_1sem'].mean():.1%}",
                    f"Edad promedio: {df[df['Target']=='Dropout']['Age at enrollment'].mean():.1f} años",
                    f"Calificación promedio 1er sem: {df[df['Target']=='Dropout']['Curricular units 1st sem (grade)'].mean():.2f}"
                ]
            },
            'graduate': {
                'descripcion': 'Estudiantes que completaron los estudios',
                'n': int((df['Target'] == 'Graduate').sum()),
                'caracteristicas_clave': [
                    f"Tasa promedio de aprobación 1er sem: {df[df['Target']=='Graduate']['Tasa_aprobacion_1sem'].mean():.1%}",
                    f"Edad promedio: {df[df['Target']=='Graduate']['Age at enrollment'].mean():.1f} años",
                    f"Calificación promedio 1er sem: {df[df['Target']=='Graduate']['Curricular units 1st sem (grade)'].mean():.2f}"
                ]
            }
        }
    }
    
    # Guardar reporte
    report_path = f'{OUTPUT_DIR}/pattern_analysis_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Reporte guardado: {report_path}")
    
    # Crear resumen en texto
    summary_path = f'{OUTPUT_DIR}/pattern_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RESUMEN DE PATRONES IDENTIFICADOS\n")
        f.write("="*70 + "\n\n")
        
        f.write("PATRONES PARA ESTUDIANTES QUE ABANDONAN (DROPOUT)\n")
        f.write("-"*70 + "\n\n")
        
        dropout_data = df[df['Target'] == 'Dropout']
        f.write(f"Número de estudiantes: {len(dropout_data):,}\n\n")
        f.write("Características principales:\n")
        f.write(f"  • Tasa aprobación 1er sem: {dropout_data['Tasa_aprobacion_1sem'].mean():.1%}\n")
        f.write(f"  • Calificación promedio 1er sem: {dropout_data['Curricular units 1st sem (grade)'].mean():.2f}\n")
        f.write(f"  • Edad promedio: {dropout_data['Age at enrollment'].mean():.1f} años\n")
        f.write(f"  • % con tasa aprobación = 0: {(dropout_data['Tasa_aprobacion_1sem']==0).sum()/len(dropout_data)*100:.1f}%\n")
        f.write(f"  • % con tasa aprobación < 50%: {(dropout_data['Tasa_aprobacion_1sem']<0.5).sum()/len(dropout_data)*100:.1f}%\n")
        
        f.write("\n\nPATRONES PARA ESTUDIANTES QUE SE GRADÚAN (GRADUATE)\n")
        f.write("-"*70 + "\n\n")
        
        graduate_data = df[df['Target'] == 'Graduate']
        f.write(f"Número de estudiantes: {len(graduate_data):,}\n\n")
        f.write("Características principales:\n")
        f.write(f"  • Tasa aprobación 1er sem: {graduate_data['Tasa_aprobacion_1sem'].mean():.1%}\n")
        f.write(f"  • Calificación promedio 1er sem: {graduate_data['Curricular units 1st sem (grade)'].mean():.2f}\n")
        f.write(f"  • Edad promedio: {graduate_data['Age at enrollment'].mean():.1f} años\n")
        f.write(f"  • % con tasa aprobación ≥ 80%: {(graduate_data['Tasa_aprobacion_1sem']>=0.8).sum()/len(graduate_data)*100:.1f}%\n")
        f.write(f"  • % con tasa aprobación = 100%: {(graduate_data['Tasa_aprobacion_1sem']==1.0).sum()/len(graduate_data)*100:.1f}%\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Resumen de texto guardado: {summary_path}")
    
    return report

def main():
    """Función principal."""
    print("\n" + "="*70)
    print("  ANÁLISIS DE PATRONES: DROPOUT vs GRADUATE")
    print("  Identificación de patrones específicos por grupo")
    print("="*70)
    
    # Preparación
    create_output_directories()
    
    # Cargar datos
    df = load_and_prepare_data('outputs/prepared_data/dataset_prepared.csv')
    
    # Análisis 1: Patrones demográficos
    demographic_patterns = analyze_demographic_patterns(df)
    
    # Análisis 2: Patrones de rendimiento
    performance_patterns = analyze_academic_performance_patterns(df)
    
    # Análisis 3: Patrones de correlación
    correlation_patterns = analyze_correlation_patterns(df)
    
    # Análisis 4: Segmentación
    segmentation_patterns = perform_clustering_analysis(df)
    
    # Análisis 5: Variables discriminantes
    differentiator_patterns = identify_key_differentiators(df)
    
    # Generar reporte final
    report = generate_summary_report(
        demographic_patterns, performance_patterns, correlation_patterns,
        segmentation_patterns, differentiator_patterns, df
    )
    
    print(f"\n{'='*70}")
    print("✅ ANÁLISIS COMPLETADO")
    print('='*70)
    print(f"\n📁 Resultados guardados en: {OUTPUT_DIR}/")
    print(f"\nArchivos generados:")
    print(f"  • pattern_analysis_report.json (reporte completo)")
    print(f"  • pattern_summary.txt (resumen ejecutivo)")
    print(f"\nFiguras generadas:")
    print(f"  • patrones_demograficos.png")
    print(f"  • patrones_rendimiento.png")
    print(f"  • patrones_correlacion.png")
    print(f"  • segmentacion_dropout.png")
    print(f"  • segmentacion_graduate.png")
    print(f"  • variables_discriminantes.png")
    print(f"\n{'='*70}\n")
    
    return df, report

if __name__ == "__main__":
    df_result, report_result = main()
