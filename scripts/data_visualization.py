"""
Script de EDA (Exploratory Data Analysis) para el proyecto.

Salida esperada:
- Imágenes: histogramas, boxplots, scatterplots, gráficos comparativos

El objetivo principal: generar gráficos automáticamente de forma robusta
cuando el conjunto de datos puede variar (detecta columnas numéricas,
intenta inferir columna objetivo, etc.).
"""
import argparse              # Manejo de argumentos desde la línea de comandos
import re                    # Expresiones regulares para validación y reemplazo de patrones
from pathlib import Path     # Manipulación de rutas y archivos de forma orientada a objetos
import pandas as pd          # Análisis y manipulación de datos en DataFrames
import numpy as np           # Operaciones numéricas y manejo eficiente de arreglos
import matplotlib.pyplot as plt   # Generación de gráficos y visualizaciones básicas
import seaborn as sns        # Visualización estadística con estilos y gráficos avanzados
import matplotlib.patches as mpatches

def safe_filename(name: str) -> str:
    """Sanitiza una cadena para que sea segura como nombre de archivo.

    Reemplaza caracteres problemáticos por guiones bajos para evitar
    errores del sistema de ficheros cuando se usan nombres de columnas
    o valores de datos como nombres de archivo.

    Args:
        name (str): Cadena de entrada que se desea sanitizar.

    Returns:
        str: Cadena resultante válida como nombre de fichero.
    """
    # Reemplaza caracteres que suelen causar problemas en nombres de fichero
    # por un guion bajo. Se asegura de devolver siempre una cadena.
    return re.sub(r'[<>:"/\\|?*]', '_', str(name))

def generate_histograms(df, numeric, out_dir, show=False):
    """Genera y guarda histogramas (con KDE) para columnas numéricas.

    Crea la carpeta `histograms` dentro de `out_dir` y guarda un PNG por
    cada columna listada en `numeric`.

    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        numeric (list[str]): Lista de nombres de columnas numéricas.
        out_dir (pathlib.Path): Carpeta base donde guardar las imágenes.

    Returns:
        None: Los archivos se escriben en disco.
    """
    hist_dir = out_dir / 'histograms'
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Para cada columna numérica, guardamos un histograma con KDE.
    # Se usa dropna() para evitar que los NaN distorsionen la gráfica.
    for c in numeric:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[c].dropna(), kde=True)
        plt.title(f'Histogram: {c}')
        plt.tight_layout()
        # Nombre seguro para evitar caracteres no válidos en el sistema de ficheros
        plt.savefig(hist_dir / f'hist_{safe_filename(c)}.png', dpi=150)
        if show:
            # Opción para mostrar en pantalla además de guardar
            plt.show()
        plt.close()

def generate_boxplots(df, numeric, out_dir, show=False):
    """Genera y guarda diagramas de caja (boxplots) para columnas numéricas.

    Crea la carpeta `boxplots` dentro de `out_dir` y guarda un PNG por
    cada columna listada en `numeric`.

    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        numeric (list[str]): Lista de nombres de columnas numéricas.
        out_dir (pathlib.Path): Carpeta base donde guardar las imágenes.

    Returns:
        None: Los archivos se escriben en disco.
    """
    box_dir = out_dir / 'boxplots'
    box_dir.mkdir(parents=True, exist_ok=True)

    # Boxplots por columna numérica. Se muestran outliers por defecto,
    # lo cual es útil en EDA para detectar valores extremos.
    for c in numeric:
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[c].dropna())
        plt.title(f'Boxplot: {c}')
        plt.tight_layout()
        plt.savefig(box_dir / f'box_{safe_filename(c)}.png', dpi=150)
        if show:
            plt.show()
        plt.close()

def generate_target_piechart(df, out_dir, target_col='Target', show=False):
    """Genera y guarda un gráfico de pastel con la distribución de estado del estudiante.

    Cuenta cuántos son `graduado`, `abandonado` y los que siguen estudiando (`en_estudio`).
    Busca la columna `target_col` o una candidata parecida si no existe.
    """
    # Aseguramos que exista el directorio de salida
    out_dir.mkdir(parents=True, exist_ok=True)

    # Función auxiliar para normalizar distintas formas de representar la etiqueta
    def classify_target(val):
        if pd.isna(val):
            return None
        # Si la columna es numérica, se asume la convención 1=graduado, 0=abandonado
        if isinstance(val, (int, np.integer, float, np.floating)):
            if val == 1:
                return 'graduado'
            if val == 0:
                return 'abandonado'
        s = str(val).lower()
        if any(k in s for k in ('grad', 'success', 'aprob', 'pass', 'passed')):
            return 'graduado'
        if any(k in s for k in ('drop', 'aband', 'fail', 'deser', 'abandono')):
            return 'abandonado'
        return 'en_estudio'

    # Intentamos usar la columna explícita `target_col`; si no existe,
    # buscamos una columna parecida (target/status/result). Si no hay
    # candidatas, asumimos que todos siguen estudiando.
    if target_col in df.columns:
        labels = df[target_col].apply(classify_target)
    else:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ('target', 'status', 'result'))]
        if candidates:
            labels = df[candidates[0]].apply(classify_target)
        else:
            print('No se encontró columna target; todos los registros marcados como en_estudio.')
            labels = pd.Series(['en_estudio'] * len(df), index=df.index)

    # Conteo de cada categoría para construir el pie chart
    counts = labels.value_counts()
    graduado = int(counts.get('graduado', 0))
    abandonado = int(counts.get('abandonado', 0))
    en_estudio = int(counts.get('en_estudio', 0))

    sizes = [graduado, abandonado, en_estudio]
    names = ['Graduados', 'Desertores', 'Siguen estudiando']

    # Si no hay observaciones útiles, no intentamos dibujar el gráfico
    if sum(sizes) == 0:
        print('No hay datos para generar el gráfico de pastel.')
        return

    # Dibujar y guardar el gráfico de pastel
    plt.figure(figsize=(6, 6))
    colors = sns.color_palette('pastel')[0:3]
    plt.pie(sizes, labels=names, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Distribución: Graduados / Desertores / Siguen estudiando')
    plt.tight_layout()
    out_file = out_dir / f'pie_{safe_filename(target_col)}.png'
    plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print('Gráfico de pastel guardado en:', out_file)

def plot_scholarship_pie(df, out_dir, target_col='Target', scholarship_col='Scholarship holder', show=False):
    """
    Dibuja dos gráficos de pastel (lado a lado) mostrando la proporción
    de estudiantes con y sin beca dentro de los graduados y de los no graduados.

    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        out_dir (pathlib.Path or str): Directorio donde guardar la imagen.
        target_col (str): Nombre de la columna objetivo/estado (por defecto 'Target').
        scholarship_col (str): Nombre de la columna que indica si tiene beca.
        show (bool): Si True, muestra las figuras además de guardarlas.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _classify_target(val):
        if pd.isna(val):
            return None
        if isinstance(val, (int, np.integer, float, np.floating)):
            if val == 1:
                return 'graduado'
            if val == 0:
                return 'abandonado'
        s = str(val).lower()
        if any(k in s for k in ('grad', 'success', 'aprob', 'pass', 'passed')):
            return 'graduado'
        if any(k in s for k in ('drop', 'aband', 'fail', 'deser', 'abandono')):
            return 'abandonado'
        return None

    def _normalize_scholar(val):
        if pd.isna(val):
            return None
        if isinstance(val, (int, np.integer, float, np.floating)):
            return 'Con beca' if int(val) == 1 else 'Sin beca'
        s = str(val).strip().lower()
        if s in ('1', 'yes', 'y', 'si', 's', 'true', 't'):
            return 'Con beca'
        if s in ('0', 'no', 'n', 'false', 'f'):
            return 'Sin beca'
        # Valores no evidentes los clasificamos conservadoramente como 'Sin beca'
        return 'Sin beca'

    # Localizar columnas objetivo y de beca si no existen exactamente con ese nombre
    if target_col in df.columns:
        targets = df[target_col].apply(_classify_target)
    else:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ('target', 'status', 'result'))]
        targets = df[candidates[0]].apply(_classify_target) if candidates else pd.Series([None] * len(df), index=df.index)

    if scholarship_col in df.columns:
        scholars = df[scholarship_col].apply(_normalize_scholar)
    else:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ('scholar', 'beca', 'grant'))]
        if candidates:
            scholars = df[candidates[0]].apply(_normalize_scholar)
        else:
            print(f'No se encontró columna de beca ({scholarship_col}).')
            return

    combined = pd.DataFrame({'target': targets, 'scholar': scholars})

    # Filtrar sólo graduado/abandonado
    groups = {'Graduados': 'graduado', 'Desertores': 'abandonado'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    colors = sns.color_palette('Set2')[0:2]

    any_data = False
    for ax, (title, label_val) in zip(axes, groups.items()):
        subset = combined[combined['target'] == label_val]['scholar'].dropna()
        counts = subset.value_counts()
        con = int(counts.get('Con beca', 0))
        sin = int(counts.get('Sin beca', 0))
        total = con + sin
        if total == 0:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center')
            ax.set_title(title)
            ax.axis('off')
            continue
        any_data = True
        sizes = [con, sin]
        labels = [f'Con beca ({con})', f'Sin beca ({sin})']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title(title)

    if not any_data:
        print('No hay datos válidos de beca para Graduados ni Desertores.')
        plt.close(fig)
        return

    plt.suptitle('Proporción de estudiantes con/sin beca por estado')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = out_dir / f'pie_scholarship_{safe_filename(scholarship_col)}.png'
    plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print('Gráfico de becas guardado en:', out_file)

def generate_rendimiento_comparison(df, out_dir, target_col='Target', show=False):
    """
    Genera una figura comparativa con boxplot (izquierda) y histogramas superpuestos (derecha).
    - Calcula `rendimiento` como la media por fila de las columnas de nota.
    - Clasifica la columna objetivo en dos grupos: `Dropout` y `Graduate`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detectar columnas de nota
    grade_cols = [c for c in df.columns if 'grade' in c.lower()]
    if not grade_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print('No se encontraron columnas de nota ni numéricas para calcular rendimiento.')
            return
        grade_cols = numeric_cols

    rendimiento = df[grade_cols].mean(axis=1, skipna=True)

    def classify_target(val):
        if pd.isna(val):
            return None
        if isinstance(val, (int, np.integer, float, np.floating)):
            if val == 1:
                return 'Graduate'
            if val == 0:
                return 'Dropout'
        s = str(val).lower()
        if any(k in s for k in ('grad', 'success', 'aprob', 'pass', 'passed')):
            return 'Graduate'
        if any(k in s for k in ('drop', 'aband', 'fail', 'deser', 'abandono')):
            return 'Dropout'
        return None

    if target_col in df.columns:
        labels = df[target_col].apply(classify_target)
    else:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ('target', 'status', 'result'))]
        if candidates:
            labels = df[candidates[0]].apply(classify_target)
        else:
            labels = pd.Series([None] * len(df), index=df.index)

    plot_df = pd.DataFrame({'rendimiento': rendimiento, 'label': labels})
    plot_df = plot_df.dropna(subset=['rendimiento', 'label'])

    if plot_df.empty:
        print('No hay datos suficientes para generar la figura comparativa (falta etiqueta o notas).')
        return

    # Asegurar orden consistente en ejes
    order = ['Dropout', 'Graduate']

    # Calcular medias para anotarlas
    means = plot_df.groupby('label')['rendimiento'].mean().reindex(order)

    # Figura con dos subplots lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 1.2]})

    # Boxplot (izquierda)
    ax = axes[0]
    sns.boxplot(x='label', y='rendimiento', data=plot_df, order=order, ax=ax, palette=['#f49a9a', '#6bdb9a'])
    # Añadir marcador de la media y etiqueta mu
    for i, lab in enumerate(order):
        m = means.get(lab)
        if pd.notna(m):
            ax.scatter(i, m, color='white', edgecolor='black', marker='D', zorder=10, s=80)
            ax.text(i, m, f' μ={m:.3f}', va='center', ha='left', fontsize=10)

    ax.set_title('Comparación de Rendimientos\nBoxplot Conjuntos')
    ax.set_xlabel('')
    ax.set_ylabel('Notas (puntuación)')

    # Histograma superpuesto (derecha)
    ax2 = axes[1]
    sns.histplot(data=plot_df, x='rendimiento', hue='label', stat='count', element='step',
                 common_norm=False, alpha=0.5, bins=20, palette=['#f49a9a', '#6bdb9a'], ax=ax2)
    ax2.set_title('Distribución de Rendimientos\nHistogramas Superpuestos')
    ax2.set_xlabel('Notas (puntuación)')
    ax2.set_ylabel('Número de estudiantes')
    # Añadir leyenda en esquina indicando a qué grupo corresponde cada color
    legend_handles = [
        mpatches.Patch(color='#f49a9a', label='Dropout'),
        mpatches.Patch(color='#6bdb9a', label='Graduate')
    ]
    ax2.legend(handles=legend_handles, title='', loc='upper right', frameon=True)

    plt.tight_layout()
    out_file = out_dir / f'rendimiento_comparison_{safe_filename(target_col)}.png'
    plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print('Histograma de rendimientos guardado en:', out_file)

def generate_age_overlay_histogram(df, out_dir, age_col=None, target_col='Target', show=False):
    """
    Superpone histogramas de edad para ambos grupos (Dropout vs Graduate).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detectar columna de edad
    if age_col and age_col in df.columns:
        age_col_name = age_col
    else:
        age_col_name = None
        for c in df.columns:
            if 'age' in c.lower():
                age_col_name = c
                break
    if age_col_name is None:
        print('No se encontró columna de edad.')
        return

    # Clasificar objetivo en dos grupos
    def _classify(val):
        if pd.isna(val):
            return None
        if isinstance(val, (int, np.integer, float, np.floating)):
            if val == 1:
                return 'Graduate'
            if val == 0:
                return 'Dropout'
        s = str(val).lower()
        if any(k in s for k in ('grad', 'success', 'aprob', 'pass', 'passed')):
            return 'Graduate'
        if any(k in s for k in ('drop', 'aband', 'fail', 'deser', 'abandono')):
            return 'Dropout'
        return None

    if target_col in df.columns:
        labels = df[target_col].apply(_classify)
    else:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ('target', 'status', 'result'))]
        labels = df[candidates[0]].apply(_classify) if candidates else pd.Series([None]*len(df), index=df.index)

    plot_df = pd.DataFrame({age_col_name: df[age_col_name], 'label': labels})
    # Convertir edad a numérico y eliminar valores no convertibles
    plot_df[age_col_name] = pd.to_numeric(plot_df[age_col_name], errors='coerce')
    plot_df = plot_df.dropna(subset=[age_col_name])

    g1 = plot_df[plot_df['label'] == 'Dropout'][age_col_name]
    g2 = plot_df[plot_df['label'] == 'Graduate'][age_col_name]

    if g1.empty and g2.empty:
        print('No hay datos de edad con etiquetas válidas para superponer.')
        return

    # Definir rango x (min/max edad) y bins apropiados
    min_age = int(plot_df[age_col_name].min())
    max_age = int(plot_df[age_col_name].max())
    # Si el rango es pequeño y entero, usar bins por año; si es amplio, usar 20 bins
    if max_age - min_age <= 50:
        bins = list(range(min_age, max_age + 2))
    else:
        bins = 20

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    # Usar recuento (cantidad) en el eje Y y superponer con transparencia
    if not g1.empty:
        sns.histplot(g1, color='#f49a9a', label='Dropout', stat='count', kde=False, bins=bins, alpha=0.6, ax=ax)
    if not g2.empty:
        sns.histplot(g2, color='#6bdb9a', label='Graduate', stat='count', kde=False, bins=bins, alpha=0.6, ax=ax)
    plt.xlabel('Edad')
    plt.ylabel('Cantidad')
    plt.title('Comparación de Edades: Dropout vs Graduate')
    # Añadir leyenda con parches de color en la esquina
    legend_handles = []
    if not g1.empty:
        legend_handles.append(mpatches.Patch(color='#f49a9a', label='Dropout'))
    if not g2.empty:
        legend_handles.append(mpatches.Patch(color='#6bdb9a', label='Graduate'))
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', frameon=True)
    # Ajustar límites x para cubrir exactamente el rango de edades
    plt.xlim(min_age - 0.5, max_age + 0.5)

    # Marcar y anotar las medias de cada grupo
    ymin, ymax = ax.get_ylim()
    if not g1.empty:
        m1 = g1.mean()
        ax.axvline(m1, color='#f49a9a', linestyle='--', linewidth=2)
        ax.text(m1, ymax * 0.95, f'μ={m1:.2f}', color='#f49a9a', rotation=90, va='top', ha='right', backgroundcolor='white')
    if not g2.empty:
        m2 = g2.mean()
        ax.axvline(m2, color='#6bdb9a', linestyle='--', linewidth=2)
        ax.text(m2, ymax * 0.95, f'μ={m2:.2f}', color='#6bdb9a', rotation=90, va='top', ha='left', backgroundcolor='white')

    plt.tight_layout()
    out_file = out_dir / f'age_overlay_{safe_filename(age_col_name or "age")}.png'
    plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print('Histograma de edades guardado en:', out_file)

def plot_marital_status_pie(df, out_dir, target_col='Target', marital_col='Marital status', show=False):
    """
    Genera dos gráficos de pastel mostrando la proporción de estado civil
    dentro de los Graduados y de los Desertores.

    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        out_dir (pathlib.Path or str): Directorio donde guardar la imagen.
        target_col (str): Nombre de la columna objetivo/estado.
        marital_col (str): Nombre de la columna de estado civil.
        show (bool): Si True, muestra las figuras además de guardarlas.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mapeo de códigos numéricos a etiquetas descriptivas
    MARITAL_LABELS = {
        '1': 'Soltero/a',
        '2': 'Casado/a',
        '3': 'Viudo/a',
        '4': 'Divorciado/a',
        '5': 'Unión de hecho',
        '6': 'Separado/a',
        '1.0': 'Soltero/a',
        '2.0': 'Casado/a',
        '3.0': 'Viudo/a',
        '4.0': 'Divorciado/a',
        '5.0': 'Unión de hecho',
        '6.0': 'Separado/a',
    }

    def _classify_target(val):
        if pd.isna(val):
            return None
        if isinstance(val, (int, np.integer, float, np.floating)):
            if val == 1:
                return 'graduado'
            if val == 0:
                return 'abandonado'
        s = str(val).lower()
        if any(k in s for k in ('grad', 'success', 'aprob', 'pass', 'passed')):
            return 'graduado'
        if any(k in s for k in ('drop', 'aband', 'fail', 'deser', 'abandono')):
            return 'abandonado'
        return None

    def _map_marital(val):
        """Convierte código numérico a etiqueta descriptiva."""
        s = str(val).strip()
        return MARITAL_LABELS.get(s, s)

    # Localizar columnas objetivo y de estado civil si no existen exactamente con ese nombre
    if target_col in df.columns:
        targets = df[target_col].apply(_classify_target)
    else:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ('target', 'status', 'result'))]
        targets = df[candidates[0]].apply(_classify_target) if candidates else pd.Series([None] * len(df), index=df.index)

    if marital_col in df.columns:
        marital = df[marital_col].astype(str).fillna('Desconocido').apply(_map_marital)
    else:
        # intentar encontrar columnas parecidas
        candidates = [c for c in df.columns if any(k in c.lower() for k in ('marital', 'civil', 'estado'))]
        if candidates:
            marital = df[candidates[0]].astype(str).fillna('Desconocido').apply(_map_marital)
            marital_col = candidates[0]
        else:
            print(f'No se encontró columna de estado civil ({marital_col}).')
            return

    combined = pd.DataFrame({'target': targets, 'marital': marital})

    groups = {'Graduados': 'graduado', 'Desertores': 'abandonado'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    palette = sns.color_palette('pastel')

    any_data = False
    all_labels_for_legend = []
    all_colors_for_legend = []

    for ax, (title, label_val) in zip(axes, groups.items()):
        subset = combined[combined['target'] == label_val]['marital'].dropna()
        if subset.empty:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
            continue
        any_data = True
        counts = subset.value_counts()

        # Ordenar por código original para consistencia visual
        order = ['Soltero/a', 'Casado/a', 'Viudo/a', 'Divorciado/a', 'Unión de hecho', 'Separado/a']
        counts = counts.reindex([o for o in order if o in counts.index]).dropna()

        sizes = counts.values.tolist()
        cat_labels = counts.index.tolist()

        # Guardar para la leyenda compartida
        for i, lab in enumerate(cat_labels):
            if lab not in all_labels_for_legend:
                all_labels_for_legend.append(lab)
                all_colors_for_legend.append(palette[len(all_labels_for_legend) - 1])

        # Asignar colores consistentes según la categoría
        colors = [palette[all_labels_for_legend.index(lab)] for lab in cat_labels]

        # Dibujar pastel SIN etiquetas externas, solo con porcentajes dentro
        wedges, texts, autotexts = ax.pie(
            sizes,
            autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
            startangle=90,
            colors=colors,
            pctdistance=0.75,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
        )
        # Ajustar tamaño de fuente de porcentajes
        for autotext in autotexts:
            autotext.set_fontsize(13)
            autotext.set_fontweight('bold')

        ax.set_title(title, fontsize=18, fontweight='bold')

    if not any_data:
        print('No hay datos válidos de estado civil para Graduados ni Desertores.')
        plt.close(fig)
        return

    # Crear leyenda compartida debajo del gráfico
    # Reconstruir etiquetas con conteos totales
    legend_handles = [mpatches.Patch(color=all_colors_for_legend[i], label=lab)
                      for i, lab in enumerate(all_labels_for_legend)]
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(len(all_labels_for_legend), 6),
               fontsize=15, frameon=True, bbox_to_anchor=(0.5, 0.05))

    plt.suptitle('Distribución de Estado Civil por Resultado Académico', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.14, 1, 0.93])
    out_file = out_dir / f'pie_marital_status_{safe_filename(marital_col)}.png'
    plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print('Gráfico de estado civil guardado en:', out_file)

def main(input_path, out_dir, display=False):
    """
    Punto de entrada para generar figuras EDA a partir de nuestro CSV
    """
    # Aseguramos el directorio de salida
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lectura del CSV. 
    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)

    # Detectar columnas numéricas para decidir qué gráficos generar
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    # Generación de gráficos básicos. Se podrían añadir más llamadas
    generate_histograms(df, numeric, out_dir, show=display)
    generate_boxplots(df, numeric, out_dir, show=display)

    print('Gráficas generadas exitosamente en:', out_dir)

if __name__ == '__main__':
    """
    Ejecuta el script desde la línea de comandos.
    Parsea los argumentos de entrada y salida, y llama a la función principal.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=False, help='Path to CSV input', default=str(Path(__file__).parent.parent / 'outputs' / 'prepared_data' / 'dataset_prepared.csv'))
    p.add_argument('--out', required=False, help='Output directory', default=str(Path(__file__).parent.parent / 'outputs' / 'data_visualization'))
    p.add_argument('--display', action='store_true', help='Show plots in GUI windows in addition to saving them')
    args = p.parse_args()
    main(args.input, args.out, display=args.display)
