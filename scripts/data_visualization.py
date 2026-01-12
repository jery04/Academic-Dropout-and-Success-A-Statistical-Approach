"""data_visualization.py
Script de EDA (Exploratory Data Analysis) para el proyecto.

Uso (ejemplo):
    python "d:/Cibernética/Statistics project/scripts/data_visualization.py" \
        --input "d:/Cibernética/Statistics project/dataset/dataset.csv" \
        --out "d:/Cibernética/Statistics project/outputs/data_visualization"

Salida esperada:
- CSV de estadísticas descriptivas (si se implementa)
- Informes de valores faltantes (si se implementa)
- Imágenes: histogramas, boxplots, scatterplots, gráficos comparativos
- Resumen breve (si se implementa)

El objetivo principal: generar gráficos automáticamente de forma robusta
cuando el conjunto de datos puede variar (detecta columnas numéricas,
intenta inferir columna objetivo, etc.).

He añadido comentarios en español a lo largo del script para aclarar
qué hace cada bloque y por qué se toman ciertas decisiones.
"""
import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliotecas:
# - argparse: parseo de argumentos de línea de comandos
# - re: sanitizar nombres de fichero (expresiones regulares)
# - pathlib.Path: manejo de rutas (multiplataforma)
# - pandas/numpy: manipulación de datos
# - matplotlib/seaborn: visualización

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

def generate_scatterplots(df, numeric, out_dir, show=False):
    """Genera y guarda scatterplots para pares de columnas numéricas.

    Si hay al menos dos columnas numéricas, crea la carpeta `scatterplots`
    dentro de `out_dir` y guarda una imagen por cada par único (a vs b).
    Los nombres de fichero usan `safe_filename` para evitar caracteres
    inválidos.

    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        numeric (list[str]): Lista de nombres de columnas numéricas.
        out_dir (pathlib.Path): Carpeta base donde guardar las imágenes.

    Returns:
        None: Los archivos se escriben en disco.
    """
    scatter_dir = out_dir / 'scatterplots'
    scatter_dir.mkdir(parents=True, exist_ok=True)

    # Genera scatterplots para cada par único de variables numéricas
    # (combinaciones sin repetición). Si hay muchas columnas esto puede
    # generar muchas imágenes; en datasets muy grandes podría filtrarse
    # por correlación alta para reducir el número de pares.
    if len(numeric) >= 2:
        for i in range(len(numeric)):
            for j in range(i + 1, len(numeric)):
                a, b = numeric[i], numeric[j]
                plt.figure(figsize=(6, 5))
                sns.scatterplot(x=df[a], y=df[b], alpha=0.6)
                plt.title(f'Scatter: {a} vs {b}')
                plt.tight_layout()
                safe_name = f'scatter_{safe_filename(a)}__vs__{safe_filename(b)}.png'
                plt.savefig(scatter_dir / safe_name, dpi=150)
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

def generate_overlaid_rendimiento_histogram(df, out_dir, target_col='Target', show=False):
    """Genera un histograma superpuesto de rendimiento para dos grupos: graduados y abandonos.

    Rendimiento se define como la media por fila de todas las columnas cuyo nombre
    contiene la cadena 'grade' (se espera que incluya notas de 1st y 2nd semester).

    El resultado se guarda en `out_dir/histograms/overlaid_rendimiento_<target_col>.png`.
    """
    # Usamos out_dir directamente para guardar el histograma agregado
    hist_dir = out_dir 
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Buscar columnas de nota (grade)
    # Buscamos columnas que contengan 'grade' (puede haber varias por semestre).
    # Si no hay tales columnas, como fallback usamos todas las numéricas.
    grade_cols = [c for c in df.columns if 'grade' in c.lower()]
    if not grade_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print('No se encontraron columnas de nota ni numéricas para calcular rendimiento.')
            return
        grade_cols = numeric_cols

    # Rendimiento por fila: media de columnas de nota (ignora NaN)
    rendimiento = df[grade_cols].mean(axis=1, skipna=True)

    # Reutilizamos lógica de clasificación para etiquetas objetivo
    def classify_target(val):
        if pd.isna(val):
            return None
        # Números: 1 => graduado, 0 => abandonado (convención común)
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

    # Identificamos la columna objetivo de la misma forma que en otras funciones
    if target_col in df.columns:
        labels = df[target_col].apply(classify_target)
    else:
        # Intentar encontrar una columna similar (target/status/result)
        candidates = [c for c in df.columns if any(k in c.lower() for k in ('target', 'status', 'result'))]
        if candidates:
            labels = df[candidates[0]].apply(classify_target)
        else:
            labels = pd.Series([None] * len(df), index=df.index)

    # Construimos un DataFrame temporal con rendimiento y etiqueta
    plot_df = pd.DataFrame({'rendimiento': rendimiento, 'label': labels})
    # Eliminamos filas sin rendimiento o sin etiqueta
    plot_df = plot_df.dropna(subset=['rendimiento', 'label'])

    if plot_df.empty:
        print('No hay datos suficientes para el histograma superpuesto (falta etiqueta o notas).')
        return

    # Dibujar histogramas superpuestos por etiqueta (densidad)
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=plot_df,
        x='rendimiento',
        hue='label',
        stat='density',
        element='step',
        common_norm=False,
        alpha=0.4,
        kde=False
    )
    plt.title('Rendimiento: Graduados vs Abandonos')
    plt.tight_layout()
    out_file = hist_dir / f'overlaid_rendimiento_{safe_filename(target_col)}.png'
    plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print('Histograma superpuesto guardado en:', out_file)

def generate_overlaid_rendimiento_boxplot(df, out_dir, target_col='Target', show=False):
    """Genera un boxplot de rendimiento comparando graduados y abandonos.

    Guarda el archivo en `out_dir/boxplots/overlaid_rendimiento_box_<target_col>.png`.
    """
    # Guardamos el boxplot en el directorio principal de salida (o se puede
    # cambiar a out_dir / 'boxplots' para mantener consistencia).
    box_dir = out_dir 
    box_dir.mkdir(parents=True, exist_ok=True)

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
                return 'graduado'
            if val == 0:
                return 'abandonado'
        s = str(val).lower()
        if any(k in s for k in ('grad', 'success', 'aprob', 'pass', 'passed')):
            return 'graduado'
        if any(k in s for k in ('drop', 'aband', 'fail', 'deser', 'abandono')):
            return 'abandonado'
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
        print('No hay datos suficientes para el boxplot superpuesto (falta etiqueta o notas).')
        return

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='label', y='rendimiento', data=plot_df)
    plt.title('Rendimiento (boxplot): Graduados vs Abandonos')
    plt.tight_layout()
    out_file = box_dir / f'overlaid_rendimiento_box_{safe_filename(target_col)}.png'
    plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print('Boxplot superpuesto guardado en:', out_file)

def generate_overlaid_rendimiento_scatter(df, out_dir, target_col='Target', show=False):
    """Genera un scatterplot superpuesto para comparar dos columnas de nota

    Si hay al menos dos columnas `grade` usa las dos primeras; en caso contrario
    usa `rendimiento` frente a la primera columna numérica disponible.
    Guarda el archivo en `out_dir/scatterplots/overlaid_rendimiento_scatter_<target_col>.png`.
    """
    # Directoriio de salida para scatterplots agregados
    scatter_dir = out_dir 
    scatter_dir.mkdir(parents=True, exist_ok=True)

    grade_cols = [c for c in df.columns if 'grade' in c.lower()]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Si existen al menos dos columnas de nota, las usamos como eje X e Y.
    # De lo contrario construimos un par con una columna numérica y el
    # rendimiento promedio por fila.
    if len(grade_cols) >= 2:
        x_col, y_col = grade_cols[0], grade_cols[1]
        x = df[x_col]
        y = df[y_col]
        xlabel, ylabel = x_col, y_col
    else:
        if grade_cols:
            rendimiento = df[grade_cols].mean(axis=1, skipna=True)
        else:
            if numeric_cols:
                rendimiento = df[numeric_cols[0]]
            else:
                print('No hay columnas numéricas para generar scatterplot.')
                return
        candidates = [c for c in numeric_cols if c not in grade_cols]
        if not candidates:
            print('No hay pares de columnas numéricas para scatterplot.')
            return
        x_col = candidates[0]
        x = df[x_col]
        y = rendimiento
        xlabel, ylabel = x_col, 'rendimiento'

    # Clasificación de la etiqueta objetivo para colorear puntos
    def classify_target(val):
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

    if target_col in df.columns:
        labels = df[target_col].apply(classify_target)
    else:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ('target', 'status', 'result'))]
        if candidates:
            labels = df[candidates[0]].apply(classify_target)
        else:
            labels = pd.Series([None] * len(df), index=df.index)

    # DataFrame para graficar: columnas seleccionadas + etiqueta
    plot_df = pd.DataFrame({xlabel: x, ylabel: y, 'label': labels})
    plot_df = plot_df.dropna(subset=[xlabel, ylabel, 'label'])

    if plot_df.empty:
        print('No hay datos suficientes para el scatterplot superpuesto (falta etiqueta o valores).')
        return

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df, x=xlabel, y=ylabel, hue='label', alpha=0.6)
    plt.title(f'Scatter: {xlabel} vs {ylabel} (Graduados vs Abandonos)')
    plt.tight_layout()
    out_file = scatter_dir / f'overlaid_rendimiento_scatter_{safe_filename(target_col)}.png'
    plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print('Scatterplot superpuesto guardado en:', out_file)

def main(input_path, out_dir, display=False):
    """Punto de entrada para generar figuras EDA a partir de un CSV.

    Lee el fichero CSV indicado por `input_path`, detecta las columnas
    numéricas y llama a las funciones que generan histogramas, boxplots
    y scatterplots guardando los resultados en `out_dir`.

    Args:
        input_path (str or pathlib.Path): Ruta al fichero CSV de entrada.
        out_dir (str or pathlib.Path): Ruta al directorio donde escribir
            los resultados y subcarpetas con imágenes.

    Returns:
        None
    """
    # Aseguramos el directorio de salida
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lectura del CSV. `low_memory=False` evita advertencias sobre tipos al
    # concatenar fragmentos; `encoding='utf-8'` es la opción estándar.
    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)

    # Detectar columnas numéricas para decidir qué gráficos generar
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    # Generación de gráficos básicos. Se podrían añadir más llamadas
    # (piechart, overlaid_rendimiento, etc.) según se requiera.
    generate_histograms(df, numeric, out_dir, show=display)
    generate_boxplots(df, numeric, out_dir, show=display)
    generate_scatterplots(df, numeric, out_dir, show=display)

    print('Gráficas generadas exitosamente en:', out_dir)

if __name__ == '__main__':
    """Ejecuta el script desde la línea de comandos.

    Parsea los argumentos de entrada y salida, y llama a la función principal.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=False, help='Path to CSV input', default=str(Path(__file__).parent.parent / 'dataset' / 'dataset.csv'))
    p.add_argument('--out', required=False, help='Output directory', default=str(Path(__file__).parent.parent / 'outputs' / 'data_visualization'))
    p.add_argument('--display', action='store_true', help='Show plots in GUI windows in addition to saving them')
    args = p.parse_args()
    main(args.input, args.out, display=args.display)
