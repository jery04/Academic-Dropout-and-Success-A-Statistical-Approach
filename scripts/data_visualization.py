"""
EDA script for the project dataset
Usage:
    python "d:/Cibernética/Statistics project/scripts/data_visualization.py" --input "d:/Cibernética/Statistics project/dataset/dataset.csv" --out "d:/Cibernética/Statistics project/outputs/data_visualization"

Generates:
- Descriptive stats CSV
- Missing values report
- Plots: histograms, boxplots, scatter for top correlations, correlation heatmap
- A short markdown summary with initial findings

Designed to be robust to varied datasets (infers numeric/categorical columns).
"""
import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Replace problematic characters with underscore
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
    
    for c in numeric:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[c].dropna(), kde=True)
        plt.title(f'Histogram: {c}')
        plt.tight_layout()
        plt.savefig(hist_dir / f'hist_{safe_filename(c)}.png', dpi=150)
        if show:
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


def generate_overlaid_rendimiento_histogram(df, out_dir, target_col='Target', show=False):
    """Genera un histograma superpuesto de rendimiento para dos grupos: graduados y abandonos.

    Rendimiento se define como la media por fila de todas las columnas cuyo nombre
    contiene la cadena 'grade' (se espera que incluya notas de 1st y 2nd semester).

    El resultado se guarda en `out_dir/histograms/overlaid_rendimiento_<target_col>.png`.
    """
    hist_dir = out_dir 
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Buscar columnas de nota (grade)
    grade_cols = [c for c in df.columns if 'grade' in c.lower()]
    if not grade_cols:
        # Fallback: usar todas las columnas numéricas si no hay columnas con 'grade'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print('No se encontraron columnas de nota ni numéricas para calcular rendimiento.')
            return
        grade_cols = numeric_cols

    rendimiento = df[grade_cols].mean(axis=1, skipna=True)

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

    if target_col in df.columns:
        labels = df[target_col].apply(classify_target)
    else:
        # Intentar encontrar una columna similar (target/status/result)
        candidates = [c for c in df.columns if any(k in c.lower() for k in ('target', 'status', 'result'))]
        if candidates:
            labels = df[candidates[0]].apply(classify_target)
        else:
            labels = pd.Series([None] * len(df), index=df.index)

    plot_df = pd.DataFrame({'rendimiento': rendimiento, 'label': labels})
    plot_df = plot_df.dropna(subset=['rendimiento', 'label'])

    if plot_df.empty:
        print('No hay datos suficientes para el histograma superpuesto (falta etiqueta o notas).')
        return

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
    scatter_dir = out_dir 
    scatter_dir.mkdir(parents=True, exist_ok=True)

    grade_cols = [c for c in df.columns if 'grade' in c.lower()]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()

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
