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

def generate_histograms(df, numeric, out_dir):
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
        plt.close()

def generate_boxplots(df, numeric, out_dir):
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
        plt.close()

def generate_scatterplots(df, numeric, out_dir):
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
                plt.close()

def main(input_path, out_dir):
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

    generate_histograms(df, numeric, out_dir)
    generate_boxplots(df, numeric, out_dir)
    generate_scatterplots(df, numeric, out_dir)

    print('Gráficas generadas en:', out_dir)

if __name__ == '__main__':
    """Ejecuta el script desde la línea de comandos.

    Parsea los argumentos de entrada y salida, y llama a la función principal.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=False, help='Path to CSV input', default=str(Path(__file__).parent.parent / 'dataset' / 'dataset.csv'))
    p.add_argument('--out', required=False, help='Output directory', default=str(Path(__file__).parent.parent / 'outputs' / 'data_visualization'))
    args = p.parse_args()
    main(args.input, args.out)
