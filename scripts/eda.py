"""
EDA script for the project dataset
Usage:
    python "d:/Cibernética/Statistics project/scripts/eda.py" --input "d:/Cibernética/Statistics project/dataset/dataset.csv" --out "d:/Cibernética/Statistics project/outputs/eda_outputs"

Generates:
- Descriptive stats CSV
- Missing values report
- Plots: histograms, boxplots, scatter for top correlations, correlation heatmap
- A short markdown summary with initial findings

Designed to be robust to varied datasets (infers numeric/categorical columns).
"""
import argparse
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def safe_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    # Replace problematic characters with underscore
    return re.sub(r'[<>:"/\\|?*]', '_', str(name))


def generate_histograms(df, numeric, out_dir):
    """Genera histogramas para todas las columnas numéricas."""
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
    """Genera boxplots para todas las columnas numéricas."""
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
    """Genera scatterplots para todos los pares de columnas numéricas."""
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    generate_histograms(df, numeric, out_dir)
    generate_boxplots(df, numeric, out_dir)
    generate_scatterplots(df, numeric, out_dir)

    print('Gráficas generadas en:', out_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=False, help='Path to CSV input', default=str(Path(__file__).parent.parent / 'dataset' / 'dataset.csv'))
    p.add_argument('--out', required=False, help='Output directory', default=str(Path(__file__).parent.parent / 'outputs' / 'eda_outputs'))
    args = p.parse_args()
    main(args.input, args.out)
