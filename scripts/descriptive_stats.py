"""
Script para generar estadísticas descriptivas por variable.

Genera en consola (en español) para cada variable:
- media aritmética
- mediana
- moda
- medidas de dispersión (mínimo, máximo, rango)
- varianza
- desviación estándar
- rango intercuartílico

Guarda un CSV resumen y un archivo de texto por variable dentro de outputs/descriptive_stats.
"""
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np


def safe_filename(name: str) -> str:
        """Genera un nombre de fichero seguro reemplazando caracteres inválidos.

        Parámetros:
        - name: valor de entrada (normalmente el nombre de la variable) que puede
            contener caracteres no permitidos en sistemas de archivos.

        Retorna:
        - Una cadena con los caracteres inválidos sustituidos por guiones bajos.
        """
        return re.sub(r'[<>:"/\\|?*]', '_', str(name))


def compute_descriptive_stats(df: pd.DataFrame, out_dir: Path):
    """Calcula y guarda estadísticas descriptivas para cada columna del DataFrame.

    Para cada columna en `df` se calculan medidas principales (media, mediana,
    moda, mínimo, máximo, rango, varianza, desviación estándar y rango
    intercuartílico cuando aplica). Produce en consola un resumen en español
    y guarda un archivo de texto por variable más un CSV resumen en `out_dir`.

    Parámetros:
    - df: DataFrame de pandas con los datos de entrada.
    - out_dir: Path al directorio donde se guardarán los archivos de salida.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for col in df.columns:
        s = df[col].dropna()
        # Prepare a dict of results; handle non-numeric gracefully
        result = {'variable': str(col)}

        if s.empty:
            result.update({
                'media': np.nan,
                'mediana': np.nan,
                'moda': '',
                'minimo': np.nan,
                'maximo': np.nan,
                'rango': np.nan,
                'varianza': np.nan,
                'desviacion_estandar': np.nan,
                'rango_intercuartilico': np.nan,
            })

            print(f"Variable: {col}")
            print("(sin datos válidos)\n")

        else:
            # If numeric, compute full set; otherwise compute mode and simple dispersion
            if pd.api.types.is_numeric_dtype(s):
                mean = float(s.mean())
                median = float(s.median())
                modes = s.mode().tolist()
                minimo = float(s.min())
                maximo = float(s.max())
                rango = maximo - minimo
                var = float(s.var())
                std = float(s.std())
                q75 = float(s.quantile(0.75))
                q25 = float(s.quantile(0.25))
                iqr = q75 - q25

                result.update({
                    'media': mean,
                    'mediana': median,
                    'moda': ';'.join(map(str, modes)),
                    'minimo': minimo,
                    'maximo': maximo,
                    'rango': rango,
                    'varianza': var,
                    'desviacion_estandar': std,
                    'rango_intercuartilico': iqr,
                })

                # Imprimir en consola en español, línea por línea
                print(f"Variable: {col}")
                print(f"media aritmética: {mean}")
                print(f"mediana: {median}")
                print(f"moda: {', '.join(map(str, modes))}")
                print(f"medidas de dispersión: minimo: {minimo}, maximo: {maximo}, rango: {rango}")
                print(f"varianza: {var}")
                print(f"desviación estándar: {std}")
                print(f"rango intercuartílico: {iqr}\n")

            else:
                # Categorical / non-numeric: compute mode and counts
                modes = s.mode().tolist()
                top = s.value_counts().head(5).to_dict()

                result.update({
                    'media': np.nan,
                    'mediana': np.nan,
                    'moda': ';'.join(map(str, modes)),
                    'minimo': np.nan,
                    'maximo': np.nan,
                    'rango': np.nan,
                    'varianza': np.nan,
                    'desviacion_estandar': np.nan,
                    'rango_intercuartilico': np.nan,
                })

                print(f"Variable: {col}")
                print(f"media aritmética: N/A")
                print(f"mediana: N/A")
                print(f"moda: {', '.join(map(str, modes))}")
                print(f"medidas de dispersión: N/A")
                print(f"varianza: N/A")
                print(f"desviación estándar: N/A")
                print(f"rango intercuartílico: N/A")
                print(f"valores más frecuentes (hasta 5): {top}\n")

        rows.append(result)

        # Guardar archivo de texto por variable
        file_path = out_dir / f"{safe_filename(col)}_stats.txt"
        with file_path.open('w', encoding='utf-8') as fh:
            fh.write(f"Variable: {col}\n")
            # Write the same human-readable lines
            if pd.api.types.is_numeric_dtype(s) and not s.empty:
                fh.write(f"media aritmética: {result['media']}\n")
                fh.write(f"mediana: {result['mediana']}\n")
                fh.write(f"moda: {result['moda']}\n")
                fh.write(f"medidas de dispersión: minimo: {result['minimo']}, maximo: {result['maximo']}, rango: {result['rango']}\n")
                fh.write(f"varianza: {result['varianza']}\n")
                fh.write(f"desviación estándar: {result['desviacion_estandar']}\n")
                fh.write(f"rango intercuartílico: {result['rango_intercuartilico']}\n")
            else:
                fh.write(f"media aritmética: N/A\n")
                fh.write(f"mediana: N/A\n")
                fh.write(f"moda: {result['moda']}\n")

    # Guardar resumen CSV
    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / 'descriptive_stats_summary.csv'
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8')


def main(input_path: str, out: str):
    """Punto de entrada para ejecutar estadísticas descriptivas desde CSV.

    Lee un archivo CSV desde `input_path`, crea el directorio de salida
    apropiado y llama a `compute_descriptive_stats` para generar los informes.

    Parámetros:
    - input_path: ruta al archivo CSV de entrada (cadena o Path convertible).
    - out: ruta al directorio de salida donde guardar los resultados.
    """
    input_path = Path(input_path)
    out_dir = Path(out)
    # Colocar los resultados dentro de outputs/descriptive_stats por defecto
    if out_dir.name == 'outputs' or 'outputs' in str(out_dir):
        target = out_dir / 'descriptive_stats'
    else:
        target = out_dir

    target.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
    compute_descriptive_stats(df, target)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=False, help='Path to CSV input', default=str(Path(__file__).parent.parent / 'dataset' / 'dataset.csv'))
    p.add_argument('--out', required=False, help='Output directory', default=str(Path(__file__).parent.parent / 'outputs'))
    args = p.parse_args()
    main(args.input, args.out)
