"""
Script para generar estadísticas descriptivas por variable.

Descripción general:
- Lee un CSV con datos tabulares y calcula estadísticas básicas por columna.
- Para variables numéricas calcula: media, mediana, moda, mínimo, máximo,
    rango, varianza, desviación estándar y rango intercuartílico (IQR).
- Para variables categóricas calcula la(s) moda(s) y muestra los valores más
    frecuentes.
- Guarda un archivo de texto por variable y un `descriptive_stats_summary.csv`
    con un resumen para todas las variables en la carpeta de salida.

Notas de uso:
- Ejecutar como script desde la carpeta del proyecto o importarlo como módulo
    llamando a `compute_descriptive_stats(df, out_dir)`.
"""
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------
# Comentarios generales sobre imports:
# - `argparse` se usa para exponer una interfaz sencilla cuando se ejecuta
#   el script desde la línea de comandos.
# - `Path` (pathlib) facilita la manipulación de rutas multiplataforma.
# - `re` se usa para sanitizar nombres de ficheros.
# - `pandas` y `numpy` realizan la carga/transformación y los cálculos numéricos.
# ----------------------------------------------------------------------------------

def safe_filename(name: str) -> str:
        """Genera un nombre de fichero seguro reemplazando caracteres inválidos.

        Parámetros:
        - name: valor de entrada (normalmente el nombre de la variable) que puede
            contener caracteres no permitidos en sistemas de archivos (por ejemplo
            `:` o `\`).

        Retorna:
        - Una cadena con los caracteres inválidos sustituidos por guiones bajos.

        Observación: convierte el argumento a `str` antes de aplicar la expresión
        regular para evitar errores si se pasan tipos no string (p. ej. números).
        """
        return re.sub(r'[<>:"/\\|?*]', '_', str(name))

def compute_descriptive_stats(df: pd.DataFrame, out_dir: Path):
    """Itera por columnas y calcula estadísticas, guardando resultados.

    Comportamiento clave:
    - Crea el directorio de salida si no existe.
    - Para cada columna, elimina valores faltantes con `dropna()` antes de
      calcular medidas.
    - Si la serie resultante está vacía, se registran `NaN`/valores por defecto
      en el resumen para mantener consistencia de columnas en el CSV final.
    - Distingue entre columnas numéricas y categóricas usando
      `pd.api.types.is_numeric_dtype`.
    """
    # Asegurar que la carpeta de salida exista
    out_dir.mkdir(parents=True, exist_ok=True)

    # `rows` acumulará un diccionario por variable para convertirlo en DataFrame
    rows = []

    # Recorremos las columnas en el orden que presenta el DataFrame
    for col in df.columns:
        # `s` es la serie sin valores NA para cálculos limpios
        s = df[col].dropna()
        # Inicializamos el registro para esta variable
        result = {'variable': str(col)}

        # Caso: columna sin datos válidos
        if s.empty:
            # Mantenemos las mismas claves que para variables numéricas,
            # pero usando NaN o cadena vacía según el caso.
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
            # Si es numérico, calculamos todas las medidas relevantes
            if pd.api.types.is_numeric_dtype(s):
                # Convertimos a float para evitar objetos numpy en el CSV
                mean = float(s.mean())
                median = float(s.median())
                modes = s.mode().tolist()  # puede devolver múltiples modas
                minimo = float(s.min())
                maximo = float(s.max())
                rango = maximo - minimo
                # Varianza por defecto de pandas usa ddof=1 (muestra)
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

                # Salida humana por consola (útil para inspección rápida)
                print(f"Variable: {col}")
                print(f"media aritmética: {mean}")
                print(f"mediana: {median}")
                print(f"moda: {', '.join(map(str, modes))}")
                print(f"medidas de dispersión: minimo: {minimo}, maximo: {maximo}, rango: {rango}")
                print(f"varianza: {var}")
                print(f"desviación estándar: {std}")
                print(f"rango intercuartílico: {iqr}\n")

            else:
                # Para variables no numéricas (categóricas) calculamos la(s) moda(s)
                # y los valores más frecuentes (top 5) para dar contexto.
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

                # Impresión en consola indicando que las medidas numéricas no aplican
                print(f"Variable: {col}")
                print(f"media aritmética: N/A")
                print(f"mediana: N/A")
                print(f"moda: {', '.join(map(str, modes))}")
                print(f"medidas de dispersión: N/A")
                print(f"varianza: N/A")
                print(f"desviación estándar: N/A")
                print(f"rango intercuartílico: N/A")
                print(f"valores más frecuentes (hasta 5): {top}\n")

        # Añadimos el resultado acumulado para la columna actual
        rows.append(result)

        # Guardar archivo de texto por variable con la misma información
        # `safe_filename` asegura que no haya caracteres inválidos en el nombre.
        file_path = out_dir / f"{safe_filename(col)}_stats.txt"
        with file_path.open('w', encoding='utf-8') as fh:
            fh.write(f"Variable: {col}\n")
            # Escribimos las líneas de forma legible; para numéricas incluimos
            # todas las medidas, mientras que para otras escribimos N/A cuando
            # correspondan.
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

    # Convertimos la lista de dicts a DataFrame y la guardamos como CSV resumen.
    # Esto permite una revisión programática posterior (p. ej. para gráficos).
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
