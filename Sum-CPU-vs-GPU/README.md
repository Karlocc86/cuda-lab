# Sum CPU vs GPU

Comparacion del tiempo de ejecucion de una suma de arreglos usando un ciclo en CPU y un kernel CUDA ejecutado con PyCUDA.

La operacion calculada es:

```text
c[i] = a[i] + b[i]
```

## Requisitos

- Python 3.12 o compatible
- GPU NVIDIA y controlador CUDA instalado
- Toolkit de CUDA, incluyendo `nvcc`
- PyCUDA
- NumPy

Instalacion de dependencias:

```powershell
python -m pip install numpy pycuda
```

## Ejecucion

Ejecuta el programa desde esta carpeta:

```powershell
python suma_gpu.py
```

El programa muestra los primeros resultados de CPU y GPU, verifica que sean iguales, calcula el tiempo de ejecucion y muestra el speedup.

## Nota de memoria

El programa crea dos arreglos de `1,000,000,000` elementos `float32`, por lo que requiere varios gigabytes de memoria RAM y memoria de video. Si el equipo no tiene suficiente memoria, reduce el valor de `N` antes de ejecutar.
