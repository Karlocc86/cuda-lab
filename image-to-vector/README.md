# Image to Vector

Procesamiento de una imagen en escala de grises usando CUDA. El programa convierte la imagen en un arreglo unidimensional, ejecuta en la GPU la operacion `out[i] = 255 - img[i]` y reconstruye la imagen transformada.

## Requisitos

- Python 3.12 o compatible
- GPU NVIDIA con controlador CUDA instalado
- Dependencias del archivo `requirements.txt`

Instalacion:

```powershell
python -m pip install -r requirements.txt
```

## Ejecucion

Ejecuta el programa desde esta carpeta:

```powershell
python imageProcessor.py
```

El programa muestra el arreglo de entrada y el arreglo transformado en la consola. La imagen resultante se guarda como:

```text
imagen-transformada.jpg
```

La operacion transforma cada pixel mediante la inversion de escala de grises: los pixeles oscuros se vuelven claros y los claros se vuelven oscuros.
