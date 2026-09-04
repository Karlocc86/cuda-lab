import cv2
import numpy as np
from numba import cuda


@cuda.jit
def invertir_imagen(img, out):
	i = cuda.grid(1)
	if i < img.size:
		out[i] = 255 - img[i]


img = cv2.imread('imagen-escala-de-grises.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
	raise FileNotFoundError('No se encontró imagen-escala-de-grises.jpg')

img_flat = img.flatten().astype(np.uint8)
print('Arreglo de entrada:')
print(img_flat)

try:
	cuda.select_device(0)
except Exception as error:
	raise RuntimeError(
		'No se pudo inicializar el dispositivo CUDA.'
	) from error

dst_flat = np.empty_like(img_flat)
threads_per_block = 256
blocks_per_grid = (img_flat.size + threads_per_block - 1) // threads_per_block

d_img = cuda.to_device(img_flat)
d_dst = cuda.device_array_like(img_flat)
invertir_imagen[blocks_per_grid, threads_per_block](d_img, d_dst)
cuda.synchronize()
d_dst.copy_to_host(dst_flat)

print('Arreglo transformado:')
print(dst_flat)

imagen_transformada = dst_flat.reshape(img.shape)
archivo_salida = 'imagen-transformada.jpg'
if not cv2.imwrite(archivo_salida, imagen_transformada):
	raise IOError(f'No se pudo guardar {archivo_salida}')

print(f'Imagen transformada guardada en: {archivo_salida}')


