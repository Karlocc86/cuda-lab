import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import time
# -------------------------------------
# N = Size of the arrays
# a = array containing N random numbers
# b = array containing N random numbers
# c_cpu = result of a + b computed on the CPU
# c_gpu = result of a + b computed on the GPU
# -------------------------------------
N = 1000000000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c_cpu = np.zeros_like(a)
c_gpu = np.zeros_like(a)
# =============================
# Instructions computed in CPU
# =============================
start_cpu = time.time()
for i in range(N):
	c_cpu[i] = a[i] + b[i]
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
# =============================
# This is the KERNEL: Instructions computed in GPU
# =============================
mod = SourceModule("""
__global__ void suma_array_gpu(float *a, float *b, float *c, int n) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i < n) {
c[i] = a[i] + b[i];
}
}
""")
suma_array_gpu = mod.get_function("suma_array_gpu")
# -----------------------------------------
# 1. Allocate memory on the GPU
# -----------------------------------------
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu_mem = cuda.mem_alloc(c_gpu.nbytes)
# -----------------------------------------
# 2. Copy input data from CPU to GPU
# Host (CPU) --> Device (GPU)
# -----------------------------------------
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)
# ------------------------------------------------------------------
# Define the number of blocks and threads used to execute the kernel.
#
# A BLOCK is a group of threads that execute the same CUDA kernel.
# Threads within a block are identified by threadIdx.x.
# Blocks are identified by blockIdx.x.
#
# Example:
# If N = 1000 and block_size = 256:
#
# grid_size = ceil(1000 / 256) = 4 blocks
#
# Therefore, the GPU launches:
#
# 4 blocks × 256 threads = 1024 threads
#
# Each thread is assigned to process one element of the array.
# Therefore, each thread needs to determine which array element
# it is responsible for processing.
#
# Since threadIdx.x identifies a thread only within its block,
# we calculate a GLOBAL INDEX (i) to identify its corresponding
# position in the array:
#
# i = blockIdx.x * blockDim.x + threadIdx.x
#
# where:
#
# blockIdx.x = ID of the block
# blockDim.x = number of threads per block
# threadIdx.x = ID of the thread within the block
#
# Of the 1024 threads launched, only the first 1000 process
# array elements. The remaining 24 threads do not perform
# any computation.
#
# This is controlled inside the CUDA kernel with:
#
# if (i < n)
# c[i] = a[i] + b[i];
#
# Global thread indices:
#
# Block 0 -> Global indices 0 - 255
# Block 1 -> Global indices 256 - 511
# Block 2 -> Global indices 512 - 767
# Block 3 -> Global indices 768 - 1023
#
# ------------------------------------------------------------------
block_size = 256
grid_size = int(np.ceil(N / block_size))
# ---------------------------------------------------------
# 3. Execute the CUDA kernel on the GPU
# ---------------------------------------------------------
start_gpu = cuda.Event()
end_gpu = cuda.Event()
start_gpu.record()
suma_array_gpu(
a_gpu,
b_gpu,
c_gpu_mem,
np.int32(N),
block=(block_size, 1, 1),
grid=(grid_size, 1)
)
end_gpu.record()
end_gpu.synchronize()
# ---------------------------------------------------------
# 4. Calculate GPU execution time
# ---------------------------------------------------------
gpu_time = start_gpu.time_till(end_gpu) * 1e-3 # ms → segundos
# ---------------------------------------------------------
# 5. Once the GPU has completed the computation,
# copy the result from the GPU back to the CPU
# Device (GPU) --> Host (CPU)
# ---------------------------------------------------------
cuda.memcpy_dtoh(c_gpu, c_gpu_mem)
# =========================
# Outcome
# =========================
print(f"Number of N = {N}")
print("\nFirst 10 results CPU:")
print(c_cpu[:10])
print("\nFirst 10 results GPU:")
print(c_gpu[:10])
print("\n¿Equal Results?:", np.allclose(c_cpu, c_gpu))
print("\nTiempo CPU:", cpu_time, "seconds")
print("Tiempo GPU:", gpu_time, "seconds")
print("\nSpeedup:", cpu_time / gpu_time)
