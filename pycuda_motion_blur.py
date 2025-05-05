import numpy as np
import math
import time
from scipy.signal import convolve2d
import numpy as np
from PIL import Image

import pycuda.autoinit
from pycuda import driver as drv
from pycuda.compiler import SourceModule
from numba import jit

# Referencia al contexto creado por autoinit
auto_context = pycuda.autoinit.context

mod = SourceModule("""
            __global__ void motion_blur_45(float *img, float *out, float *mask, int width, int height, int channels, int mask_size)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total_pixels = width * height;

                if (idx < total_pixels)
                {
                    int x = idx % width;
                    int y = idx / width;
                    int half = mask_size / 2;

                    for (int c = 0; c < channels; ++c)
                    {
                        float sum = 0.0;
                        for (int k = -half; k <= half; ++k)
                        {
                            int new_x = x + k;
                            int new_y = y + k;

                            if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height)
                            {
                                sum += img[(new_y * width + new_x) * channels + c] * mask[k + half];
                            }
                        }
                        out[(y * width + x) * channels + c] = sum;
                    }
                }
            }
""")

def create_diagonal_kernel(kernel_size):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(kernel_size):
        kernel[i, i] = 1.0
    kernel /= kernel_size
    return kernel

@jit(nopython=True, parallel=False)
def motion_blur_cpu(input_image, kernel):
    height, width, channels = input_image.shape
    kernel_size = kernel.shape[0]
    offset = kernel_size // 2

    output_image = np.zeros_like(input_image, dtype=np.float32)

    # Loop por todos los pixeles
    for y in range(height):  
        for x in range(width):
            sum_r = 0.0
            sum_g = 0.0
            sum_b = 0.0

            for ky in range(-offset, offset + 1):
                for kx in range(-offset, offset + 1):
                    nx = x + kx
                    ny = y + ky

                    # Verificar si las coordenadas existen en el tamaño de pixeles
                    if 0 <= nx < width and 0 <= ny < height:
                        kernel_idx = (ky + offset) * kernel_size + (kx + offset)

                        sum_r += input_image[ny, nx, 0] * kernel[ky + offset, kx + offset]  # Canal R
                        sum_g += input_image[ny, nx, 1] * kernel[ky + offset, kx + offset]  # Canal G
                        sum_b += input_image[ny, nx, 2] * kernel[ky + offset, kx + offset]  # Canal B

            output_image[y, x, 0] = max(0, min(255, sum_r))
            output_image[y, x, 1] = max(0, min(255, sum_g))
            output_image[y, x, 2] = max(0, min(255, sum_b))

    return output_image

def process_image_motion_blur(img_np: np.ndarray, mask_size: int, mode: str):
    h, w, c = img_np.shape
    total_pixels = w * h

    stats = {'mask_size': mask_size}

    if mode.lower() == 'cpu':
        start = time.time()
        output_cpu = motion_blur_cpu(img_np, create_diagonal_kernel(mask_size).astype(np.float32))
        elapsed = time.time() - start
        stats.update({'mode': 'CPU', 'time_s': elapsed})
        return np.clip(output_cpu, 0, 255).astype(np.uint8), stats

    else:
        # GPU con PyCUDA
        threads_per_block = 1024
        blocks = (total_pixels + threads_per_block - 1) // threads_per_block
        stats.update({'mode': 'GPU', 'threads': threads_per_block, 'blocks': blocks})

        # Preparar memoria de salida
        output = np.zeros_like(img_np, dtype=np.float32)

        mask_kernel = create_diagonal_kernel(mask_size)
        mask = np.diag(mask_kernel).astype(np.float32)  # extrae solo la diagonal

        # Aplanar imagen para tratarla como float
        flat_img = img_np.astype(np.float32).flatten()
        flat_out = output.flatten()

        auto_context.push()
        try:
            # Asignar memoria GPU
            d_in = drv.mem_alloc(flat_img.nbytes)
            d_out = drv.mem_alloc(flat_out.nbytes)
            d_mask = drv.mem_alloc(mask.nbytes)

            # Copiar datos
            drv.memcpy_htod(d_in, flat_img)
            drv.memcpy_htod(d_mask, mask)

            func = mod.get_function("motion_blur_45")

            # Medir tiempo GPU
            start_evt = drv.Event(); end_evt = drv.Event()
            start_evt.record()

            func(d_in, d_out, d_mask,
                 np.int32(w), np.int32(h), np.int32(c), np.int32(mask_size),
                 block=(threads_per_block, 1, 1), grid=(blocks, 1))

            end_evt.record()
            end_evt.synchronize()
            gpu_time = end_evt.time_since(start_evt) / 1000.0
            stats['time_s'] = gpu_time

            # Copiar resultado de vuelta a CPU
            drv.memcpy_dtoh(flat_out, d_out)
            output = flat_out.reshape((h, w, c))

        finally:
            auto_context.pop()

        return np.clip(output, 0, 255).astype(np.uint8), stats
