import numpy as np
import time
from PIL import Image
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class MeanFilter:
    def __init__(self):
        self.auto_context = pycuda.autoinit.context
        self._mod = None
        self._mean_filter_gpu = None
        self._compiled_mask_size = None

    def _compile_kernel(self, mask_size):
        kernel_code = f"""
        #define FILTER_SIZE {mask_size}
        #define FILTER_RADIUS {mask_size // 2}
        
        __global__ void mean_filter_gpu(
            unsigned char* input, unsigned char* output,
            int width, int height, int channels)
        {{
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) return;

            for (int c = 0; c < channels; c++) {{
                float sum = 0.0f;
                int count = 0;

                for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {{
                    for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {{
                        int nx = x + dx;
                        int ny = y + dy;

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {{
                            int idx = (ny * width + nx) * channels + c;
                            sum += input[idx];
                            count++;
                        }}
                    }}
                }}

                int out_idx = (y * width + x) * channels + c;
                output[out_idx] = (unsigned char)(sum / count);
            }}
        }}
        """
        self._mod = SourceModule(kernel_code)
        self._mean_filter_gpu = self._mod.get_function("mean_filter_gpu")
        self._compiled_mask_size = mask_size

    def process_cpu(self, img_np, mask_size):
        height, width, channels = img_np.shape
        pad = mask_size // 2
        padded = np.pad(img_np, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        output = np.zeros_like(img_np, dtype=np.float32)
        for c in range(channels):
            for dy in range(mask_size):
                for dx in range(mask_size):
                    output[:, :, c] += padded[dy:dy+height, dx:dx+width, c]
        output /= (mask_size * mask_size)
        return np.clip(output, 0, 255).astype(np.uint8)

    def process_gpu(self, img_np, mask_size):
        height, width, channels = img_np.shape
        output = np.zeros_like(img_np, dtype=np.uint8)
        block_dim = (16, 16, 1)
        grid_dim = ((width + 15) // 16, (height + 15) // 16, 1)

        self.auto_context.push()
        try:
            if self._mod is None or self._compiled_mask_size != mask_size:
                self._compile_kernel(mask_size)

            d_input = cuda.mem_alloc(img_np.nbytes)
            d_output = cuda.mem_alloc(output.nbytes)
            cuda.memcpy_htod(d_input, img_np)

            self._mean_filter_gpu(
                d_input, d_output,
                np.int32(width), np.int32(height), np.int32(channels),
                block=block_dim, grid=grid_dim
            )

            cuda.memcpy_dtoh(output, d_output)
        finally:
            self.auto_context.pop()

        return output

# Esta es la función que usarás en app.py
def process_image_mean_filter(img_np, mask_size, mode='gpu'):
    filter = MeanFilter()
    start = time.time()
    
    if mode == 'cpu':
        result = filter.process_cpu(img_np, mask_size)
    else:
        result = filter.process_gpu(img_np, mask_size)

    elapsed = time.time() - start
    stats = {
        'mode': mode.upper(),
        'mask_size': mask_size,
        'time_s': round(elapsed, 4)
    }
    if mode == 'gpu':
        stats.update({
            'threads': 256,
            'blocks': ((img_np.shape[0] * img_np.shape[1]) + 255) // 256
        })

    return result, stats
